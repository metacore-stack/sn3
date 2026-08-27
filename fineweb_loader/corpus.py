"""The evaluation corpus: several sources combined in fixed proportions.

Until 2026-08-27 SN3 evaluated on FineWeb-Edu alone. It now draws each
evaluation from three corpora at published proportions::

    finewebedu          0.22      1.57 T tokens
    automathtext-v2     0.26      1.90 T tokens
    dclm-baseline-1.0   0.52      3.73 T tokens

An ``n=2000`` evaluation is therefore 440 + 520 + 1040 sequences, each drawn
from its own corpus. Training on one source alone is now training on 22% of the
score, and the live history shows exactly what that costs: challengers built for
the old contract scored between -0.5 and -1.05 within hours of the change.

All three use the same tokenizer, the same ``uint32`` dtype, the same 2048
sequence length and the same 128-byte ``.npy`` header, so everything below the
manifest layer is shared.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from .cache import DEFAULT_BUDGET_BYTES, ShardCache
from .errors import LoaderError, ManifestError, ShardNotFoundError
from .manifest import ShardEntry, ShardManifest, USER_AGENT

DATASET_CONFIG_URL = "https://teutonic.ai/datasets/manifest.json"
DATASET_CONFIG_MIRROR = (
    "https://pub-fedac496355c4edc9aed57189e6e190f.r2.dev/datasets/manifest.json"
)


def source_targets(total: int, proportions: Sequence[float]) -> list[int]:
    """Split ``total`` across sources exactly as the validator does.

    Transcribed from ``teutonic/evaluation/configuration.py:_source_targets``:
    floor each share, then hand the remainder to the largest fractional parts,
    ties broken by index. Largest-remainder apportionment -- a naive ``round()``
    gives different counts and would make an offline holdout subtly
    unrepresentative.
    """
    raw = [total * value for value in proportions]
    targets = [int(value) for value in raw]
    remainder = total - sum(targets)
    order = sorted(
        range(len(raw)), key=lambda index: (-(raw[index] - targets[index]), index)
    )
    for index in order[:remainder]:
        targets[index] += 1
    return targets


@dataclass(frozen=True)
class SourceSpec:
    """One corpus as declared by the live dataset configuration."""

    name: str
    proportion: float
    manifest_url: str
    manifest_sha256: str
    tokenizer: str | None = None
    sequence_length: int | None = None
    dtype: str | None = None
    total_shards: int | None = None
    total_tokens: int | None = None

    @property
    def base_url(self) -> str:
        """Bucket prefix for this corpus, derived from its manifest URL.

        ``.../automathtext-v2/manifest.json`` -> ``.../automathtext-v2``. Shard
        keys are relative to that, which is why deriving the base beats
        reconstructing it from ``shard_prefix``.
        """
        return self.manifest_url.rsplit("/", 1)[0]

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> "SourceSpec":
        return cls(
            name=str(entry["name"]),
            proportion=float(entry.get("proportion", 0.0)),
            manifest_url=str(entry["manifest_url"]),
            manifest_sha256=str(entry.get("manifest_sha256", "")),
            tokenizer=entry.get("tokenizer"),
            sequence_length=entry.get("sequence_length"),
            dtype=entry.get("dtype"),
            total_shards=entry.get("total_shards"),
            total_tokens=entry.get("total_tokens"),
        )


@dataclass(frozen=True)
class DatasetConfig:
    """The live evaluation data contract."""

    config_version: str
    dataset_label: str
    delta_threshold: float
    eval_n: int
    sources: tuple[SourceSpec, ...]
    generated_at: str | None = None
    sampling: dict[str, Any] = field(default_factory=dict)
    source_url: str | None = None

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.sources]

    @property
    def proportions(self) -> list[float]:
        return [s.proportion for s in self.sources]

    def targets(self, n: int | None = None) -> dict[str, int]:
        """Sequences drawn per corpus for an evaluation of ``n``."""
        counts = source_targets(n or self.eval_n, self.proportions)
        return dict(zip(self.names, counts))

    def check(self) -> list[str]:
        """Internal consistency problems worth refusing to proceed on."""
        problems: list[str] = []
        total = sum(self.proportions)
        if abs(total - 1.0) > 1e-6:
            problems.append(f"proportions sum to {total}, not 1.0")
        tokenizers = {s.tokenizer for s in self.sources if s.tokenizer}
        if len(tokenizers) > 1:
            problems.append(f"sources disagree on tokenizer: {sorted(tokenizers)}")
        lengths = {s.sequence_length for s in self.sources if s.sequence_length}
        if len(lengths) > 1:
            problems.append(f"sources disagree on sequence_length: {sorted(lengths)}")
        if not self.sources:
            problems.append("no sources declared")
        return problems

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], *, source_url: str | None = None
    ) -> "DatasetConfig":
        sources = tuple(
            SourceSpec.from_entry(e)
            for e in (payload.get("sources") or [])
            if isinstance(e, dict)
        )
        return cls(
            config_version=str(payload.get("config_version", "")),
            dataset_label=str(payload.get("dataset_label", "")),
            delta_threshold=float(payload.get("delta_threshold", 0.0)),
            eval_n=int(payload.get("eval_n", 0)),
            sources=sources,
            generated_at=payload.get("generated_at"),
            sampling=dict(payload.get("sampling") or {}),
            source_url=source_url,
        )

    @classmethod
    def fetch(cls, url: str = DATASET_CONFIG_URL, *, timeout: float = 60.0) -> "DatasetConfig":
        last: Exception | None = None
        for candidate in (url, DATASET_CONFIG_MIRROR):
            try:
                request = urllib.request.Request(
                    candidate, headers={"User-Agent": USER_AGENT}
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return cls.from_payload(payload, source_url=candidate)
            except Exception as exc:  # noqa: BLE001
                last = exc
        raise ManifestError(f"could not fetch the dataset configuration: {last}")

    @classmethod
    def load(cls, path: Path) -> "DatasetConfig":
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ManifestError(f"{path} does not exist; run 'corpus sync'") from exc
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path} is not valid JSON: {exc}") from exc
        return cls.from_payload(payload, source_url=str(path))

    def save(self, path: Path, payload: dict[str, Any]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path


@dataclass
class Corpus:
    """One source, with its inventory and its own cache directory."""

    spec: SourceSpec
    manifest: ShardManifest
    cache: ShardCache

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def proportion(self) -> float:
        return self.spec.proportion

    def verify(self) -> list[str]:
        problems = self.manifest.verify(
            expected_digest=self.spec.manifest_sha256 or None,
            expected_seq_len=self.spec.sequence_length,
            expected_tokenizer=self.spec.tokenizer,
            expected_dtype=self.spec.dtype,
        )
        return [f"{self.name}: {p}" for p in problems]


class CorpusSet:
    """Every evaluation corpus, addressed as one collection.

    Shard names carry their corpus as a prefix (``finewebedu__…``,
    ``dclm-baseline-1.0__…``), so a bare shard name resolves without needing the
    caller to say which source it came from.
    """

    def __init__(self, config: DatasetConfig, corpora: Sequence[Corpus]):
        self.config = config
        self.corpora: dict[str, Corpus] = {c.name: c for c in corpora}
        if not self.corpora:
            raise LoaderError("a corpus set needs at least one corpus")

    # -- construction ------------------------------------------------------

    @classmethod
    def open(
        cls,
        config: DatasetConfig,
        root: Path,
        *,
        budget_bytes: int = DEFAULT_BUDGET_BYTES,
        only: Sequence[str] | None = None,
    ) -> "CorpusSet":
        """Load every source's manifest from ``root/manifests/<name>.json``."""
        root = Path(root)
        corpora: list[Corpus] = []
        wanted = set(only) if only else None
        for spec in config.sources:
            if wanted is not None and spec.name not in wanted:
                continue
            path = root / "manifests" / f"{spec.name}.json"
            if not path.is_file():
                raise ManifestError(
                    f"{path} missing; run 'fineweb corpus sync' to download "
                    f"the {spec.name} inventory"
                )
            # base_url must come from the source spec: without it the manifest
            # falls back to reconstructing the prefix from shard_prefix, and the
            # corpus name ends up in the URL twice.
            manifest = ShardManifest.load(path, base_url=spec.base_url)
            # Each corpus caches under its own name so budgets and eviction do
            # not interfere across sources. bucket_root stays at the default so
            # the manifest's own base_url is what resolves shard URLs.
            cache = ShardCache(
                root / "cache" / spec.name,
                manifest,
                budget_bytes=budget_bytes,
            )
            corpora.append(Corpus(spec=spec, manifest=manifest, cache=cache))
        return cls(config, corpora)

    # -- access ------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.config.sources if s.name in self.corpora]

    def __getitem__(self, name: str) -> Corpus:
        if name not in self.corpora:
            raise ShardNotFoundError(f"no corpus {name!r}; have {self.names}")
        return self.corpora[name]

    def __iter__(self) -> Iterator[Corpus]:
        for name in self.names:
            yield self.corpora[name]

    def __len__(self) -> int:
        return len(self.corpora)

    def corpus_of(self, shard_name: str) -> Corpus:
        """Resolve a bare shard name to its corpus via the filename prefix."""
        prefix = shard_name.rsplit("/", 1)[-1].split("__", 1)[0]
        if prefix in self.corpora:
            return self.corpora[prefix]
        for corpus in self.corpora.values():
            try:
                corpus.manifest.lookup(shard_name)
                return corpus
            except ShardNotFoundError:
                continue
        raise ShardNotFoundError(
            f"{shard_name!r} belongs to no loaded corpus ({self.names})"
        )

    def lookup(self, shard_name: str) -> tuple[Corpus, ShardEntry]:
        corpus = self.corpus_of(shard_name)
        return corpus, corpus.manifest.lookup(shard_name)

    def targets(self, n: int | None = None) -> dict[str, int]:
        return self.config.targets(n)

    # -- verification ------------------------------------------------------

    def verify(self) -> list[str]:
        problems = list(self.config.check())
        for corpus in self:
            problems.extend(corpus.verify())
        missing = [s.name for s in self.config.sources if s.name not in self.corpora]
        if missing:
            problems.append(
                f"not loaded: {missing} — evaluation draws from these, so any "
                "offline measurement without them is unrepresentative"
            )
        return problems

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"sources": {}, "config_version": self.config.config_version}
        total_shards = total_tokens = total_bytes = 0
        for corpus in self:
            s = corpus.manifest.stats()
            out["sources"][corpus.name] = {
                "proportion": corpus.proportion,
                "shards": s.total_shards,
                "tokens": s.total_tokens,
                "sequences": s.total_sequences,
                "bytes": s.total_bytes,
                "cached": len(corpus.cache.entries()),
            }
            total_shards += s.total_shards
            total_tokens += s.total_tokens
            total_bytes += s.total_bytes
        out["total"] = {
            "shards": total_shards,
            "tokens": total_tokens,
            "bytes": total_bytes,
        }
        out["targets"] = self.targets()
        return out


# -- blended holdouts -------------------------------------------------------


def build_blended_holdout(
    corpora: CorpusSet,
    *,
    name: str,
    seed: int,
    total: int | None = None,
    per_shard: int = 128,
    exclude: Sequence[Any] = (),
    full_shards_only: bool = True,
    min_sequences: int | None = None,
    notes: Sequence[str] = (),
):
    """A holdout that mirrors the validator's corpus proportions.

    A FineWeb-Edu-only holdout now measures 22% of the score. This allocates
    ``total`` across sources with the same largest-remainder split the validator
    uses, then stratifies across groups inside each corpus.

    Returns a single :class:`~fineweb_loader.refs.SequenceSet`; shard names carry
    their corpus prefix, so the blend can be split apart again for per-corpus
    reporting.
    """
    from .refs import SequenceSet

    total = total or corpora.config.eval_n or 2000
    allocation = corpora.targets(total)
    banned: list[Any] = list(exclude)
    parts: list[SequenceSet] = []

    for corpus in corpora:
        want = allocation.get(corpus.name, 0)
        if want <= 0:
            continue
        # Spread across as many shards as ``per_shard`` implies, but never ask
        # for more shards than the corpus has: a smaller source should widen its
        # per-shard sample rather than fail the whole blend.
        available = len(
            corpus.manifest.full_shards()
            if full_shards_only
            else corpus.manifest.entries
        )
        if available < 1:
            raise LoaderError(
                f"{corpus.name}: no shards satisfy the selection constraints"
            )
        n_shards = max(1, min(-(-want // per_shard), available))
        this_per_shard = -(-want // n_shards)  # ceil, so n_shards * this >= want
        piece = SequenceSet.build(
            corpus.manifest,
            name=f"{name}:{corpus.name}",
            # Offset the seed per corpus so two sources never mirror each
            # other's shard choices when they share a shard count.
            seed=seed + (abs(hash(corpus.name)) % 10_000),
            n_shards=n_shards,
            per_shard=this_per_shard,
            full_shards_only=full_shards_only,
            **({"min_sequences": min_sequences} if min_sequences is not None else {}),
            exclude=banned,
        )
        refs = piece.refs[:want]
        parts.append(refs)
        banned.append(set(refs))

    merged = tuple(r for refs in parts for r in refs)
    if not merged:
        raise LoaderError("blended holdout came out empty")

    manifest_digests = ",".join(
        f"{c.name}={c.manifest.digest[:12]}" for c in corpora
    )
    return SequenceSet(
        name=name,
        refs=merged,
        seed=seed,
        manifest_sha256=corpora.config.config_version or manifest_digests,
        seq_len=next(iter(corpora)).manifest.seq_len,
        created=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        strategy=(
            f"blended per_shard={per_shard} total={total} "
            + " ".join(f"{k}={v}" for k, v in allocation.items())
        ),
        notes=tuple(notes),
    )


def split_by_corpus(sequence_set) -> dict[str, list]:
    """Group a blended holdout's refs by corpus prefix."""
    out: dict[str, list] = {}
    for ref in sequence_set:
        prefix = ref.shard.split("__", 1)[0]
        out.setdefault(prefix, []).append(ref)
    return dict(sorted(out.items()))
