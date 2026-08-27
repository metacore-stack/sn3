"""Sequence identity and frozen, provenance-carrying sequence sets.

Sequences are addressed as ``(shard_name, index)``. That pair is stable across
machines, cache evictions and re-downloads, which is what makes a measurement
reproducible six weeks later.

Sets are built stratified across Common Crawl dumps on purpose. The validator
draws all 2000 of its sequences from a *single* shard, so between-shard
difficulty is a real source of variance in ``mu_hat``. A holdout concentrated in
one shard gives a confident estimate of the wrong thing.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .errors import LoaderError
from .manifest import FULL_SHARD_SEQUENCES, ShardManifest

SET_VERSION = 1


@dataclass(frozen=True, order=True)
class SequenceRef:
    """A single 2048-token sequence, addressed portably."""

    shard: str
    index: int

    def __str__(self) -> str:
        return f"{self.shard}#{self.index}"

    @classmethod
    def parse(cls, text: str) -> "SequenceRef":
        shard, _, index = text.rpartition("#")
        if not shard or not index.isdigit():
            raise ValueError(f"cannot parse sequence ref {text!r}; expected 'shard#index'")
        return cls(shard=shard, index=int(index))


@dataclass(frozen=True)
class SequenceSet:
    """A named, frozen collection of sequence refs that carries its provenance."""

    name: str
    refs: tuple[SequenceRef, ...]
    seed: int
    manifest_sha256: str
    seq_len: int
    created: str
    strategy: str
    version: int = SET_VERSION
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.refs)) != len(self.refs):
            raise LoaderError(f"sequence set {self.name!r} contains duplicate refs")

    # -- set semantics -----------------------------------------------------

    def as_set(self) -> frozenset[SequenceRef]:
        return frozenset(self.refs)

    def shards(self) -> list[str]:
        return sorted({r.shard for r in self.refs})

    def crawls(self) -> list[str]:
        found = set()
        for ref in self.refs:
            parts = ref.shard.split("__")
            if len(parts) > 2:
                found.add(parts[1])
        return sorted(found)

    def overlaps(self, other: "SequenceSet | Iterable[SequenceRef]") -> set[SequenceRef]:
        other_refs = other.as_set() if isinstance(other, SequenceSet) else set(other)
        return self.as_set() & other_refs

    def __len__(self) -> int:
        return len(self.refs)

    def __iter__(self) -> Iterator[SequenceRef]:
        return iter(self.refs)

    def __contains__(self, ref: object) -> bool:
        return ref in self.as_set()

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "seed": self.seed,
            "manifest_sha256": self.manifest_sha256,
            "seq_len": self.seq_len,
            "created": self.created,
            "strategy": self.strategy,
            "notes": list(self.notes),
            "n_refs": len(self.refs),
            "refs": [str(r) for r in self.refs],
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "SequenceSet":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=payload["name"],
            refs=tuple(SequenceRef.parse(r) for r in payload["refs"]),
            seed=int(payload["seed"]),
            manifest_sha256=payload["manifest_sha256"],
            seq_len=int(payload.get("seq_len", 2048)),
            created=payload.get("created", ""),
            strategy=payload.get("strategy", "unknown"),
            version=int(payload.get("version", SET_VERSION)),
            notes=tuple(payload.get("notes", [])),
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def build(
        cls,
        manifest: ShardManifest,
        *,
        name: str,
        seed: int,
        n_shards: int = 40,
        per_shard: int = 128,
        full_shards_only: bool = True,
        min_sequences: int = FULL_SHARD_SEQUENCES,
        exclude: Iterable["SequenceSet | Iterable[SequenceRef]"] = (),
        notes: Sequence[str] = (),
    ) -> "SequenceSet":
        """Build a holdout stratified across crawls, deterministically.

        Determinism comes from ``random.Random(seed)`` driven over *sorted*
        inputs only, so the same seed and manifest always yield the same refs
        regardless of dict ordering or platform.
        """
        if n_shards < 1 or per_shard < 1:
            raise LoaderError("n_shards and per_shard must both be at least 1")

        pool = (
            manifest.full_shards(min_sequences)
            if full_shards_only
            else list(manifest.entries)
        )
        if not pool:
            raise LoaderError("no shards satisfy the selection constraints")

        banned: set[SequenceRef] = set()
        for other in exclude:
            banned |= other.as_set() if isinstance(other, SequenceSet) else set(other)

        by_crawl: dict[str, list] = {}
        for entry in pool:
            by_crawl.setdefault(entry.crawl or "unknown", []).append(entry)
        for shards in by_crawl.values():
            shards.sort(key=lambda e: e.key)

        rng = random.Random(seed)
        crawls = sorted(by_crawl)
        rng.shuffle(crawls)

        # Round-robin the crawls so a request for 40 shards spans 40 different
        # dumps where possible, rather than clustering inside one.
        chosen = []
        used_keys: set[str] = set()
        cursor = 0
        while len(chosen) < n_shards and cursor < n_shards * len(crawls) + len(crawls):
            crawl = crawls[cursor % len(crawls)]
            cursor += 1
            candidates = [e for e in by_crawl[crawl] if e.key not in used_keys]
            if not candidates:
                continue
            pick = rng.choice(candidates)
            used_keys.add(pick.key)
            chosen.append(pick)
        if len(chosen) < n_shards:
            raise LoaderError(
                f"requested {n_shards} shards but only {len(chosen)} are available"
            )

        refs: list[SequenceRef] = []
        for entry in sorted(chosen, key=lambda e: e.key):
            available = entry.sequences(manifest.seq_len)
            allowed = [
                i
                for i in range(available)
                if SequenceRef(entry.name, i) not in banned
            ]
            if len(allowed) < per_shard:
                raise LoaderError(
                    f"{entry.name} has {len(allowed)} selectable sequences, "
                    f"fewer than the {per_shard} requested"
                )
            for index in sorted(rng.sample(allowed, per_shard)):
                refs.append(SequenceRef(entry.name, index))

        return cls(
            name=name,
            refs=tuple(refs),
            seed=seed,
            manifest_sha256=manifest.digest,
            seq_len=manifest.seq_len,
            created=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            strategy=(
                f"stratified-by-crawl n_shards={n_shards} per_shard={per_shard} "
                f"full_shards_only={full_shards_only}"
            ),
            notes=tuple(notes),
        )
