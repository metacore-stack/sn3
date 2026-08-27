"""The published shard inventory.

Two things here are easy to get wrong and both are load-bearing:

1. ``manifest_sha256`` in the dataset config is computed over *canonical* JSON
   (sorted keys, compact separators), not over the bytes you download. Hashing
   the raw file produces a mismatch and looks like tampering.
2. Shard sizes are not uniform. 98.6% of shards hold exactly 6144 sequences, but
   580 of them hold fewer than 2000 and the smallest holds 3.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .errors import ManifestError, ShardNotFoundError

DEFAULT_MANIFEST_URL = (
    "https://pub-d923bc4e8fcb45f6b703bc750bcf8aa6.r2.dev/finewebedu/manifest.json"
)
BUCKET_ROOT = "https://pub-d923bc4e8fcb45f6b703bc750bcf8aa6.r2.dev"
USER_AGENT = "fineweb-loader/1.0 (+read-only public dataset mirror)"

FULL_SHARD_SEQUENCES = 6144

# Every published shard carries exactly 128 bytes of .npy header; the verifier
# allows a little slack rather than pinning to that constant.
NPY_HEADER_BYTES = 128
MAX_NPY_HEADER_BYTES = 1024


def canonical_sha256(payload: dict[str, Any]) -> str:
    """The hash the dataset config publishes: sorted-key, compact-separator JSON."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ShardEntry:
    """One row of the inventory."""

    key: str
    n_tokens: int
    size_bytes: int
    sha256: str

    @property
    def name(self) -> str:
        """Bare filename, which is how ``shards_used`` in the dashboard names it."""
        return self.key.rsplit("/", 1)[-1]

    @property
    def crawl(self) -> str | None:
        """Common Crawl dump, e.g. ``CC-MAIN-2019-43``."""
        parts = self.name.split("__")
        return parts[1] if len(parts) > 2 else None

    @property
    def part(self) -> str | None:
        parts = self.name.split("__")
        return parts[2] if len(parts) > 3 else None

    def sequences(self, seq_len: int = 2048) -> int:
        return self.n_tokens // seq_len

    def is_full(self, seq_len: int = 2048) -> bool:
        return self.sequences(seq_len) >= FULL_SHARD_SEQUENCES


@dataclass(frozen=True)
class ManifestStats:
    total_shards: int
    total_tokens: int
    total_bytes: int
    total_sequences: int
    crawls: int
    min_sequences: int
    max_sequences: int
    full_shards: int
    short_shards: int


class ShardManifest:
    """The 125,441-entry shard inventory, plus verification and selection."""

    def __init__(self, payload: dict[str, Any], *, source: str | None = None):
        if not isinstance(payload, dict):
            raise ManifestError("manifest payload is not a JSON object")
        rows = payload.get("shards")
        if not isinstance(rows, list) or not rows:
            raise ManifestError("manifest contains no shards")
        self.payload = payload
        self.source = source
        self.entries: tuple[ShardEntry, ...] = tuple(
            ShardEntry(
                key=str(row["key"]),
                n_tokens=int(row["n_tokens"]),
                size_bytes=int(row["size_bytes"]),
                sha256=str(row["sha256"]),
            )
            for row in rows
            if isinstance(row, dict) and "key" in row
        )
        if not self.entries:
            raise ManifestError("manifest shard rows are malformed")
        self._by_key = {e.key: e for e in self.entries}
        self._by_name = {e.name: e for e in self.entries}

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "ShardManifest":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ManifestError(f"{path} does not exist; run 'manifest sync'") from exc
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path} is not valid JSON: {exc}") from exc
        return cls(payload, source=str(path))

    @classmethod
    def download(
        cls, url: str = DEFAULT_MANIFEST_URL, *, timeout: float = 120.0
    ) -> tuple["ShardManifest", bytes]:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except OSError as exc:
            raise ManifestError(f"could not download {url}: {exc}") from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ManifestError(f"{url} returned invalid JSON: {exc}") from exc
        return cls(payload, source=url), raw

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return path

    # -- metadata ----------------------------------------------------------

    @property
    def seq_len(self) -> int:
        return int(self.payload.get("seq_len", 2048))

    @property
    def dtype(self) -> str | None:
        return self.payload.get("dtype")

    @property
    def tokenizer(self) -> str | None:
        return self.payload.get("tokenizer")

    @property
    def source_revision(self) -> str | None:
        return self.payload.get("source_revision")

    @property
    def shard_prefix(self) -> str:
        return str(self.payload.get("shard_prefix", "finewebedu/shards/"))

    @property
    def digest(self) -> str:
        """Canonical hash, comparable to ``sources[].manifest_sha256``."""
        return canonical_sha256(self.payload)

    # -- verification ------------------------------------------------------

    def verify(
        self,
        *,
        expected_digest: str | None = None,
        expected_seq_len: int | None = None,
        expected_tokenizer: str | None = None,
        expected_dtype: str | None = None,
    ) -> list[str]:
        """Return a list of problems; empty means the inventory is trustworthy."""
        problems: list[str] = []

        if expected_digest and self.digest != expected_digest:
            problems.append(
                f"manifest digest {self.digest} != expected {expected_digest}"
            )
        if expected_seq_len is not None and self.seq_len != expected_seq_len:
            problems.append(f"seq_len {self.seq_len} != expected {expected_seq_len}")
        if expected_tokenizer and self.tokenizer != expected_tokenizer:
            problems.append(
                f"tokenizer {self.tokenizer!r} != expected {expected_tokenizer!r}"
            )
        if expected_dtype and self.dtype != expected_dtype:
            problems.append(f"dtype {self.dtype!r} != expected {expected_dtype!r}")

        declared_shards = self.payload.get("total_shards")
        if declared_shards is not None and int(declared_shards) != len(self.entries):
            problems.append(
                f"total_shards {declared_shards} != {len(self.entries)} rows present"
            )
        declared_tokens = self.payload.get("total_tokens")
        actual_tokens = sum(e.n_tokens for e in self.entries)
        if declared_tokens is not None and int(declared_tokens) != actual_tokens:
            problems.append(
                f"total_tokens {declared_tokens} != {actual_tokens} summed from rows"
            )

        # size_bytes is the size of the file on disk, so it exceeds the token
        # block by the .npy header. Every shard in the published inventory
        # carries exactly NPY_HEADER_BYTES of it, but the check tolerates any
        # small, non-negative, consistent overhead rather than pinning to 128.
        overheads: set[int] = set()
        for entry in self.entries:
            overhead = entry.size_bytes - entry.n_tokens * 4
            overheads.add(overhead)
            if overhead < 0:
                problems.append(
                    f"{entry.name}: size_bytes {entry.size_bytes} is smaller than "
                    f"its {entry.n_tokens * 4}-byte token block"
                )
                break
            if overhead > MAX_NPY_HEADER_BYTES:
                problems.append(
                    f"{entry.name}: {overhead} bytes of non-token data exceeds the "
                    f"{MAX_NPY_HEADER_BYTES}-byte header allowance"
                )
                break
        if len(overheads) > 1 and not problems:
            problems.append(
                f"inconsistent .npy header sizes across shards: {sorted(overheads)}"
            )
        return problems

    @property
    def header_overhead(self) -> int | None:
        """Bytes of .npy header per shard, when uniform across the inventory."""
        overheads = {e.size_bytes - e.n_tokens * 4 for e in self.entries}
        return overheads.pop() if len(overheads) == 1 else None

    # -- selection ---------------------------------------------------------

    def lookup(self, key_or_name: str) -> ShardEntry:
        entry = self._by_key.get(key_or_name) or self._by_name.get(key_or_name)
        if entry is None:
            raise ShardNotFoundError(f"no shard {key_or_name!r} in the manifest")
        return entry

    def url_for(self, entry: ShardEntry | str, *, root: str = BUCKET_ROOT) -> str:
        """Resolve a shard's download URL.

        ``shard_prefix`` is ``finewebedu/shards/`` while ``key`` is
        ``shards/finewebedu__…``; they overlap on ``shards/``. The bucket path is
        the prefix's leading segment joined to the key.
        """
        if isinstance(entry, str):
            entry = self.lookup(entry)
        namespace = self.shard_prefix.split("/", 1)[0]
        return f"{root.rstrip('/')}/{namespace}/{entry.key.lstrip('/')}"

    def by_crawl(self) -> dict[str, list[ShardEntry]]:
        grouped: dict[str, list[ShardEntry]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.crawl or "unknown", []).append(entry)
        for shards in grouped.values():
            shards.sort(key=lambda e: e.key)
        return dict(sorted(grouped.items()))

    def full_shards(self, minimum: int = FULL_SHARD_SEQUENCES) -> list[ShardEntry]:
        """Shards with at least ``minimum`` addressable sequences.

        Every observed validator draw landed on a full-size shard, which a
        sampler needing 2000 sequences would have to do. Mirror that by default
        so holdouts come from the same population the validator can reach.
        """
        return [e for e in self.entries if e.sequences(self.seq_len) >= minimum]

    def stats(self) -> ManifestStats:
        seqs = [e.sequences(self.seq_len) for e in self.entries]
        crawls = Counter(e.crawl for e in self.entries)
        return ManifestStats(
            total_shards=len(self.entries),
            total_tokens=sum(e.n_tokens for e in self.entries),
            total_bytes=sum(e.size_bytes for e in self.entries),
            total_sequences=sum(seqs),
            crawls=len(crawls),
            min_sequences=min(seqs),
            max_sequences=max(seqs),
            full_shards=sum(1 for s in seqs if s >= FULL_SHARD_SEQUENCES),
            short_shards=sum(1 for s in seqs if s < 2000),
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[ShardEntry]:
        return iter(self.entries)

    def __repr__(self) -> str:
        return (
            f"<ShardManifest shards={len(self.entries)} seq_len={self.seq_len} "
            f"digest={self.digest[:12]}…>"
        )
