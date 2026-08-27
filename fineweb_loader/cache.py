"""Local shard cache with a hard byte budget.

The corpus is 6.27 TB across 125,441 shards. Nothing here ever fetches more than
what is asked for, a download that fails verification is deleted rather than
kept, and the cache refuses to exceed its budget instead of filling the disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .errors import BudgetExceededError, IntegrityError
from .manifest import BUCKET_ROOT, ShardEntry, ShardManifest, USER_AGENT

DEFAULT_BUDGET_BYTES = 4 * 1024**3  # 4 GiB ≈ 80 full shards
CHUNK = 1024 * 1024


@dataclass(frozen=True)
class CacheEntry:
    key: str
    name: str
    size_bytes: int
    sha256: str
    last_used: float

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "last_used": self.last_used,
        }


class ShardCache:
    """Fetch-and-verify with LRU eviction against a byte budget."""

    def __init__(
        self,
        root: Path,
        manifest: ShardManifest,
        *,
        budget_bytes: int = DEFAULT_BUDGET_BYTES,
        timeout: float = 300.0,
        bucket_root: str = BUCKET_ROOT,
    ):
        self.root = Path(root).expanduser()
        self.manifest = manifest
        self.budget_bytes = int(budget_bytes)
        self.timeout = timeout
        # Overridable so a mirror — or a local directory in tests — can serve shards.
        self.bucket_root = bucket_root
        self.shards_dir = self.root / "shards"
        self.index_path = self.root / "index.json"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, CacheEntry] = {}
        self._load_index()

    # -- index -------------------------------------------------------------

    def _load_index(self) -> None:
        if not self.index_path.exists():
            return
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for key, row in (raw.get("entries") or {}).items():
            try:
                entry = CacheEntry(
                    key=key,
                    name=row["name"],
                    size_bytes=int(row["size_bytes"]),
                    sha256=row["sha256"],
                    last_used=float(row.get("last_used", 0.0)),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if (self.shards_dir / entry.name).exists():
                self._index[key] = entry

    def _save_index(self) -> None:
        payload = {
            "budget_bytes": self.budget_bytes,
            "entries": {k: e.to_dict() for k, e in self._index.items()},
        }
        tmp = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.root, delete=False
        )
        try:
            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.replace(tmp.name, self.index_path)
        except BaseException:
            tmp.close()
            Path(tmp.name).unlink(missing_ok=True)
            raise

    # -- state -------------------------------------------------------------

    @property
    def used_bytes(self) -> int:
        return sum(e.size_bytes for e in self._index.values())

    @property
    def free_bytes(self) -> int:
        return max(0, self.budget_bytes - self.used_bytes)

    def path_for(self, entry: ShardEntry | str) -> Path:
        if isinstance(entry, str):
            entry = self.manifest.lookup(entry)
        return self.shards_dir / entry.name

    def has(self, entry: ShardEntry | str) -> bool:
        if isinstance(entry, str):
            entry = self.manifest.lookup(entry)
        return entry.key in self._index and self.path_for(entry).exists()

    def entries(self) -> list[CacheEntry]:
        return sorted(self._index.values(), key=lambda e: e.last_used)

    # -- fetch -------------------------------------------------------------

    def ensure(
        self,
        entry: ShardEntry | str,
        *,
        progress: Callable[[int, int], None] | None = None,
        verify: bool = True,
    ) -> Path:
        """Return a local, verified path for a shard, downloading if needed."""
        if isinstance(entry, str):
            entry = self.manifest.lookup(entry)
        path = self.path_for(entry)

        if self.has(entry):
            self._touch(entry.key)
            return path

        self._make_room(entry.size_bytes, protect=entry.key)
        self._download(entry, path, progress=progress, verify=verify)

        self._index[entry.key] = CacheEntry(
            key=entry.key,
            name=entry.name,
            size_bytes=entry.size_bytes,
            sha256=entry.sha256,
            last_used=_now(),
        )
        self._save_index()
        return path

    def _download(
        self,
        entry: ShardEntry,
        path: Path,
        *,
        progress: Callable[[int, int], None] | None,
        verify: bool,
    ) -> None:
        url = self.manifest.url_for(entry, root=self.bucket_root)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        partial = path.with_suffix(path.suffix + ".part")
        digest = hashlib.sha256()
        written = 0
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                with partial.open("wb") as handle:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        handle.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                        if progress:
                            progress(written, entry.size_bytes)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

        if written != entry.size_bytes:
            partial.unlink(missing_ok=True)
            raise IntegrityError(
                f"{entry.name}: downloaded {written} bytes, manifest says "
                f"{entry.size_bytes}"
            )
        if verify and digest.hexdigest() != entry.sha256:
            partial.unlink(missing_ok=True)
            raise IntegrityError(
                f"{entry.name}: sha256 {digest.hexdigest()} != manifest {entry.sha256}"
            )
        os.replace(partial, path)

    def verify_local(self, entry: ShardEntry | str) -> bool:
        """Re-hash a cached shard. Removes it from the cache if it has rotted."""
        if isinstance(entry, str):
            entry = self.manifest.lookup(entry)
        path = self.path_for(entry)
        if not path.exists():
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK), b""):
                digest.update(chunk)
        if digest.hexdigest() == entry.sha256:
            return True
        self.evict(entry.key)
        return False

    # -- eviction ----------------------------------------------------------

    def _touch(self, key: str) -> None:
        existing = self._index.get(key)
        if existing is None:
            return
        self._index[key] = CacheEntry(
            key=existing.key,
            name=existing.name,
            size_bytes=existing.size_bytes,
            sha256=existing.sha256,
            last_used=_now(),
        )
        self._save_index()

    def _make_room(self, needed: int, *, protect: str | None = None) -> None:
        if needed > self.budget_bytes:
            raise BudgetExceededError(
                f"a single shard needs {needed:,} bytes but the cache budget is "
                f"{self.budget_bytes:,}; raise --budget"
            )
        while self.used_bytes + needed > self.budget_bytes:
            victims = [e for e in self.entries() if e.key != protect]
            if not victims:
                raise BudgetExceededError(
                    f"cannot free {needed:,} bytes within a "
                    f"{self.budget_bytes:,}-byte budget"
                )
            self.evict(victims[0].key)

    def evict(self, key: str) -> bool:
        entry = self._index.pop(key, None)
        if entry is None:
            return False
        (self.shards_dir / entry.name).unlink(missing_ok=True)
        self._save_index()
        return True

    def prune(self, *, keep: Iterable[str] = ()) -> int:
        """Evict least-recently-used shards until inside budget."""
        protected = set(keep)
        freed = 0
        while self.used_bytes > self.budget_bytes:
            victims = [e for e in self.entries() if e.key not in protected]
            if not victims:
                break
            freed += victims[0].size_bytes
            self.evict(victims[0].key)
        return freed

    def clear(self) -> None:
        shutil.rmtree(self.shards_dir, ignore_errors=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self._index.clear()
        self._save_index()


def _now() -> float:
    import time

    return time.time()
