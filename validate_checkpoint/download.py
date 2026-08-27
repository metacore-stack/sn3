"""Fetching a king checkpoint: 220 GB, verified, resumable.

The reference files (`manifest.json`, `config.json`, the weight index) are a few
kilobytes and :class:`KingReference` already fetches those. This is the other
220.58 GB.

Three properties matter, and all three are about not wasting metered GPU time:

* **Resumable.** A failure at 90% must not mean starting over. Partial files are
  kept and continued with an HTTP Range request.
* **Verified per file.** Every file is hashed against the published manifest and
  deleted if it does not match, so a truncated download can never become the
  model you train.
* **Refuses early.** Disk space is checked before the first byte, not after 200 GB.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .errors import KingUnavailableError, ValidationError
from .king import KingFile, KingReference

CHUNK = 8 * 1024 * 1024
USER_AGENT = "validate-checkpoint/1.0 (+king downloader)"
DEFAULT_WORKERS = 4
# Headroom beyond the checkpoint itself, for the partial files and filesystem
# slack. Deliberately generous: running out at 95% wastes the whole transfer.
DISK_MARGIN_BYTES = 20 * 1024**3


@dataclass
class FileProgress:
    path: str
    size: int
    downloaded: int = 0
    verified: bool = False
    resumed_from: int = 0
    error: str | None = None

    @property
    def complete(self) -> bool:
        return self.verified and self.downloaded >= self.size


@dataclass
class DownloadPlan:
    """What a fetch would do, before it does it."""

    destination: Path
    present: tuple[KingFile, ...] = ()
    missing: tuple[KingFile, ...] = ()
    partial: dict[str, int] = field(default_factory=dict)
    free_bytes: int = 0

    @property
    def bytes_needed(self) -> int:
        return sum(f.size for f in self.missing) - sum(self.partial.values())

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.present) + sum(f.size for f in self.missing)

    @property
    def enough_disk(self) -> bool:
        return self.free_bytes >= self.bytes_needed + DISK_MARGIN_BYTES

    def summary(self) -> dict:
        return {
            "destination": str(self.destination),
            "present": len(self.present),
            "missing": len(self.missing),
            "resumable": len(self.partial),
            "bytes_needed": self.bytes_needed,
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "enough_disk": self.enough_disk,
        }


class KingDownloader:
    """Parallel, verified, resumable fetch of a king checkpoint."""

    def __init__(
        self,
        king: KingReference,
        destination: Path,
        *,
        workers: int = DEFAULT_WORKERS,
        timeout: float = 300.0,
    ):
        if not king.files:
            raise KingUnavailableError("king reference lists no files")
        if not king.source:
            raise KingUnavailableError("king reference has no source URL")
        if not king.source.startswith(("http://", "https://")):
            # from_directory() sets source to a local path. There is nothing to
            # download; the caller wanted a local reference.
            raise KingUnavailableError(
                f"king reference points at a local directory ({king.source}), "
                "not the model store; nothing to download"
            )
        self.king = king
        self.destination = Path(destination).expanduser()
        self.workers = max(1, int(workers))
        self.timeout = timeout

    # -- planning ----------------------------------------------------------

    def url_for(self, path: str) -> str:
        return f"{self.king.source.rstrip('/')}/{path}"

    def local_path(self, path: str) -> Path:
        return self.destination / path

    def partial_path(self, path: str) -> Path:
        return self.destination / f"{path}.part"

    def plan(self, *, verify_present: bool = False) -> DownloadPlan:
        """What is already here, what is missing, and whether the disk suffices."""
        self.destination.mkdir(parents=True, exist_ok=True)
        present: list[KingFile] = []
        missing: list[KingFile] = []
        partial: dict[str, int] = {}

        for path in self.king.paths:
            entry = self.king.files[path]
            local = self.local_path(path)
            if local.is_file() and local.stat().st_size == entry.size:
                if not verify_present or sha256_file(local) == entry.sha256:
                    present.append(entry)
                    continue
                local.unlink()
            partial_file = self.partial_path(path)
            if partial_file.is_file():
                got = partial_file.stat().st_size
                if 0 < got < entry.size:
                    partial[path] = got
                else:
                    partial_file.unlink(missing_ok=True)
            missing.append(entry)

        free = shutil.disk_usage(self.destination).free
        return DownloadPlan(
            destination=self.destination,
            present=tuple(present),
            missing=tuple(missing),
            partial=partial,
            free_bytes=free,
        )

    # -- fetching ----------------------------------------------------------

    def fetch(
        self,
        *,
        on_file: Callable[[FileProgress], None] | None = None,
        on_bytes: Callable[[int, int], None] | None = None,
        allow_insufficient_disk: bool = False,
    ) -> list[FileProgress]:
        """Download whatever is missing, verifying every file."""
        plan = self.plan()
        if not plan.enough_disk and not allow_insufficient_disk:
            raise ValidationError(
                f"need {plan.bytes_needed / 1e9:.1f} GB plus "
                f"{DISK_MARGIN_BYTES / 1e9:.0f} GB margin but only "
                f"{plan.free_bytes / 1e9:.1f} GB is free at {self.destination}"
            )

        results: list[FileProgress] = [
            FileProgress(path=f.path, size=f.size, downloaded=f.size, verified=True)
            for f in plan.present
        ]
        if not plan.missing:
            return results

        total = plan.bytes_needed
        done = [0]

        def track(n: int) -> None:
            done[0] += n
            if on_bytes:
                on_bytes(done[0], total)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self._fetch_one, entry, track): entry
                for entry in plan.missing
            }
            for future in as_completed(futures):
                progress = future.result()
                results.append(progress)
                if on_file:
                    on_file(progress)

        failed = [r for r in results if not r.complete]
        if failed:
            raise ValidationError(
                f"{len(failed)} file(s) failed: "
                + "; ".join(f"{r.path}: {r.error}" for r in failed[:3])
            )
        return sorted(results, key=lambda r: r.path)

    def _fetch_one(self, entry: KingFile, track: Callable[[int], None]) -> FileProgress:
        target = self.local_path(entry.path)
        partial = self.partial_path(entry.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        progress = FileProgress(path=entry.path, size=entry.size)

        try:
            start = partial.stat().st_size if partial.is_file() else 0
            if start >= entry.size:
                partial.unlink(missing_ok=True)
                start = 0
            progress.resumed_from = start

            # Rehash whatever was already on disk so the digest covers the whole
            # file, not only the bytes fetched this run.
            digest = hashlib.sha256()
            if start:
                with partial.open("rb") as handle:
                    for block in iter(lambda: handle.read(CHUNK), b""):
                        digest.update(block)

            written = start
            if written < entry.size:
                request = urllib.request.Request(
                    self.url_for(entry.path), headers={"User-Agent": USER_AGENT}
                )
                if start:
                    request.add_header("Range", f"bytes={start}-")
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    if start and response.status != 206:
                        # Server ignored the range; restart cleanly rather than
                        # append duplicate bytes onto the partial file.
                        partial.unlink(missing_ok=True)
                        digest = hashlib.sha256()
                        written = 0
                    mode = "ab" if written else "wb"
                    with partial.open(mode) as handle:
                        while True:
                            block = response.read(CHUNK)
                            if not block:
                                break
                            handle.write(block)
                            digest.update(block)
                            written += len(block)
                            track(len(block))

            progress.downloaded = written
            if written < entry.size:
                # A dropped connection. KEEP the partial -- continuing it is the
                # entire point of this module. Corruption in the kept prefix is
                # still caught, because the digest below covers the whole file.
                progress.error = (
                    f"short read: {written} of {entry.size} bytes; resumable"
                )
                return progress
            if written > entry.size:
                progress.error = f"got {written} bytes, manifest says {entry.size}"
                partial.unlink(missing_ok=True)
                return progress
            if digest.hexdigest() != entry.sha256:
                progress.error = "sha256 mismatch"
                partial.unlink(missing_ok=True)
                return progress

            os.replace(partial, target)
            progress.verified = True
            return progress
        except (urllib.error.URLError, OSError) as exc:
            progress.error = str(exc)
            return progress

    # -- verification ------------------------------------------------------

    def verify(self, *, workers: int | None = None) -> list[str]:
        """Re-hash everything on disk. Returns the paths that do not match."""
        bad: list[str] = []
        with ThreadPoolExecutor(max_workers=workers or self.workers) as pool:
            futures = {
                pool.submit(sha256_file, self.local_path(p)): p
                for p in self.king.paths
                if self.local_path(p).is_file()
            }
            for future in as_completed(futures):
                path = futures[future]
                if future.result() != self.king.files[path].sha256:
                    bad.append(path)
        missing = [p for p in self.king.paths if not self.local_path(p).is_file()]
        return sorted(bad + missing)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"
