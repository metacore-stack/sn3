"""On-disk state: immutable pinned targets plus an append-only observation log."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from .errors import TargetNotFoundError
from .target import Target
from .timeutil import iso, now, parse_ts

DEFAULT_ROOT = Path.home() / "Documents" / "sn3" / "state"
ENV_ROOT = "SN3_MONITOR_HOME"


def default_root() -> Path:
    """Where state lives; override with the SN3_MONITOR_HOME environment variable."""
    override = os.environ.get(ENV_ROOT)
    return Path(override).expanduser() if override else DEFAULT_ROOT


@dataclass(frozen=True)
class Store:
    """Filesystem layout for targets and observations."""

    root: Path

    @classmethod
    def open(cls, root: Path | None = None) -> "Store":
        store = cls(root=(root or default_root()).expanduser())
        store.targets_dir.mkdir(parents=True, exist_ok=True)
        return store

    @property
    def targets_dir(self) -> Path:
        return self.root / "targets"

    @property
    def observations_path(self) -> Path:
        return self.root / "observations.jsonl"

    # -- targets -----------------------------------------------------------

    def save_target(self, target: Target) -> Path:
        """Write a target atomically. Existing snapshots are never overwritten."""
        path = self.targets_dir / f"{target.snapshot_id}.json"
        if path.exists():
            return path
        payload = json.dumps(target.to_dict(), indent=2, sort_keys=True)
        _atomic_write(path, payload)
        return path

    def list_targets(self) -> list[str]:
        """Snapshot ids, oldest first. Ids are timestamp-prefixed so this sorts."""
        return sorted(p.stem for p in self.targets_dir.glob("*.json"))

    def load_target(self, snapshot_id: str = "latest") -> Target:
        """Load a target by id, or the most recent one for ``latest``."""
        if snapshot_id in ("latest", "", None):
            available = self.list_targets()
            if not available:
                raise TargetNotFoundError(
                    f"no targets pinned yet in {self.targets_dir}; "
                    "run 'sn3-monitor snapshot' first"
                )
            snapshot_id = available[-1]
        path = self.targets_dir / f"{snapshot_id}.json"
        if not path.exists():
            raise TargetNotFoundError(f"no target {snapshot_id!r} in {self.targets_dir}")
        return Target.from_dict(json.loads(path.read_text(encoding="utf-8")))

    # -- observations ------------------------------------------------------

    def append_observation(self, record: dict[str, Any]) -> None:
        """Append one polling record. Newline-terminated so partial writes show up."""
        record.setdefault("ts", iso(now()))
        self.root.mkdir(parents=True, exist_ok=True)
        with self.observations_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def read_observations(self, since: timedelta | None = None) -> list[dict[str, Any]]:
        """Read observations, optionally limited to a trailing window.

        Malformed lines are skipped rather than raising: this log is appended to
        by long-running processes and a truncated final line is expected.
        """
        cutoff: datetime | None = (now() - since) if since else None
        records: list[dict[str, Any]] = []
        for record in self._iter_observations():
            if cutoff is not None:
                stamp = parse_ts(record.get("ts"))
                if stamp is None or stamp < cutoff:
                    continue
            records.append(record)
        return records

    def _iter_observations(self) -> Iterator[dict[str, Any]]:
        if not self.observations_path.exists():
            return
        with self.observations_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    )
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, path)
    except BaseException:
        handle.close()
        Path(handle.name).unlink(missing_ok=True)
        raise
