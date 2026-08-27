"""The safetensors reuse limit, and a local ledger of what you have submitted.

Added to the validator on 2026-08-27. A checkpoint's combined safetensors
digest may complete at most ``MAX_COMPLETED_EVALS = 3`` evaluations; the fourth
is refused with ``safetensors_reuse_limit``.

That closes the "resubmit the same weights and hope for a favourable draw"
strategy, which the fixed 22/26/52 blend already made weak. Every attempt now has
to be genuinely different weights.

The validator keeps the count server-side, so it cannot be queried. What can be
done locally is keep an honest record of which weights you have already sent,
so a fourth submission of the same bytes is caught before it costs a hotkey.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

# engine.py:81
MAX_COMPLETED_EVALS = 3

LEDGER_FILENAME = "submissions.json"


def safetensors_digest_from_file_digests(file_digests: Mapping[str, str]) -> str:
    """The validator's combined digest, transcribed from ``engine.py``.

    Sorted by name, then ``name || NUL || raw-32-bytes`` folded into one SHA-256.
    Hashing the concatenated hex, or skipping the NUL, gives a different value
    and would make the local ledger key on something the validator never sees.
    """
    if not file_digests:
        raise FileNotFoundError("no .safetensors file digests found")
    h = hashlib.sha256()
    for name, digest in sorted(file_digests.items()):
        digest = digest.lower().removeprefix("sha256:")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{name}: invalid SHA-256 digest {digest!r}")
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(bytes.fromhex(digest))
    return h.hexdigest()


def snapshot_safetensors_digest(model_dir: Path, hasher=None) -> tuple[str, dict[str, str]]:
    """Combined digest for a checkpoint directory, plus the per-file digests."""
    from .checks import sha256_file

    hasher = hasher or sha256_file
    model_dir = Path(model_dir)
    names = sorted(
        p.relative_to(model_dir).as_posix() for p in model_dir.rglob("*.safetensors")
    )
    if not names:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    digests = {name: hasher(model_dir / name) for name in names}
    return safetensors_digest_from_file_digests(digests), digests


@dataclass
class Submission:
    """One recorded submission of a set of weights."""

    digest: str
    model_dir: str
    model_name: str | None
    hotkey: str | None
    submitted_at: str
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "model_dir": self.model_dir,
            "model_name": self.model_name,
            "hotkey": self.hotkey,
            "submitted_at": self.submitted_at,
            "note": self.note,
        }


@dataclass
class SubmissionLedger:
    """Local record of which weights have been sent, and how often."""

    path: Path
    entries: list[Submission] = field(default_factory=list)

    @classmethod
    def load(cls, root: Path) -> "SubmissionLedger":
        path = Path(root) / LEDGER_FILENAME
        entries: list[Submission] = []
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            for row in payload.get("submissions") or []:
                try:
                    entries.append(
                        Submission(
                            digest=row["digest"],
                            model_dir=row.get("model_dir", ""),
                            model_name=row.get("model_name"),
                            hotkey=row.get("hotkey"),
                            submitted_at=row.get("submitted_at", ""),
                            note=row.get("note"),
                        )
                    )
                except (KeyError, TypeError):
                    continue
        return cls(path=path, entries=entries)

    def uses(self, digest: str) -> int:
        return sum(1 for e in self.entries if e.digest == digest)

    def remaining(self, digest: str) -> int:
        return max(0, MAX_COMPLETED_EVALS - self.uses(digest))

    def would_exceed(self, digest: str) -> bool:
        return self.uses(digest) >= MAX_COMPLETED_EVALS

    def record(
        self,
        digest: str,
        model_dir: Path | str,
        *,
        model_name: str | None = None,
        hotkey: str | None = None,
        note: str | None = None,
    ) -> Submission:
        entry = Submission(
            digest=digest,
            model_dir=str(model_dir),
            model_name=model_name,
            hotkey=hotkey,
            submitted_at=datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            note=note,
        )
        self.entries.append(entry)
        self.save()
        return entry

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_completed_evals": MAX_COMPLETED_EVALS,
            "submissions": [e.to_dict() for e in self.entries],
        }
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        )
        try:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(handle.name, self.path)
        except BaseException:
            handle.close()
            Path(handle.name).unlink(missing_ok=True)
            raise
        return self.path

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.digest] = counts.get(entry.digest, 0) + 1
        return {
            "total_submissions": len(self.entries),
            "distinct_weights": len(counts),
            "at_limit": [d for d, n in counts.items() if n >= MAX_COMPLETED_EVALS],
            "max_completed_evals": MAX_COMPLETED_EVALS,
        }
