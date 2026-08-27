"""Loss vectors: one number per sequence, never a mean.

The paired bootstrap consumes the whole vector. An average discards exactly the
information the test is built on, so nothing in this package ever stores one as
a primary artifact.

Vectors are expensive to produce -- roughly 57 minutes of eight GPUs for 2,000
sequences across two models -- so they are treated as durable artifacts that
carry enough provenance to be trusted weeks later.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .engine import EngineSpec
from .errors import AlignmentError, EvaluationError

VECTOR_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def shard_of(ref: str) -> str:
    """Shard name from a ``shard#index`` reference."""
    return ref.rsplit("#", 1)[0]


@dataclass(frozen=True)
class LossVector:
    """Per-sequence losses aligned to the refs that produced them."""

    refs: tuple[str, ...]
    losses: tuple[float, ...]
    model_label: str
    model_digest: str = ""
    sequence_set: str = ""
    manifest_sha256: str = ""
    engine: dict[str, Any] = field(default_factory=lambda: EngineSpec().to_dict())
    stats_hint: dict[str, Any] = field(default_factory=dict)
    wall_time_s: float | None = None
    created: str = field(default_factory=_now)
    version: int = VECTOR_VERSION
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.refs) != len(self.losses):
            raise EvaluationError(
                f"{self.model_label}: {len(self.refs)} refs but "
                f"{len(self.losses)} losses"
            )
        if not self.refs:
            raise EvaluationError(f"{self.model_label}: empty loss vector")
        if len(set(self.refs)) != len(self.refs):
            raise EvaluationError(f"{self.model_label}: duplicate refs")
        for ref, value in zip(self.refs, self.losses):
            if not math.isfinite(value):
                raise EvaluationError(
                    f"{self.model_label}: non-finite loss {value!r} at {ref}"
                )

    # -- shape -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.refs)

    @property
    def mean(self) -> float:
        """Only for display. Never feed this to the decision function."""
        return sum(self.losses) / len(self.losses)

    def as_map(self) -> dict[str, float]:
        return dict(zip(self.refs, self.losses))

    def shards(self) -> list[str]:
        return sorted({shard_of(r) for r in self.refs})

    def by_shard(self) -> dict[str, "LossVector"]:
        """Split into one vector per shard.

        The validator draws every sequence of an evaluation from a single shard,
        so per-shard behaviour is what actually predicts your odds.
        """
        grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for ref, value in zip(self.refs, self.losses):
            grouped[shard_of(ref)].append((ref, value))
        out: dict[str, LossVector] = {}
        for shard, rows in sorted(grouped.items()):
            out[shard] = LossVector(
                refs=tuple(r for r, _ in rows),
                losses=tuple(v for _, v in rows),
                model_label=self.model_label,
                model_digest=self.model_digest,
                sequence_set=f"{self.sequence_set}:{shard}",
                manifest_sha256=self.manifest_sha256,
                engine=dict(self.engine),
                created=self.created,
            )
        return out

    def subset(self, refs: Sequence[str]) -> "LossVector":
        """A vector restricted to ``refs``, in the order given."""
        lookup = self.as_map()
        missing = [r for r in refs if r not in lookup]
        if missing:
            raise AlignmentError(
                f"{self.model_label}: {len(missing)} ref(s) absent, e.g. {missing[:3]}"
            )
        return LossVector(
            refs=tuple(refs),
            losses=tuple(lookup[r] for r in refs),
            model_label=self.model_label,
            model_digest=self.model_digest,
            sequence_set=self.sequence_set,
            manifest_sha256=self.manifest_sha256,
            engine=dict(self.engine),
            stats_hint=dict(self.stats_hint),
            wall_time_s=self.wall_time_s,
            created=self.created,
            notes=self.notes,
        )

    # -- alignment ---------------------------------------------------------

    def assert_aligned(self, other: "LossVector") -> None:
        """Refuse to let misaligned vectors reach the decision function.

        A silent misalignment produces a well-formed number built from unrelated
        pairs. There is no downstream check that would catch it, so this raises.
        """
        if len(self) != len(other):
            raise AlignmentError(
                f"length mismatch: {self.model_label} has {len(self)}, "
                f"{other.model_label} has {len(other)}"
            )
        if self.refs != other.refs:
            first = next(
                (i for i, (a, b) in enumerate(zip(self.refs, other.refs)) if a != b),
                None,
            )
            detail = (
                f" first divergence at position {first}: "
                f"{self.refs[first]!r} vs {other.refs[first]!r}"
                if first is not None
                else ""
            )
            raise AlignmentError(
                f"{self.model_label} and {other.model_label} cover different "
                f"sequences or a different order.{detail}"
            )
        if (
            self.manifest_sha256
            and other.manifest_sha256
            and self.manifest_sha256 != other.manifest_sha256
        ):
            raise AlignmentError(
                f"vectors were built against different shard manifests: "
                f"{self.manifest_sha256[:12]}… vs {other.manifest_sha256[:12]}…"
            )

    def engine_differences(self, other: "LossVector") -> list[str]:
        """Engine settings that differ between two vectors."""
        keys = set(self.engine) | set(other.engine)
        return [
            f"{k}: {self.engine.get(k)!r} vs {other.engine.get(k)!r}"
            for k in sorted(keys)
            if self.engine.get(k) != other.engine.get(k)
        ]

    def align_to(self, refs: Sequence[str]) -> "LossVector":
        return self.subset(refs)

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_label": self.model_label,
            "model_digest": self.model_digest,
            "sequence_set": self.sequence_set,
            "manifest_sha256": self.manifest_sha256,
            "engine": self.engine,
            "stats_hint": self.stats_hint,
            "wall_time_s": self.wall_time_s,
            "created": self.created,
            "notes": list(self.notes),
            "n": len(self.refs),
            "mean_loss": self.mean,
            "refs": list(self.refs),
            "losses": list(self.losses),
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "LossVector":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LossVector":
        return cls(
            refs=tuple(payload["refs"]),
            losses=tuple(float(x) for x in payload["losses"]),
            model_label=payload.get("model_label", "unknown"),
            model_digest=payload.get("model_digest", ""),
            sequence_set=payload.get("sequence_set", ""),
            manifest_sha256=payload.get("manifest_sha256", ""),
            engine=dict(payload.get("engine") or {}),
            stats_hint=dict(payload.get("stats_hint") or {}),
            wall_time_s=payload.get("wall_time_s"),
            created=payload.get("created", ""),
            version=int(payload.get("version", VECTOR_VERSION)),
            notes=tuple(payload.get("notes") or ()),
        )

    def __repr__(self) -> str:
        return (
            f"<LossVector {self.model_label!r} n={len(self)} "
            f"mean={self.mean:.6f} shards={len(self.shards())}>"
        )
