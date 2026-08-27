"""A store of measurements, not of conclusions.

The mistake this module exists to prevent is storing verdicts. A verdict --
"this run beat the king by 0.13" -- is a statement about two models, and the
king changes roughly every five hours. Once it changes, every stored verdict is
wrong, and there is no way to tell which ones by looking at them. The usual
consequence is re-running experiments to find out.

Storing the loss vector instead makes a king change cost exactly one scoring
run: score the new king on the same holdout, then recompute every past verdict
from vectors already on disk. Twenty experiments re-baseline in seconds.

That only works if the vectors are comparable, so a record carries the identity
of everything the comparison depends on -- the holdout, the corpus manifests,
the engine shape -- and :meth:`EvidenceStore.comparable` refuses to pair vectors
that disagree on any of it. A paired bootstrap over sequences that are not the
same sequences is not a weaker measurement; it is a meaningless one.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .compare import Comparison, compare
from .engine import StatsSpec
from .errors import EvaluationError
from .lossvec import LossVector

INDEX_NAME = "index.json"
VECTOR_DIR = "vectors"
STORE_VERSION = 1


@dataclass
class Cost:
    """What a run consumed. Kept beside the measurement, not in a spreadsheet."""

    gpu_hours: float = 0.0
    usd_per_gpu_hour: float = 0.0
    wall_hours: float = 0.0
    n_gpus: int = 0
    notes: str = ""

    @property
    def usd(self) -> float:
        return round(self.gpu_hours * self.usd_per_gpu_hour, 2)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["usd"] = self.usd
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "Cost":
        payload = dict(payload or {})
        payload.pop("usd", None)
        return cls(**{k: v for k, v in payload.items() if k in cls.__annotations__})


@dataclass
class EvidenceRecord:
    """One measurement, plus everything needed to reproduce or re-baseline it."""

    run_id: str
    vector_path: str
    kind: str = "challenger"  # challenger | king
    model_label: str = ""
    model_digest: str = ""
    sequence_set: str = ""
    manifest_sha256: str = ""
    engine: dict[str, Any] = field(default_factory=dict)
    n: int = 0
    mean_loss: float = 0.0
    created: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def basis(self) -> tuple[str, str, str]:
        """What two vectors must agree on before they can be paired."""
        engine = self.engine or {}
        shape = json.dumps(
            {k: engine.get(k) for k in sorted(engine)}, sort_keys=True, separators=(",", ":")
        )
        return (self.sequence_set, self.manifest_sha256, shape)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceRecord":
        payload = dict(payload)
        payload["notes"] = tuple(payload.get("notes") or ())
        known = {k: v for k, v in payload.items() if k in cls.__annotations__}
        return cls(**known)


@dataclass
class Standing:
    """A record placed against a particular king."""

    record: EvidenceRecord
    comparison: Comparison | None
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return bool(self.comparison and self.comparison.overall.accepted)

    @property
    def mu_hat(self) -> float | None:
        return self.comparison.overall.mu_hat if self.comparison else None

    @property
    def lcb(self) -> float | None:
        return self.comparison.overall.lcb if self.comparison else None

    @property
    def margin(self) -> float | None:
        """How far the lower bound clears the threshold. Negative means it does not."""
        if not self.comparison:
            return None
        return self.comparison.overall.lcb - self.comparison.overall.delta

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.record.run_id,
            "model_label": self.record.model_label,
            "n": self.record.n,
            "mean_loss": round(self.record.mean_loss, 6),
            "mu_hat": None if self.mu_hat is None else round(self.mu_hat, 6),
            "lcb": None if self.lcb is None else round(self.lcb, 6),
            "margin": None if self.margin is None else round(self.margin, 6),
            "accepted": self.accepted,
            "reason": self.reason,
            "cost_usd": Cost.from_dict(self.record.cost).usd,
        }


class EvidenceStore:
    """Loss vectors on disk, with verdicts computed on demand."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser()
        self.vector_dir = self.root / VECTOR_DIR
        self._records: dict[str, EvidenceRecord] = {}
        self._loaded = False

    # -- persistence -------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_NAME

    def load(self) -> "EvidenceStore":
        self._records = {}
        if self.index_path.is_file():
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            for entry in payload.get("records") or []:
                record = EvidenceRecord.from_dict(entry)
                self._records[record.run_id] = record
        self._loaded = True
        return self

    def _ensure(self) -> None:
        if not self._loaded:
            self.load()

    def save(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STORE_VERSION,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "records": [r.to_dict() for r in self.ordered()],
        }
        self.index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.index_path

    # -- writing -----------------------------------------------------------

    def record(
        self,
        vector: LossVector,
        *,
        run_id: str,
        kind: str = "challenger",
        provenance: dict[str, Any] | None = None,
        cost: Cost | None = None,
        overwrite: bool = False,
    ) -> EvidenceRecord:
        """Store a loss vector and index it. No verdict is computed or kept."""
        self._ensure()
        if kind not in ("challenger", "king"):
            raise EvaluationError(f"kind must be challenger or king, got {kind!r}")
        if run_id in self._records and not overwrite:
            raise EvaluationError(
                f"run_id {run_id!r} already recorded; pass overwrite=True to replace it"
            )

        self.vector_dir.mkdir(parents=True, exist_ok=True)
        path = self.vector_dir / f"{run_id}.json"
        vector.save(path)

        entry = EvidenceRecord(
            run_id=run_id,
            vector_path=str(path.relative_to(self.root)),
            kind=kind,
            model_label=vector.model_label,
            model_digest=vector.model_digest,
            sequence_set=vector.sequence_set,
            manifest_sha256=getattr(vector, "manifest_sha256", ""),
            engine=dict(vector.engine or {}),
            n=len(vector),
            mean_loss=vector.mean,
            created=vector.created,
            provenance=dict(provenance or {}),
            cost=(cost or Cost()).to_dict(),
            notes=tuple(vector.notes or ()),
        )
        self._records[run_id] = entry
        self.save()
        return entry

    def forget(self, run_id: str, *, delete_vector: bool = False) -> bool:
        self._ensure()
        entry = self._records.pop(run_id, None)
        if entry is None:
            return False
        if delete_vector:
            (self.root / entry.vector_path).unlink(missing_ok=True)
        self.save()
        return True

    # -- reading -----------------------------------------------------------

    def ordered(self) -> list[EvidenceRecord]:
        self._ensure()
        return sorted(self._records.values(), key=lambda r: (r.created, r.run_id))

    def get(self, run_id: str) -> EvidenceRecord:
        self._ensure()
        if run_id not in self._records:
            raise EvaluationError(f"no record {run_id!r}")
        return self._records[run_id]

    def vector(self, run_id: str) -> LossVector:
        entry = self.get(run_id)
        path = self.root / entry.vector_path
        if not path.is_file():
            raise EvaluationError(f"{path} is missing; the index references a vector that is gone")
        return LossVector.load(path)

    def kings(self) -> list[EvidenceRecord]:
        return [r for r in self.ordered() if r.kind == "king"]

    def challengers(self) -> list[EvidenceRecord]:
        return [r for r in self.ordered() if r.kind == "challenger"]

    def latest_king(self, *, sequence_set: str | None = None) -> EvidenceRecord | None:
        candidates = self.kings()
        if sequence_set:
            candidates = [r for r in candidates if r.sequence_set == sequence_set]
        return candidates[-1] if candidates else None

    def comparable(self, a: EvidenceRecord, b: EvidenceRecord) -> str:
        """Empty if the two can be paired, otherwise why they cannot."""
        if a.basis == b.basis:
            return ""
        if a.sequence_set != b.sequence_set:
            return f"different holdouts ({a.sequence_set!r} vs {b.sequence_set!r})"
        if a.manifest_sha256 != b.manifest_sha256:
            return "corpus manifests differ; the same ref may name different tokens"
        return "engine shape differs; the losses were not produced the same way"

    # -- the point of all this --------------------------------------------

    def standings(
        self,
        *,
        king_run_id: str | None = None,
        stats: StatsSpec | None = None,
        run_ids: Sequence[str] | None = None,
    ) -> list[Standing]:
        """Rank every stored challenger against one king.

        This is the re-baselining operation. When the throne turns over, score
        the new king once, record it, and call this again -- no challenger is
        re-run, because nothing about a challenger's loss vector depends on who
        it was compared against.
        """
        self._ensure()
        king_entry = (
            self.get(king_run_id) if king_run_id else self.latest_king()
        )
        if king_entry is None:
            raise EvaluationError(
                "no king vector recorded; score the current king once with "
                "kind='king' before asking for standings"
            )
        king_vector = self.vector(king_entry.run_id)

        wanted = self.challengers()
        if run_ids is not None:
            keep = set(run_ids)
            wanted = [r for r in wanted if r.run_id in keep]

        out: list[Standing] = []
        for entry in wanted:
            reason = self.comparable(king_entry, entry)
            if reason:
                out.append(Standing(record=entry, comparison=None, reason=reason))
                continue
            try:
                comparison = compare(
                    king_vector, self.vector(entry.run_id), stats=stats
                )
            except EvaluationError as exc:
                out.append(Standing(record=entry, comparison=None, reason=str(exc)))
                continue
            out.append(Standing(record=entry, comparison=comparison))

        # Best first; unrankable last.
        out.sort(key=lambda s: (s.mu_hat is None, -(s.mu_hat or 0.0)))
        return out

    def leaderboard(self, **kwargs) -> dict[str, Any]:
        standings = self.standings(**kwargs)
        king = self.get(kwargs["king_run_id"]) if kwargs.get("king_run_id") else self.latest_king()
        accepted = [s for s in standings if s.accepted]
        return {
            "king": {
                "run_id": king.run_id if king else None,
                "label": king.model_label if king else None,
                "digest": king.model_digest if king else None,
                "mean_loss": round(king.mean_loss, 6) if king else None,
            },
            "challengers": len(standings),
            "accepted": len(accepted),
            "unrankable": sum(1 for s in standings if s.comparison is None),
            "rows": [s.to_dict() for s in standings],
        }

    # -- cost telemetry ----------------------------------------------------

    def spend(self, *, run_ids: Iterable[str] | None = None) -> dict[str, Any]:
        """What has been spent, and what a nat of improvement has cost.

        Divides by the *best* improvement rather than the total, because the
        runs that did not win still had to happen to find the one that did.
        """
        self._ensure()
        records = (
            [self.get(r) for r in run_ids] if run_ids is not None else self.challengers()
        )
        costs = [Cost.from_dict(r.cost) for r in records]
        total_usd = round(sum(c.usd for c in costs), 2)
        total_gpu_hours = round(sum(c.gpu_hours for c in costs), 2)

        best = None
        try:
            standings = self.standings()
            ranked = [s for s in standings if s.mu_hat is not None]
            best = max((s.mu_hat for s in ranked), default=None)
        except EvaluationError:
            pass

        return {
            "runs": len(records),
            "gpu_hours": total_gpu_hours,
            "usd": total_usd,
            "best_mu_hat": None if best is None else round(best, 6),
            "usd_per_nat": (
                round(total_usd / best, 2) if best and best > 0 else None
            ),
            "usd_per_run": round(total_usd / len(records), 2) if records else None,
        }
