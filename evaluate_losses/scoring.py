"""One command from a checkpoint on disk to a loss vector.

``TorchBackend`` already knows how to score a sequence exactly the way the
validator does. What was missing was everything around it: opening the three
corpora, resolving a named holdout, enforcing the blend, refusing to score
against a set that does not resemble what the validator draws, and writing the
result somewhere the evidence store can find it.

The point of separating this from the backend is that a loss vector is the
durable artefact. A verdict is not: the king changes every few hours and any
stored verdict silently rots, while a stored vector can be re-compared against a
new king for the cost of scoring that king once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .engine import EngineSpec
from .errors import EvaluationError
from .lossvec import LossVector

DEFAULT_STATE_ROOT = Path.home() / "Documents" / "sn3" / "state"
# Below this the paired bootstrap's confidence interval is wider than the
# effects worth chasing; a run that reports on 50 sequences is noise.
MIN_USEFUL_SEQUENCES = 200


@dataclass
class ScoringPlan:
    """What a scoring run will do, before a GPU-hour is spent on it."""

    model_dir: Path
    holdout_name: str
    sequences: int
    per_corpus: dict[str, int] = field(default_factory=dict)
    expected_share: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def actual_share(self) -> dict[str, float]:
        total = sum(self.per_corpus.values())
        if not total:
            return {}
        return {k: v / total for k, v in sorted(self.per_corpus.items())}

    @property
    def max_share_error(self) -> float:
        """Largest gap between this holdout's mixture and the validator's."""
        actual = self.actual_share
        if not actual or not self.expected_share:
            return 0.0
        keys = set(actual) | set(self.expected_share)
        return max(abs(actual.get(k, 0.0) - self.expected_share.get(k, 0.0)) for k in keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "holdout": self.holdout_name,
            "sequences": self.sequences,
            "per_corpus": dict(sorted(self.per_corpus.items())),
            "actual_share": {k: round(v, 4) for k, v in self.actual_share.items()},
            "expected_share": {k: round(v, 4) for k, v in sorted(self.expected_share.items())},
            "max_share_error": round(self.max_share_error, 4),
            "warnings": list(self.warnings),
        }


def open_blend(
    *,
    state_root: Path | None = None,
    budget_gib: float = 8.0,
    holdouts: Sequence[Any] = (),
):
    """A loader over every evaluation corpus, honouring the given holdouts."""
    from fineweb_loader import BlendedLoader, CorpusSet, DatasetConfig

    root = Path(state_root or DEFAULT_STATE_ROOT)
    config_path = root / "dataset-config.json"
    if not config_path.is_file():
        raise EvaluationError(
            f"{config_path} not found; run 'fineweb corpus sync' first"
        )
    dataset = DatasetConfig.load(config_path)
    corpora = CorpusSet.open(dataset, root, budget_bytes=int(budget_gib * 1024**3))
    return BlendedLoader(corpora, holdouts=list(holdouts)), corpora


def load_holdout(name: str, *, state_root: Path | None = None):
    from fineweb_loader import SequenceSet

    root = Path(state_root or DEFAULT_STATE_ROOT)
    path = root / "holdouts" / f"{name}.json"
    if not path.is_file():
        raise EvaluationError(
            f"holdout {name!r} not found at {path}; build one with "
            "'fineweb holdout build --blend'"
        )
    return SequenceSet.load(path)


def plan(
    model_dir: Path | str,
    holdout_name: str,
    *,
    state_root: Path | None = None,
) -> ScoringPlan:
    """Check the holdout before spending anything on scoring it."""
    from fineweb_loader import DatasetConfig
    from fineweb_loader.corpus import split_by_corpus

    root = Path(state_root or DEFAULT_STATE_ROOT)
    holdout = load_holdout(holdout_name, state_root=root)
    parts = split_by_corpus(holdout)
    per_corpus = {name: len(part) for name, part in sorted(parts.items())}

    expected: dict[str, float] = {}
    config_path = root / "dataset-config.json"
    if config_path.is_file():
        dataset = DatasetConfig.load(config_path)
        expected = {s.name: s.proportion for s in dataset.sources}

    warnings: list[str] = []
    total = sum(per_corpus.values())
    if total < MIN_USEFUL_SEQUENCES:
        warnings.append(
            f"only {total} sequences; the paired interval will be wider than "
            "the effects worth chasing"
        )
    missing = sorted(set(expected) - set(per_corpus))
    if missing:
        warnings.append(
            f"holdout has nothing from {missing}, which the validator scores; "
            "the measurement covers only part of the contract"
        )

    result = ScoringPlan(
        model_dir=Path(model_dir),
        holdout_name=holdout_name,
        sequences=total,
        per_corpus=per_corpus,
        expected_share=expected,
        warnings=tuple(warnings),
    )
    if result.max_share_error > 0.05:
        result.warnings = result.warnings + (
            f"mixture is off by {result.max_share_error:.1%}; a score measured "
            "here will not track the validator's",
        )
    return result


def score_checkpoint(
    model_dir: Path | str,
    holdout_name: str,
    *,
    state_root: Path | None = None,
    model_label: str = "",
    model_digest: str = "",
    device_map: str = "auto",
    spec: EngineSpec | None = None,
    budget_gib: float = 8.0,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    out: Path | None = None,
) -> LossVector:
    """Score a checkpoint on a named holdout and return the loss vector.

    The holdout is deliberately *not* excluded from the loader here: evaluation
    is exactly the case where reading held-out sequences is correct. Training
    reads through the same loader with ``allow_holdout=False``, which is what
    keeps the two apart.
    """
    from .backends import TorchBackend

    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise EvaluationError(f"{model_dir} is not a directory")

    holdout = load_holdout(holdout_name, state_root=state_root)
    loader, _ = open_blend(state_root=state_root, budget_gib=budget_gib)

    refs = [str(ref) for ref in holdout.refs]
    if limit is not None:
        refs = refs[: max(0, limit)]
    if not refs:
        raise EvaluationError(f"holdout {holdout_name!r} is empty")

    started = time.monotonic()
    with TorchBackend(
        model_dir,
        loader,
        spec=spec or EngineSpec(),
        model_digest=model_digest,
        device_map=device_map,
    ) as backend:
        vector = backend.score(
            refs, model_label=model_label or model_dir.name, progress=progress
        )

    vector = LossVector(
        refs=vector.refs,
        losses=vector.losses,
        model_label=vector.model_label,
        model_digest=vector.model_digest,
        sequence_set=holdout_name,
        manifest_sha256=vector.manifest_sha256,
        engine=vector.engine,
        wall_time_s=round(time.monotonic() - started, 3),
        notes=(f"scored against holdout {holdout_name!r}",),
    )
    if out is not None:
        vector.save(Path(out))
    return vector
