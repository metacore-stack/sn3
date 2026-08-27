"""Paired comparison, with the per-corpus and per-shard views that predict odds.

The overall LCB answers "is the challenger better across my whole holdout".
That is not quite the question the validator asks. Since 2026-08-27 it draws
2000 sequences split 22/26/52 across finewebedu, automathtext-v2 and
dclm-baseline-1.0, taking a shard from each. So two things matter beyond the
pooled number:

* **per corpus** -- the blend is fixed, so a weakness in one source is paid for
  on every submission, forever.
* **per shard** -- within a corpus you still get one draw, so spread across
  shards is the variance you actually face.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from .engine import StatsSpec
from .lossvec import LossVector
from .policy import paired_bootstrap_verdict, policy_source


@dataclass(frozen=True)
class Verdict:
    """One paired-bootstrap decision."""

    label: str
    n: int
    mu_hat: float
    lcb: float
    delta: float
    accepted: bool
    avg_king_loss: float
    avg_challenger_loss: float
    alpha: float
    n_bootstrap: int
    bootstrap_seed: int
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def margin(self) -> float:
        """How far the LCB clears the bar. Negative means it fell short."""
        return self.lcb - self.delta

    @property
    def share_of_bar(self) -> float | None:
        return self.mu_hat / self.delta if self.delta else None

    @property
    def bootstrap_penalty(self) -> float:
        """``mu_hat - lcb``.

        Small in practice -- 0.4-2% of mu_hat in every observed evaluation --
        because the test is *paired* on identical sequences, so shared sequence
        difficulty cancels. Chasing the mean is the right instinct; the LCB is a
        final check, not a design target.
        """
        return self.mu_hat - self.lcb

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n": self.n,
            "mu_hat": self.mu_hat,
            "lcb": self.lcb,
            "delta": self.delta,
            "accepted": self.accepted,
            "margin": self.margin,
            "bootstrap_penalty": self.bootstrap_penalty,
            "avg_king_loss": self.avg_king_loss,
            "avg_challenger_loss": self.avg_challenger_loss,
            "alpha": self.alpha,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_seed": self.bootstrap_seed,
        }


@dataclass(frozen=True)
class ShardBreakdown:
    """Per-shard verdicts and the spread across them."""

    shards: tuple[Verdict, ...]

    @property
    def mu_hats(self) -> list[float]:
        return [v.mu_hat for v in self.shards]

    @property
    def worst(self) -> Verdict | None:
        return min(self.shards, key=lambda v: v.mu_hat) if self.shards else None

    @property
    def best(self) -> Verdict | None:
        return max(self.shards, key=lambda v: v.mu_hat) if self.shards else None

    @property
    def spread(self) -> float | None:
        values = self.mu_hats
        return (max(values) - min(values)) if len(values) > 1 else None

    @property
    def stdev(self) -> float | None:
        values = self.mu_hats
        return statistics.stdev(values) if len(values) > 1 else None

    @property
    def median(self) -> float | None:
        return statistics.median(self.mu_hats) if self.shards else None

    def clearing_fraction(self, delta: float) -> float | None:
        """Share of shards whose mu_hat clears the bar.

        Read this as an empirical estimate of the probability that a single
        validator draw succeeds.
        """
        if not self.shards:
            return None
        return sum(1 for v in self.shards if v.mu_hat > delta) / len(self.shards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_shards": len(self.shards),
            "median_mu_hat": self.median,
            "spread": self.spread,
            "stdev": self.stdev,
            "worst": self.worst.to_dict() if self.worst else None,
            "best": self.best.to_dict() if self.best else None,
            "shards": [v.to_dict() for v in self.shards],
        }


@dataclass(frozen=True)
class Comparison:
    """Overall verdict, plus the per-corpus and per-shard views."""

    overall: Verdict
    by_shard: ShardBreakdown
    king_label: str
    challenger_label: str
    sequence_set: str
    policy_path: str
    by_corpus: tuple[Verdict, ...] = ()

    @property
    def weakest_corpus(self) -> Verdict | None:
        """Corpus with the smallest improvement.

        Evaluation is 22% finewebedu, 26% automathtext-v2, 52%
        dclm-baseline-1.0. A checkpoint can clear the bar overall while losing
        badly on one source, and since the blend is fixed that weakness is paid
        for on every future submission.
        """
        return min(self.by_corpus, key=lambda v: v.mu_hat) if self.by_corpus else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "king": self.king_label,
            "challenger": self.challenger_label,
            "sequence_set": self.sequence_set,
            "policy_path": self.policy_path,
            "overall": self.overall.to_dict(),
            "by_corpus": [v.to_dict() for v in self.by_corpus],
            "by_shard": self.by_shard.to_dict(),
        }


def _verdict(
    label: str, king: LossVector, challenger: LossVector, stats: StatsSpec
) -> Verdict:
    raw = paired_bootstrap_verdict(
        king.losses,
        challenger.losses,
        bootstrap_seed=stats.bootstrap_seed,
        n_bootstrap=stats.n_bootstrap,
        alpha=stats.alpha,
        delta_threshold=stats.delta_threshold,
    )
    return Verdict(
        label=label,
        n=int(raw["n_sequences"]),
        mu_hat=float(raw["mu_hat"]),
        lcb=float(raw["lcb"]),
        delta=float(raw["delta_threshold"]),
        accepted=bool(raw["accepted"]),
        avg_king_loss=float(raw["avg_king_loss"]),
        avg_challenger_loss=float(raw["avg_challenger_loss"]),
        alpha=float(raw["alpha"]),
        n_bootstrap=int(raw["n_bootstrap"]),
        bootstrap_seed=stats.bootstrap_seed,
        raw=raw,
    )


def compare(
    king: LossVector,
    challenger: LossVector,
    *,
    stats: StatsSpec | None = None,
    per_shard: bool = True,
    min_shard_n: int = 2,
) -> Comparison:
    """Compare two loss vectors using the validator's own decision function.

    Alignment is asserted first: the vectors must cover the same sequences in
    the same order, or this raises rather than returning a meaningless number.
    """
    stats = stats or StatsSpec()
    king.assert_aligned(challenger)

    overall = _verdict("overall", king, challenger, stats)

    corpus_verdicts: list[Verdict] = []
    king_corpora = king.by_corpus()
    challenger_corpora = challenger.by_corpus()
    if len(king_corpora) > 1:
        for corpus in sorted(set(king_corpora) & set(challenger_corpora)):
            k, c = king_corpora[corpus], challenger_corpora[corpus]
            if len(k) < min_shard_n:
                continue
            k.assert_aligned(c)
            corpus_verdicts.append(_verdict(corpus, k, c, stats))

    shard_verdicts: list[Verdict] = []
    if per_shard:
        king_shards = king.by_shard()
        challenger_shards = challenger.by_shard()
        for shard in sorted(set(king_shards) & set(challenger_shards)):
            k, c = king_shards[shard], challenger_shards[shard]
            if len(k) < min_shard_n:
                continue
            k.assert_aligned(c)
            shard_verdicts.append(_verdict(shard, k, c, stats))

    try:
        source = policy_source()
    except Exception:  # pragma: no cover - already loaded by this point
        source = "<unknown>"

    return Comparison(
        overall=overall,
        by_shard=ShardBreakdown(tuple(shard_verdicts)),
        by_corpus=tuple(corpus_verdicts),
        king_label=king.model_label,
        challenger_label=challenger.model_label,
        sequence_set=king.sequence_set,
        policy_path=source,
    )
