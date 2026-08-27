"""Competitive intelligence derived from the dashboard's evaluation log.

Every numeric field in a history row is nullable: rows that failed with an
``error_code`` carry ``null`` for ``mu_hat``, ``lcb``, ``avg_king_loss`` and
``n_sequences``. Nothing here assumes otherwise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

from .timeutil import humanize, now, parse_ts


@dataclass(frozen=True)
class Attempt:
    """One evaluation, normalised and null-safe."""

    timestamp: datetime | None
    uid: int | None
    hotkey: str | None
    coldkey: str | None
    baseline_uid: int | None
    accepted: bool
    verdict: str | None
    mu_hat: float | None
    lcb: float | None
    delta: float | None
    avg_king_loss: float | None
    avg_challenger_loss: float | None
    n_sequences: int | None
    wall_time_s: float | None
    early_stopped: bool
    error_code: str | None
    error_message: str | None
    dataset_version: str | None
    policy_version: str | None
    shards_used: tuple[str, ...] = ()

    @property
    def is_error(self) -> bool:
        return bool(self.error_code)

    @property
    def gap_to_bar(self) -> float | None:
        """How far short of the acceptance threshold this attempt fell."""
        if self.mu_hat is None or self.delta is None:
            return None
        return self.delta - self.mu_hat

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Attempt":
        shards = row.get("shards_used") or []
        return cls(
            timestamp=parse_ts(row.get("timestamp")),
            uid=row.get("uid"),
            hotkey=row.get("hotkey"),
            coldkey=row.get("coldkey"),
            baseline_uid=row.get("baseline_uid"),
            accepted=bool(row.get("accepted")),
            verdict=row.get("verdict"),
            mu_hat=_as_float(row.get("mu_hat")),
            lcb=_as_float(row.get("lcb")),
            delta=_as_float(row.get("delta")),
            avg_king_loss=_as_float(row.get("avg_king_loss")),
            avg_challenger_loss=_as_float(row.get("avg_challenger_loss")),
            n_sequences=row.get("n_sequences"),
            wall_time_s=_as_float(row.get("wall_time_s")),
            early_stopped=bool(row.get("early_stopped")),
            error_code=row.get("error_code"),
            error_message=row.get("error_message"),
            dataset_version=row.get("dataset_version"),
            policy_version=row.get("policy_version"),
            shards_used=tuple(s for s in shards if isinstance(s, str)),
        )


@dataclass(frozen=True)
class Reign:
    """One entry from ``king_chain``."""

    reign_number: int | None
    crowned_at: datetime | None
    ended_at: datetime | None
    uid: int | None
    model_digest: str | None
    weight: float | None
    alpha_per_hour: float | None
    usd_per_hour: float | None
    replacement_reason: str | None

    @property
    def duration(self) -> timedelta | None:
        if self.crowned_at is None:
            return None
        return (self.ended_at or now()) - self.crowned_at

    @property
    def is_current(self) -> bool:
        return self.ended_at is None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Reign":
        return cls(
            reign_number=row.get("reign_number"),
            crowned_at=parse_ts(row.get("crowned_at")),
            ended_at=parse_ts(row.get("ended_at")),
            uid=row.get("uid"),
            model_digest=row.get("model_digest"),
            weight=_as_float(row.get("weight")),
            alpha_per_hour=_as_float(row.get("alpha_per_hour")),
            usd_per_hour=_as_float(row.get("usd_per_hour")),
            replacement_reason=row.get("replacement_reason"),
        )


@dataclass
class Report:
    """Aggregate view of the competition over a window."""

    window: timedelta | None
    attempts: list[Attempt] = field(default_factory=list)
    reigns: list[Reign] = field(default_factory=list)

    @property
    def scored(self) -> list[Attempt]:
        """Attempts that produced a real measurement."""
        return [a for a in self.attempts if a.mu_hat is not None]

    @property
    def errors(self) -> list[Attempt]:
        return [a for a in self.attempts if a.is_error]

    @property
    def accepted(self) -> list[Attempt]:
        return [a for a in self.attempts if a.accepted]

    @property
    def best(self) -> Attempt | None:
        scored = self.scored
        return max(scored, key=lambda a: a.mu_hat or float("-inf")) if scored else None

    @property
    def best_rejected(self) -> Attempt | None:
        """Strongest attempt that still fell short — the real read on the field."""
        candidates = [a for a in self.scored if not a.accepted]
        return (
            max(candidates, key=lambda a: a.mu_hat or float("-inf"))
            if candidates
            else None
        )

    @property
    def regressions(self) -> list[Attempt]:
        """Attempts that made the model worse."""
        return [a for a in self.scored if (a.mu_hat or 0.0) < 0]

    @property
    def error_breakdown(self) -> list[tuple[str, int]]:
        counter = Counter(a.error_code for a in self.errors if a.error_code)
        return counter.most_common()

    @property
    def packaging_failure_rate(self) -> float | None:
        """Share of attempts that died on the contract rather than on quality."""
        if not self.attempts:
            return None
        packaging = sum(
            1
            for a in self.errors
            if a.error_code
            and a.error_code.lower() not in {"evaluator_busy", "evaluator_unavailable"}
        )
        return packaging / len(self.attempts)

    @property
    def median_wall_time_s(self) -> float | None:
        times = sorted(a.wall_time_s for a in self.attempts if a.wall_time_s)
        if not times:
            return None
        mid = len(times) // 2
        if len(times) % 2:
            return times[mid]
        return (times[mid - 1] + times[mid]) / 2

    @property
    def shard_usage(self) -> list[tuple[str, int]]:
        counter: Counter[str] = Counter()
        for attempt in self.attempts:
            counter.update(attempt.shards_used)
        return counter.most_common()

    @property
    def completed_reigns(self) -> list[Reign]:
        return [r for r in self.reigns if r.duration is not None and not r.is_current]

    @property
    def current_reign(self) -> Reign | None:
        for reign in self.reigns:
            if reign.is_current:
                return reign
        return None

    def reign_durations(self) -> list[tuple[int | None, str]]:
        return [
            (r.reign_number, humanize(r.duration))
            for r in sorted(
                self.reigns, key=lambda r: r.crowned_at or datetime.min.replace(tzinfo=None)
            )
            if r.duration is not None
        ]


def build_report(
    dashboard: dict[str, Any], *, since: timedelta | None = None
) -> Report:
    """Assemble a report from a dashboard payload."""
    cutoff = (now() - since) if since else None
    rows: Iterable[dict[str, Any]] = dashboard.get("history") or []
    attempts = [Attempt.from_row(r) for r in rows if isinstance(r, dict)]
    if cutoff is not None:
        attempts = [
            a for a in attempts if a.timestamp is not None and a.timestamp >= cutoff
        ]
    attempts.sort(key=lambda a: a.timestamp or datetime.min.replace(tzinfo=None))

    chain_rows: Iterable[dict[str, Any]] = dashboard.get("king_chain") or []
    reigns = [Reign.from_row(r) for r in chain_rows if isinstance(r, dict)]

    return Report(window=since, attempts=attempts, reigns=reigns)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def recent_table(attempts: Sequence[Attempt], limit: int = 10) -> list[list[str]]:
    """Rows for a compact terminal table, newest first."""
    rows: list[list[str]] = []
    for attempt in list(attempts)[::-1][:limit]:
        stamp = attempt.timestamp.strftime("%m-%d %H:%M") if attempt.timestamp else "?"
        if attempt.is_error:
            outcome = f"ERROR {attempt.error_code}"
        elif attempt.accepted:
            outcome = "CROWNED"
        else:
            outcome = "rejected"
        rows.append(
            [
                stamp,
                str(attempt.uid if attempt.uid is not None else "?"),
                _fmt(attempt.avg_king_loss),
                _fmt(attempt.avg_challenger_loss),
                _fmt(attempt.mu_hat),
                _fmt(attempt.lcb),
                outcome,
            ]
        )
    return rows


def _fmt(value: float | None, places: int = 4) -> str:
    return "—" if value is None else f"{value:.{places}f}"
