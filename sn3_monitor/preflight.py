"""The submission interlock.

``teutonic-miner ready`` is irreversible and permanently consumes the hotkey's
single submission. This module refuses by default and requires every check to
pass before it reports green.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from .drift import Severity, compare
from .fetch import Document
from .target import Target
from .timeutil import age_of, humanize

DEFAULT_MARGIN = 0.02
DEFAULT_MAX_AGE = timedelta(minutes=15)
# You are judged against whoever is king when your evaluation runs, not when you
# submit. At ~57 min per evaluation, a queue that is 11 deep puts roughly ten
# hours between the two, which is two reigns. Clearing today's bar by a hair is
# how a submission dies in the queue.
DEFAULT_MU_FLOOR = 0.11


@dataclass(frozen=True)
class Check:
    """One gate in the interlock."""

    name: str
    passed: bool
    severity: Severity
    detail: str

    @property
    def blocking(self) -> bool:
        return not self.passed and self.severity >= Severity.STALE


@dataclass(frozen=True)
class PreflightResult:
    checks: tuple[Check, ...]

    @property
    def blockers(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.blocking)

    @property
    def warnings(self) -> tuple[Check, ...]:
        return tuple(
            c for c in self.checks if not c.passed and c.severity == Severity.WARN
        )

    @property
    def ok(self) -> bool:
        return not self.blockers

    @property
    def severity(self) -> Severity:
        return max((c.severity for c in self.checks if not c.passed), default=Severity.OK)


def run_preflight(
    pinned: Target,
    live: Target,
    dashboard: Document,
    *,
    offline_lcb: float | None = None,
    offline_mu: float | None = None,
    packaging: Any = None,
    margin: float = DEFAULT_MARGIN,
    mu_floor: float = DEFAULT_MU_FLOOR,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> PreflightResult:
    """Evaluate every gate. Order is presentation order, not short-circuit order."""
    board: dict[str, Any] = dashboard.data
    checks: list[Check] = []

    # 1. Is the picture we are deciding on actually current?
    age = age_of(board.get("updated_at"), reference=dashboard.fetched_at)
    fresh = age is not None and age <= max_age
    checks.append(
        Check(
            "dashboard freshness",
            fresh,
            Severity.STALE,
            f"updated {humanize(age)} ago (limit {humanize(max_age)})"
            if age is not None
            else "dashboard carries no usable updated_at",
        )
    )

    # 2-5. Contract drift, reusing the same comparison the check command uses.
    verdict = compare(pinned, live)
    contract_fields = {
        "chain.generation": "generation",
        "king.king_digest": "king digest",
        "delta_threshold": "acceptance threshold",
        "datasets.config_version": "dataset version",
    }
    moved = {d.field for d in verdict.drifts}
    for field_name, label in contract_fields.items():
        drifted = field_name in moved
        checks.append(
            Check(
                f"{label} unchanged",
                not drifted,
                Severity.ABORT if field_name == "chain.generation" else Severity.STALE,
                "moved since you pinned this target" if drifted else "matches pinned target",
            )
        )

    # 6. Somebody else's evaluation could crown a new king before yours is judged.
    current_eval = board.get("current_eval")
    checks.append(
        Check(
            "no evaluation in flight",
            current_eval is None,
            Severity.WARN,
            "an evaluation is running; a new king could be crowned before your "
            "submission is judged"
            if current_eval is not None
            else "evaluator idle",
        )
    )

    # 7. Queue depth tells you how many submissions are ahead of yours.
    queue = board.get("queue") or []
    checks.append(
        Check(
            "queue clear",
            len(queue) == 0,
            Severity.WARN,
            f"{len(queue)} submission(s) queued ahead of you"
            if queue
            else "no submissions queued",
        )
    )

    # 8. Your own measured margin against the live bar.
    live_delta = live.delta
    if offline_lcb is None:
        checks.append(
            Check(
                "offline LCB margin",
                False,
                Severity.STALE,
                "not supplied; pass --offline-lcb with your measured value",
            )
        )
    elif live_delta is None:
        checks.append(
            Check(
                "offline LCB margin",
                False,
                Severity.STALE,
                "live delta unknown; cannot compare",
            )
        )
    else:
        required = live_delta + margin
        passed = offline_lcb > required
        checks.append(
            Check(
                "offline LCB margin",
                passed,
                Severity.STALE,
                f"measured {offline_lcb:.6f} vs required {required:.6f} "
                f"(delta {live_delta} + margin {margin})",
            )
        )

    # 9. The mean, not just the bound. Under the three-corpus blend the gap
    #    between mu_hat and the lower bound tripled (median 0.001627 -> 0.005490
    #    near the threshold), so a vector that clears the bar on lcb alone is
    #    thinner than it looks -- and it has to survive a reign change first.
    if offline_mu is None:
        checks.append(
            Check(
                "offline mu_hat headroom",
                False,
                Severity.WARN,
                "not supplied; pass --offline-mu with your measured mean",
            )
        )
    else:
        checks.append(
            Check(
                "offline mu_hat headroom",
                offline_mu >= mu_floor,
                Severity.WARN,
                f"measured {offline_mu:.6f} vs floor {mu_floor:.6f}; the queue "
                "means you are judged against a later king than this one",
            )
        )

    # 10. Packaging. Roughly 1 in 11 live submissions died here rather than on
    #     model quality, and every one of those deaths happened after 'ready'
    #     had already spent the hotkey. Folding the validator in means the
    #     interlock cannot be green while the artefact is unshippable.
    if packaging is None:
        checks.append(
            Check(
                "packaging validated",
                False,
                Severity.STALE,
                "no validation report supplied; pass --model-dir so the "
                "packaging rules run as part of this gate",
            )
        )
    else:
        would_reject = bool(getattr(packaging, "would_reject", False))
        failures = list(getattr(packaging, "failures", ()) or ())
        fatal = list(getattr(packaging, "fatal_failures", ()) or ())
        if would_reject:
            detail = f"{len(failures)} failing check(s)"
            if fatal:
                detail += (
                    f", {len(fatal)} of them after 'ready' has spent the hotkey: "
                    + ", ".join(getattr(c, "name", str(c)) for c in fatal[:3])
                )
        else:
            detail = "every packaging rule that could run, passed"
        checks.append(
            Check("packaging validated", not would_reject, Severity.ABORT, detail)
        )

        skipped = list(getattr(packaging, "skipped", ()) or ())
        determinate = bool(getattr(packaging, "determinate", True))
        checks.append(
            Check(
                "packaging fully determined",
                determinate,
                Severity.WARN,
                "every rule ran"
                if determinate
                else f"{len(skipped)} rule(s) could not run; re-check with "
                "--king-digest --thorough before submitting",
            )
        )

    # 11. Winning only pays if validator weights are landing. Publication cycles
    #    through in-flight states ("claimed", "submitted") on its way to
    #    "finalized", so a mid-cycle reading is healthy provided some earlier
    #    attempt did finalize. Only a failure with nothing ever finalized is bad.
    weight_status = board.get("weight_status") or {}
    state = weight_status.get("state")
    finalized_at = weight_status.get("finalized_at")
    in_flight = state in {"claimed", "submitted", "pending"}
    healthy = state == "finalized" or (in_flight and finalized_at is not None)
    if healthy and state == "finalized":
        detail = f"last finalized {finalized_at}"
    elif healthy:
        detail = f"publication in flight (state={state!r}), last finalized {finalized_at}"
    else:
        detail = (
            f"state={state!r} error={weight_status.get('error_code')!r} "
            f"finalized_at={finalized_at!r}; emissions to a new king may not be routed"
        )
    checks.append(Check("weight publication healthy", healthy, Severity.WARN, detail))

    return PreflightResult(checks=tuple(checks))
