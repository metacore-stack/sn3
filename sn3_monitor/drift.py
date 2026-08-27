"""Drift detection: compare a pinned target against the live contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from .errors import EXIT_ABORT, EXIT_FRESH, EXIT_STALE
from .fetch import Document
from .target import Target


class Severity(IntEnum):
    """Ordered so ``max()`` yields the worst finding."""

    OK = 0
    WARN = 1
    STALE = 2
    ABORT = 3

    @property
    def label(self) -> str:
        return {
            Severity.OK: "FRESH",
            Severity.WARN: "FRESH (with warnings)",
            Severity.STALE: "STALE",
            Severity.ABORT: "ABORT",
        }[self]


@dataclass(frozen=True)
class Drift:
    """One field that moved, and what it means for work in flight."""

    field: str
    pinned: Any
    live: Any
    severity: Severity
    consequence: str

    def render(self) -> str:
        return (
            f"{self.severity.name:<5} {self.field}\n"
            f"        pinned: {_short(self.pinned)}\n"
            f"        live:   {_short(self.live)}\n"
            f"        -> {self.consequence}"
        )


@dataclass(frozen=True)
class Verdict:
    """Aggregate result of a drift check."""

    severity: Severity
    drifts: tuple[Drift, ...]
    target: Target
    live: Target

    @property
    def is_actionable(self) -> bool:
        """True when work pinned to this target may continue unchanged."""
        return self.severity <= Severity.WARN

    @property
    def exit_code(self) -> int:
        if self.severity >= Severity.ABORT:
            return EXIT_ABORT
        if self.severity >= Severity.STALE:
            return EXIT_STALE
        return EXIT_FRESH

    def reasons(self) -> list[str]:
        return [f"{d.field}: {d.consequence}" for d in self.drifts]


def _short(value: Any, width: int = 72) -> str:
    text = "None" if value is None else str(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def compare(pinned: Target, live: Target) -> Verdict:
    """Diff a pinned target against a freshly built one."""
    drifts: list[Drift] = []

    def note(field: str, old: Any, new: Any, severity: Severity, consequence: str) -> None:
        drifts.append(Drift(field, old, new, severity, consequence))

    # A generation change means the competition itself was replaced. Nothing
    # trained against the old one can be submitted.
    if _differs(pinned.generation, live.generation):
        note(
            "chain.generation",
            pinned.generation,
            live.generation,
            Severity.ABORT,
            "the competition was re-seeded; this checkpoint is unsubmittable",
        )

    if _differs(pinned.netuid, live.netuid):
        note(
            "chain.netuid",
            pinned.netuid,
            live.netuid,
            Severity.ABORT,
            "subnet id changed; verify you are pointed at the right network",
        )

    if _differs(pinned.king_digest, live.king_digest):
        note(
            "king.king_digest",
            pinned.king_digest,
            live.king_digest,
            Severity.STALE,
            "your baseline is no longer king; re-evaluate against the new one "
            "before submitting",
        )

    if _differs(pinned.delta_from_datasets, live.delta_from_datasets) or _differs(
        pinned.delta_from_king, live.delta_from_king
    ):
        note(
            "delta_threshold",
            f"king={pinned.delta_from_king} datasets={pinned.delta_from_datasets}",
            f"king={live.delta_from_king} datasets={live.delta_from_datasets}",
            Severity.STALE,
            "the acceptance bar moved; re-judge every candidate checkpoint",
        )

    if _differs(pinned.dataset_version, live.dataset_version):
        note(
            "datasets.config_version",
            pinned.dataset_version,
            live.dataset_version,
            Severity.STALE,
            "evaluation configuration changed; refresh your data assumptions",
        )

    if _differs(pinned.eval_n, live.eval_n):
        note(
            "datasets.eval_n",
            pinned.eval_n,
            live.eval_n,
            Severity.WARN,
            "sample size changed; offline evaluation cost changes with it",
        )

    drifts.extend(_compare_sources(pinned, live))

    # Reign numbers are not contiguous upstream, so a gap is informational.
    if (
        pinned.king_reign is not None
        and live.king_reign is not None
        and live.king_reign > pinned.king_reign + 1
    ):
        note(
            "king.reign_number",
            pinned.king_reign,
            live.king_reign,
            Severity.WARN,
            f"{live.king_reign - pinned.king_reign} reigns passed; you are further "
            "behind than a single dethronement",
        )

    if live.delta_disagrees:
        note(
            "delta consistency",
            None,
            f"king={live.delta_from_king} datasets={live.delta_from_datasets}",
            Severity.WARN,
            "the two published thresholds disagree; investigate before trusting either",
        )

    severity = max((d.severity for d in drifts), default=Severity.OK)
    return Verdict(
        severity=severity, drifts=tuple(drifts), target=pinned, live=live
    )


def _compare_sources(pinned: Target, live: Target) -> list[Drift]:
    """Compare evaluation corpora by name."""
    drifts: list[Drift] = []
    pinned_by_name = {s.name: s for s in pinned.sources}
    live_by_name = {s.name: s for s in live.sources}

    for name in sorted(set(pinned_by_name) | set(live_by_name), key=lambda n: n or ""):
        before = pinned_by_name.get(name)
        after = live_by_name.get(name)
        if before is None:
            drifts.append(
                Drift(
                    f"datasets.sources[{name}]",
                    None,
                    name,
                    Severity.STALE,
                    "a new evaluation corpus appeared",
                )
            )
            continue
        if after is None:
            drifts.append(
                Drift(
                    f"datasets.sources[{name}]",
                    name,
                    None,
                    Severity.STALE,
                    "an evaluation corpus was removed",
                )
            )
            continue
        if _differs(before.manifest_sha256, after.manifest_sha256):
            drifts.append(
                Drift(
                    f"datasets.sources[{name}].manifest_sha256",
                    before.manifest_sha256,
                    after.manifest_sha256,
                    Severity.STALE,
                    "shard inventory changed; pinned sequence ids may not resolve",
                )
            )
        for attr, consequence in (
            ("tokenizer", "tokenizer changed; every packed sequence must be rebuilt"),
            ("sequence_length", "sequence length changed; repack training data"),
        ):
            if _differs(getattr(before, attr), getattr(after, attr)):
                drifts.append(
                    Drift(
                        f"datasets.sources[{name}].{attr}",
                        getattr(before, attr),
                        getattr(after, attr),
                        Severity.ABORT,
                        consequence,
                    )
                )
    return drifts


def _differs(old: Any, new: Any) -> bool:
    """Compare, treating a missing live value as 'unknown' rather than 'changed'."""
    if old is None and new is None:
        return False
    if new is None:
        return False
    return old != new


def live_target(dashboard: Document, datasets: Document) -> Target:
    """Convenience wrapper so callers do not import Target directly."""
    return Target.from_live(dashboard, datasets)
