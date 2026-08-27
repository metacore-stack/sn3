"""Turning a dashboard payload into one compact observation record.

The observation log is what makes time-series questions answerable: weight
publication uptime, reign durations, how long evaluations actually take.
A single `state` reading answers none of them.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from .timeutil import iso, now, parse_ts

# Fields compared between consecutive polls to decide whether to alert.
TRANSITION_FIELDS = (
    "king_digest",
    "reign_number",
    "generation",
    "dataset_version",
    "delta",
    "weight_state",
    "queue_depth",
    "eval_active",
)


def observation(board: dict[str, Any], datasets: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flatten the parts of a dashboard worth keeping forever."""
    king = board.get("king") or {}
    chain = board.get("chain") or {}
    market = board.get("market") or {}
    payout = board.get("king_payout") or {}
    weights = board.get("weight_status") or {}
    current = board.get("current_eval") or {}
    queue = board.get("queue") or []
    datasets = datasets or {}

    return {
        "ts": iso(now()),
        "updated_at": board.get("updated_at"),
        "king_digest": king.get("king_digest") or king.get("model_digest"),
        "reign_number": king.get("reign_number"),
        "king_uid": king.get("uid"),
        "king_loss": king.get("avg_challenger_loss"),
        "generation": chain.get("generation"),
        "delta": king.get("delta"),
        "dataset_version": datasets.get("config_version"),
        "weight_state": weights.get("state"),
        "weight_error": weights.get("error_code"),
        "weight_finalized_at": weights.get("finalized_at"),
        "queue_depth": len(queue),
        "eval_active": bool(current),
        "eval_uid": current.get("uid"),
        "eval_progress": current.get("progress"),
        "eval_total": current.get("total"),
        "eval_provisional_mu_hat": current.get("provisional_mu_hat"),
        "eval_provisional_lcb": current.get("provisional_lcb"),
        "alpha_price_tao": market.get("sn3_alpha_price_tao"),
        "tao_price_usd": market.get("tao_price_usd"),
        "reg_burn_tao": market.get("sn3_reg_burn_tao"),
        "market_stale": market.get("stale"),
        "payout_alpha_per_hour": payout.get("alpha_per_hour"),
        "payout_usd_per_hour": payout.get("usd_per_hour"),
        "payout_weight": payout.get("weight"),
    }


def transitions(previous: dict[str, Any] | None, current: dict[str, Any]) -> list[str]:
    """Human-readable descriptions of what changed since the last poll."""
    if previous is None:
        return []
    messages: list[str] = []
    for field in TRANSITION_FIELDS:
        before, after = previous.get(field), current.get(field)
        if before == after:
            continue
        if field == "king_digest":
            messages.append(
                f"NEW KING: {_short(before)} -> {_short(after)} "
                f"(reign {previous.get('reign_number')} -> {current.get('reign_number')})"
            )
        elif field == "generation":
            messages.append(f"GENERATION CHANGED: {before} -> {after}")
        elif field == "delta":
            messages.append(f"THRESHOLD CHANGED: {before} -> {after}")
        elif field == "dataset_version":
            messages.append(f"DATASET VERSION CHANGED: {_short(before)} -> {_short(after)}")
        elif field == "weight_state":
            messages.append(f"weight publication: {before} -> {after}")
        elif field == "eval_active":
            messages.append(
                f"evaluation started (uid {current.get('eval_uid')})"
                if after
                else "evaluation finished"
            )
        elif field == "queue_depth":
            messages.append(f"queue depth: {before} -> {after}")
        elif field == "reign_number":
            continue  # already reported alongside the digest change
    return messages


def weight_uptime(records: list[dict[str, Any]]) -> float | None:
    """Fraction of observations in which weight publication was finalized."""
    states = [r.get("weight_state") for r in records if r.get("weight_state")]
    if not states:
        return None
    return sum(1 for s in states if s == "finalized") / len(states)


def observed_span(records: list[dict[str, Any]]) -> timedelta | None:
    """Wall-clock span covered by an observation list."""
    stamps = [parse_ts(r.get("ts")) for r in records]
    stamps = [s for s in stamps if s is not None]
    if len(stamps) < 2:
        return None
    return max(stamps) - min(stamps)


def _short(value: Any, width: int = 12) -> str:
    text = "None" if value is None else str(value)
    return text if len(text) <= width else text[:width] + "…"
