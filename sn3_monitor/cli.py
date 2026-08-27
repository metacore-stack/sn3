"""Command-line interface.

Exit codes are the contract other scripts branch on:
  0 fresh   1 stale   2 abort   3 fetch failed   4 usage error
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from . import __version__
from .drift import Severity, compare
from .errors import (
    EXIT_ABORT,
    EXIT_FETCH_FAILED,
    EXIT_FRESH,
    EXIT_STALE,
    EXIT_USAGE,
    FetchError,
    MonitorError,
    StaleDocumentError,
    TargetNotFoundError,
)
from .fetch import DEFAULT_TIMEOUT, fetch_dashboard, fetch_datasets
from .history import build_report, recent_table
from .observe import observation, observed_span, transitions, weight_uptime
from .preflight import DEFAULT_MARGIN, run_preflight
from .render import heading, kv, num, paint, pct, table
from .store import Store
from .target import Target
from .timeutil import humanize, iso, now, parse_duration, parse_ts


# --------------------------------------------------------------------------
# shared helpers


def _load_live(args: argparse.Namespace) -> tuple[Any, Any, Target]:
    """Fetch both documents and build a live target."""
    dashboard = fetch_dashboard(
        timeout=args.timeout,
        max_age=None if args.allow_stale else timedelta(minutes=args.max_age_minutes),
        local=Path(args.local_dashboard) if args.local_dashboard else None,
    )
    datasets = fetch_datasets(
        timeout=args.timeout,
        local=Path(args.local_datasets) if args.local_datasets else None,
    )
    return dashboard, datasets, Target.from_live(dashboard, datasets)


def _print_target(target: Target) -> None:
    print(
        kv(
            [
                ("snapshot", target.snapshot_id),
                ("pinned at", target.pinned_at),
                ("competition", target.competition),
                ("generation", target.generation),
                ("netuid", target.netuid),
                ("king repo", target.king_repo),
                ("king digest", target.king_digest),
                ("king reign", target.king_reign),
                ("king uid/hotkey", f"{target.king_uid} / {target.king_hotkey}"),
                ("king loss", num(target.king_loss, 6)),
                ("crowned at", target.king_crowned_at),
                ("delta (datasets)", target.delta_from_datasets),
                ("delta (king)", target.delta_from_king),
                ("eval n", target.eval_n),
                ("dataset version", target.dataset_version),
            ]
        )
    )
    for source in target.sources:
        print(
            kv(
                [
                    (f"source {source.name}", ""),
                    ("  manifest", source.manifest_sha256),
                    ("  tokenizer", source.tokenizer),
                    ("  seq length", source.sequence_length),
                    ("  shards", f"{source.total_shards:,}" if source.total_shards else None),
                    ("  tokens", f"{source.total_tokens:,}" if source.total_tokens else None),
                ]
            )
        )
    for note in target.notes:
        print(f"  ! {note}")


# --------------------------------------------------------------------------
# commands


def cmd_snapshot(args: argparse.Namespace) -> int:
    store = Store.open(Path(args.root) if args.root else None)
    _, _, live = _load_live(args)
    path = store.save_target(live)
    print(heading("pinned target"))
    _print_target(live)
    print(f"\nwritten to {path}")
    print(f"reference future runs with:  --against {live.snapshot_id}")
    return EXIT_FRESH


def cmd_targets(args: argparse.Namespace) -> int:
    store = Store.open(Path(args.root) if args.root else None)
    ids = store.list_targets()
    if not ids:
        print("no targets pinned yet; run 'sn3-monitor snapshot'")
        return EXIT_FRESH
    print(heading(f"pinned targets ({len(ids)})"))
    rows = []
    for snapshot_id in ids:
        target = store.load_target(snapshot_id)
        rows.append(
            [
                snapshot_id,
                str(target.king_reign),
                (target.king_digest or "?")[:16],
                str(target.delta),
                target.pinned_at or "?",
            ]
        )
    print(table(["snapshot", "reign", "digest", "delta", "pinned at"], rows))
    print(f"\nlatest: {ids[-1]}")
    return EXIT_FRESH


def cmd_status(args: argparse.Namespace) -> int:
    store = Store.open(Path(args.root) if args.root else None)
    dashboard, datasets, live = _load_live(args)
    board = dashboard.data
    report = build_report(board, since=timedelta(hours=24))

    king = board.get("king") or {}
    payout = board.get("king_payout") or {}
    market = board.get("market") or {}
    weights = board.get("weight_status") or {}
    current = board.get("current_eval")
    queue = board.get("queue") or []

    print(heading("live state"))
    age = dashboard.fetched_at - (dashboard.reported_at or dashboard.fetched_at)
    print(
        kv(
            [
                ("source", dashboard.source),
                ("dashboard age", humanize(age)),
                ("generation", live.generation),
                ("king", live.king_repo),
                ("reign", live.king_reign),
                ("digest", live.king_digest),
                ("king loss", num(live.king_loss, 6)),
                ("crowned", f"{live.king_crowned_at} ({humanize(_since(live.king_crowned_at))} ago)"),
                ("delta", live.delta),
                ("eval n", live.eval_n),
            ]
        )
    )

    print(heading("evaluator"))
    if current:
        progress = current.get("progress")
        total = current.get("total")
        share = f"{progress}/{total}" if progress and total else "?"
        print(
            kv(
                [
                    ("state", "RUNNING"),
                    ("uid", current.get("uid")),
                    ("progress", share),
                    ("provisional mu_hat", num(current.get("provisional_mu_hat"), 6)),
                    ("provisional lcb", num(current.get("provisional_lcb"), 6)),
                    ("bar", live.delta),
                ]
            )
        )
    else:
        print(kv([("state", "idle"), ("queue depth", len(queue))]))

    print(heading("economics"))
    print(
        kv(
            [
                ("payout alpha/hr", num(payout.get("alpha_per_hour"), 2)),
                ("payout usd/hr", num(payout.get("usd_per_hour"), 2)),
                ("reward weight", payout.get("weight")),
                ("alpha price (TAO)", market.get("sn3_alpha_price_tao")),
                ("TAO price (USD)", market.get("tao_price_usd")),
                ("registration burn", market.get("sn3_reg_burn_tao")),
                ("market stale", market.get("stale")),
                ("weight state", weights.get("state")),
                ("weight error", weights.get("error_code")),
                ("weight finalized", weights.get("finalized_at")),
            ]
        )
    )

    observations = store.read_observations(since=timedelta(hours=args.uptime_window))
    uptime = weight_uptime(observations)
    if uptime is not None:
        span = observed_span(observations)
        print(
            kv(
                [
                    (
                        "weight uptime",
                        f"{pct(uptime)} over {len(observations)} samples "
                        f"spanning {humanize(span)}",
                    )
                ]
            )
        )
    else:
        print(kv([("weight uptime", "no observations yet; run 'watch' to build history")]))

    print(heading("last 24h of challengers"))
    print(
        table(
            ["when", "uid", "king", "challenger", "mu_hat", "lcb", "outcome"],
            recent_table(report.attempts, limit=args.limit),
        )
    )
    best = report.best_rejected
    if best is not None and best.mu_hat is not None and live.delta:
        print(
            f"\n  best rejected attempt: {best.mu_hat:.6f} "
            f"({best.mu_hat / live.delta * 100:.0f}% of the {live.delta} bar)"
        )

    try:
        pinned = store.load_target(args.against)
    except TargetNotFoundError:
        print("\n  no pinned target yet; run 'sn3-monitor snapshot' to pin one")
        return EXIT_FRESH

    verdict = compare(pinned, live)
    print(
        f"\n  against {pinned.snapshot_id}: "
        f"{paint(verdict.severity.label, verdict.severity)}"
    )
    return verdict.exit_code


def cmd_check(args: argparse.Namespace) -> int:
    store = Store.open(Path(args.root) if args.root else None)
    pinned = store.load_target(args.against)
    _, _, live = _load_live(args)
    verdict = compare(pinned, live)

    if args.json:
        print(
            json.dumps(
                {
                    "snapshot_id": pinned.snapshot_id,
                    "severity": verdict.severity.name,
                    "verdict": verdict.severity.label,
                    "actionable": verdict.is_actionable,
                    "exit_code": verdict.exit_code,
                    "drifts": [
                        {
                            "field": d.field,
                            "pinned": d.pinned,
                            "live": d.live,
                            "severity": d.severity.name,
                            "consequence": d.consequence,
                        }
                        for d in verdict.drifts
                    ],
                },
                indent=2,
            )
        )
        return verdict.exit_code

    print(heading(f"check {pinned.snapshot_id}"))
    print(f"  {paint(verdict.severity.label, verdict.severity)}")
    if not verdict.drifts:
        print("  nothing moved since this target was pinned")
    else:
        print()
        for drift in verdict.drifts:
            print("  " + drift.render())
            print()
    return verdict.exit_code


def cmd_history(args: argparse.Namespace) -> int:
    dashboard, _, live = _load_live(args)
    since = parse_duration(args.since) if args.since else None
    report = build_report(dashboard.data, since=since)

    label = f"last {args.since}" if since else "all time"
    print(heading(f"competition — {label}"))
    print(
        kv(
            [
                ("attempts", len(report.attempts)),
                ("crowned", len(report.accepted)),
                ("scored", len(report.scored)),
                ("errored", len(report.errors)),
                ("made model worse", len(report.regressions)),
                ("packaging failure rate", pct(report.packaging_failure_rate)),
                (
                    "median eval wall time",
                    humanize(timedelta(seconds=report.median_wall_time_s))
                    if report.median_wall_time_s
                    else None,
                ),
            ]
        )
    )

    if report.error_breakdown:
        print(heading("failure codes"))
        print(table(["code", "count"], [[c, str(n)] for c, n in report.error_breakdown]))

    best = report.best_rejected
    if best is not None:
        bar = best.delta or live.delta
        print(heading("strongest rejected attempt"))
        print(
            kv(
                [
                    ("when", iso(best.timestamp)),
                    ("uid", best.uid),
                    ("mu_hat", num(best.mu_hat, 6)),
                    ("lcb", num(best.lcb, 6)),
                    ("bar", bar),
                    ("shortfall", num(best.gap_to_bar, 6)),
                    (
                        "share of bar",
                        f"{best.mu_hat / bar * 100:.1f}%"
                        if best.mu_hat is not None and bar
                        else None,
                    ),
                ]
            )
        )

    durations = report.reign_durations()
    if durations:
        print(heading("reign durations"))
        print(
            table(
                ["reign", "held"],
                [[str(r if r is not None else "?"), d] for r, d in durations],
            )
        )
        current = report.current_reign
        if current is not None:
            print(f"\n  current reign {current.reign_number} has held {humanize(current.duration)}")

    print(heading(f"attempts (newest {args.limit})"))
    print(
        table(
            ["when", "uid", "king", "challenger", "mu_hat", "lcb", "outcome"],
            recent_table(report.attempts, limit=args.limit),
        )
    )

    if args.shards:
        usage = report.shard_usage
        print(heading(f"most-drawn shards ({len(usage)} distinct)"))
        print(table(["shard", "draws"], [[s, str(n)] for s, n in usage[: args.limit]]))

    return EXIT_FRESH


def cmd_watch(args: argparse.Namespace) -> int:
    store = Store.open(Path(args.root) if args.root else None)
    pinned: Target | None = None
    try:
        pinned = store.load_target(args.against)
        print(f"watching against {pinned.snapshot_id} (king {pinned.short_digest})")
    except TargetNotFoundError:
        print("watching with no pinned target; run 'snapshot' to enable drift checks")

    previous = (store.read_observations(since=timedelta(hours=1)) or [None])[-1]
    print(f"polling every {args.interval}s — Ctrl+C to stop\n")

    consecutive_failures = 0
    try:
        while True:
            try:
                dashboard, datasets, live = _load_live(args)
            except (FetchError, StaleDocumentError) as exc:
                consecutive_failures += 1
                print(f"[{_stamp()}] fetch failed ({consecutive_failures}): {exc}")
                if args.once:
                    return EXIT_FETCH_FAILED
                time.sleep(args.interval)
                continue

            consecutive_failures = 0
            record = observation(dashboard.data, datasets.data)
            store.append_observation(record)

            for message in transitions(previous, record):
                print(f"[{_stamp()}] *** {message}")
            previous = record

            line = (
                f"[{_stamp()}] reign {record['reign_number']} "
                f"{str(record['king_digest'])[:8]} "
                f"weights={record['weight_state']} "
                f"queue={record['queue_depth']}"
            )
            if record["eval_active"]:
                line += (
                    f" eval uid={record['eval_uid']} "
                    f"{record['eval_progress']}/{record['eval_total']} "
                    f"lcb={num(record['eval_provisional_lcb'], 6)}"
                )
            print(line)

            if pinned is not None:
                verdict = compare(pinned, live)
                if not verdict.is_actionable:
                    print(
                        f"[{_stamp()}] {paint(verdict.severity.label, verdict.severity)}: "
                        + "; ".join(verdict.reasons())
                    )
                    if args.exit_on_drift:
                        return verdict.exit_code

            if args.once:
                return EXIT_FRESH
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped")
        return EXIT_FRESH


def cmd_preflight(args: argparse.Namespace) -> int:
    store = Store.open(Path(args.root) if args.root else None)
    pinned = store.load_target(args.against)
    dashboard, _, live = _load_live(args)
    result = run_preflight(
        pinned,
        live,
        dashboard,
        offline_lcb=args.offline_lcb,
        margin=args.margin,
        max_age=timedelta(minutes=args.max_age_minutes),
    )

    print(heading(f"preflight for {pinned.snapshot_id}"))
    rows = [
        ["PASS" if c.passed else c.severity.name, c.name, c.detail] for c in result.checks
    ]
    print(table(["result", "check", "detail"], rows))

    print()
    if result.ok and not result.warnings:
        print("  " + paint("CLEAR — every gate passed.", Severity.OK))
    elif result.ok:
        print("  " + paint("CLEAR WITH WARNINGS", Severity.WARN))
        for check in result.warnings:
            print(f"    - {check.name}: {check.detail}")
    else:
        print("  " + paint("BLOCKED — do not run 'teutonic-miner ready'.", Severity.ABORT))
        for check in result.blockers:
            print(f"    - {check.name}: {check.detail}")

    print(
        "\n  This tool never submits anything. Running 'ready' is a separate, "
        "irreversible action you take by hand."
    )
    if result.ok:
        return EXIT_FRESH
    return EXIT_ABORT if result.severity >= Severity.ABORT else EXIT_STALE


# --------------------------------------------------------------------------


def _stamp() -> str:
    return now().strftime("%H:%M:%S")


def _since(timestamp: str | None) -> timedelta | None:
    parsed = parse_ts(timestamp)
    return (now() - parsed) if parsed else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sn3-monitor",
        description="Track what Bittensor SN3 (Teutonic) expects of a challenger, "
        "and refuse to let a stale assumption reach an irreversible submission.",
    )
    parser.add_argument("--version", action="version", version=f"sn3-monitor {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", help="state directory (default ~/Documents/sn3/state)")
    common.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds"
    )
    common.add_argument(
        "--max-age-minutes",
        type=float,
        default=30.0,
        help="reject a dashboard older than this",
    )
    common.add_argument(
        "--allow-stale", action="store_true", help="do not enforce dashboard freshness"
    )
    common.add_argument("--local-dashboard", help="read dashboard.json from a file")
    common.add_argument("--local-datasets", help="read datasets manifest from a file")

    against = argparse.ArgumentParser(add_help=False)
    against.add_argument(
        "--against", default="latest", help="snapshot id to compare against"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("snapshot", parents=[common], help="pin the current contract")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("targets", parents=[common], help="list pinned targets")
    p.set_defaults(func=cmd_targets)

    p = sub.add_parser("status", parents=[common, against], help="one-screen live state")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument(
        "--uptime-window", type=float, default=24.0, help="hours of history for uptime"
    )
    p.set_defaults(func=cmd_status)

    p = sub.add_parser(
        "check", parents=[common, against], help="FRESH / STALE / ABORT against a target"
    )
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("history", parents=[common], help="what rivals are achieving")
    p.add_argument("--since", default=None, help="window such as 24h or 7d")
    p.add_argument("--limit", type=int, default=15)
    p.add_argument("--shards", action="store_true", help="show shard draw counts")
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("watch", parents=[common, against], help="poll and log")
    p.add_argument("--interval", type=float, default=300.0)
    p.add_argument("--once", action="store_true", help="single poll then exit")
    p.add_argument("--exit-on-drift", action="store_true")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser(
        "preflight", parents=[common, against], help="submission interlock"
    )
    p.add_argument("--offline-lcb", type=float, help="your measured offline LCB")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    p.set_defaults(func=cmd_preflight)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except StaleDocumentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("       pass --allow-stale to proceed anyway", file=sys.stderr)
        return EXIT_FETCH_FAILED
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FETCH_FAILED
    except TargetNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except MonitorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ABORT
    except KeyboardInterrupt:
        return EXIT_FRESH


if __name__ == "__main__":
    sys.exit(main())
