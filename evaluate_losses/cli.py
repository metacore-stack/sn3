"""Command-line interface for offline loss evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .compare import compare
from .engine import StatsSpec, describe
from .errors import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_USAGE,
    AlignmentError,
    EvaluationError,
    PolicyUnavailableError,
)
from .lossvec import LossVector
from .parity import run_offline, tier1_statistics, tier2_sampler
from .policy import policy_source

DEFAULT_ROOT = Path.home() / "Documents" / "sn3" / "state"


def _stats(args) -> StatsSpec:
    return StatsSpec(
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        delta_threshold=args.delta,
    )


def _num(x: float | None, places: int = 6) -> str:
    return "—" if x is None else f"{x:.{places}f}"


def cmd_contract(args) -> int:
    """Print the validator contract this package targets."""
    payload = describe()
    try:
        payload["policy_path"] = policy_source()
    except PolicyUnavailableError as exc:
        payload["policy_path"] = f"UNAVAILABLE: {exc}"
    print(json.dumps(payload, indent=2))
    return EXIT_OK


def cmd_compare(args) -> int:
    king = LossVector.load(Path(args.king))
    challenger = LossVector.load(Path(args.challenger))

    differences = king.engine_differences(challenger)
    if differences:
        print("warning: vectors were scored with different engine settings:")
        for line in differences:
            print(f"  ! {line}")
        print()

    result = compare(king, challenger, stats=_stats(args), per_shard=not args.no_shards)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return EXIT_OK if result.overall.accepted else EXIT_REJECTED

    o = result.overall
    print(f"{king.model_label}  vs  {challenger.model_label}")
    print(f"  sequences        {o.n:,}")
    print(f"  king loss        {_num(o.avg_king_loss)}")
    print(f"  challenger loss  {_num(o.avg_challenger_loss)}")
    print(f"  mu_hat           {_num(o.mu_hat)}")
    print(f"  lcb              {_num(o.lcb)}")
    print(f"  bar (delta)      {_num(o.delta)}")
    print(f"  margin           {_num(o.margin)}")
    print(f"  bootstrap cost   {_num(o.bootstrap_penalty)}   (mu_hat - lcb)")
    print(f"  verdict          {'ACCEPT' if o.accepted else 'reject'}")

    shards = result.by_shard
    if shards.shards:
        print(f"\nper shard ({len(shards.shards)})")
        rows = sorted(shards.shards, key=lambda v: v.mu_hat)
        width = max(len(v.label) for v in rows)
        for v in rows[: args.limit]:
            flag = "  " if v.mu_hat > o.delta else " !"
            print(
                f" {flag} {v.label:<{width}}  n={v.n:<5} mu_hat={_num(v.mu_hat, 4)} "
                f"lcb={_num(v.lcb, 4)}"
            )
        if len(rows) > args.limit:
            print(f"    … {len(rows) - args.limit} more")
        print(f"\n  median mu_hat    {_num(shards.median)}")
        print(f"  spread           {_num(shards.spread)}")
        print(f"  stdev            {_num(shards.stdev)}")
        clearing = shards.clearing_fraction(o.delta)
        if clearing is not None:
            print(f"  shards clearing  {clearing * 100:.0f}%")
            print(
                "\n  The validator draws every sequence of an evaluation from ONE\n"
                "  shard, so treat the clearing fraction as your odds on a single\n"
                "  draw — not the overall verdict above."
            )
    return EXIT_OK if result.overall.accepted else EXIT_REJECTED


def cmd_show(args) -> int:
    vector = LossVector.load(Path(args.path))
    payload = vector.to_dict()
    payload.pop("losses")
    payload.pop("refs")
    print(json.dumps(payload, indent=2))
    print(f"\nshards ({len(vector.shards())}):")
    for shard, sub in list(vector.by_shard().items())[: args.limit]:
        print(f"  {shard}  n={len(sub):<5} mean={_num(sub.mean)}")
    return EXIT_OK


def cmd_parity(args) -> int:
    reports = []
    if args.stats:
        reports.append(tier1_statistics(stats=_stats(args)))
    if args.sampler:
        reports.append(tier2_sampler())
    if not reports:
        reports = run_offline()

    failed = False
    for report in reports:
        print(f"\ntier: {report.tier}")
        for check in report.checks:
            mark = "PASS" if check.passed else "FAIL"
            print(f"  {mark}  {check.name}")
            if check.detail and (args.verbose or not check.passed):
                print(f"        {check.detail}")
        failed = failed or not report.ok

    print()
    if failed:
        print("  PARITY FAILED — do not trust numbers from this package.")
        return EXIT_FAILED
    print("  offline parity OK.")
    print(
        "  Tier 3 (per-sequence loss vs the engine on real weights) still needs\n"
        "  a GPU run. Until it passes, losses produced here are unverified."
    )
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Reproduce the SN3 validator's paired-bootstrap decision offline.",
    )
    parser.add_argument("--version", action="version", version=f"evaluate-losses {__version__}")

    stats = argparse.ArgumentParser(add_help=False)
    stats.add_argument("--alpha", type=float, default=0.001)
    stats.add_argument("--n-bootstrap", type=int, default=10000)
    stats.add_argument("--bootstrap-seed", type=int, default=0xB007)
    stats.add_argument("--delta", type=float, default=0.5, help="live acceptance bar")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("contract", help="print the validator contract targeted")
    p.set_defaults(func=cmd_contract)

    p = sub.add_parser("compare", parents=[stats], help="paired comparison of two vectors")
    p.add_argument("--king", required=True)
    p.add_argument("--challenger", required=True)
    p.add_argument("--no-shards", action="store_true")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("show", help="summarise a saved loss vector")
    p.add_argument("path")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("parity", parents=[stats], help="prove agreement with the validator")
    p.add_argument("--stats", action="store_true", help="tier 1 only")
    p.add_argument("--sampler", action="store_true", help="tier 2 only")
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=cmd_parity)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except AlignmentError as exc:
        print(f"alignment error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except PolicyUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
