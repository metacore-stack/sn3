"""Command-line interface for offline loss evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .compare import compare
from .engine import LIVE_DELTA, StatsSpec, describe
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

    if result.by_corpus:
        print(f"\nper corpus ({len(result.by_corpus)})  —  the blend is fixed at 22/26/52")
        width = max(len(v.label) for v in result.by_corpus)
        for v in sorted(result.by_corpus, key=lambda x: x.mu_hat):
            flag = "  " if v.mu_hat > o.delta else " !"
            print(
                f" {flag} {v.label:<{width}}  n={v.n:<5} mu_hat={_num(v.mu_hat, 4)} "
                f"lcb={_num(v.lcb, 4)}  king={_num(v.avg_king_loss, 4)}"
            )
        weakest = result.weakest_corpus
        if weakest is not None and weakest.mu_hat <= o.delta:
            print(
                f"\n  weakest source is {weakest.label} at {weakest.mu_hat:.4f}; "
                "a fixed-blend evaluation pays for that on every submission"
            )

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
                "\n  The validator draws one shard per corpus, so the clearing\n"
                "  fraction is an estimate of a single draw's odds — not the\n"
                "  overall verdict above."
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


def _store(args):
    from .evidence import EvidenceStore

    root = Path(args.evidence or (DEFAULT_ROOT / "evidence"))
    return EvidenceStore(root).load()


def cmd_plan(args) -> int:
    """What a scoring run would measure, before it is paid for."""
    from .scoring import plan

    result = plan(args.model_dir, args.holdout, state_root=Path(args.root))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return EXIT_OK

    print(f"\n  {result.model_dir}  on holdout {result.holdout_name!r}")
    print(f"  {result.sequences} sequences\n")
    actual, expected = result.actual_share, result.expected_share
    for name in sorted(set(actual) | set(expected)):
        print(
            f"    {name:22} {result.per_corpus.get(name, 0):>6}  "
            f"{actual.get(name, 0.0):6.1%} measured vs {expected.get(name, 0.0):6.1%} scored"
        )
    print(f"\n  worst mixture error  {result.max_share_error:.1%}")
    for warning in result.warnings:
        print(f"  ! {warning}")
    if not result.warnings:
        print("  This holdout tracks what the validator scores.")
    return EXIT_OK


def cmd_score(args) -> int:
    """Score a checkpoint and, optionally, file the result as evidence."""
    from .scoring import plan, score_checkpoint

    preview = plan(args.model_dir, args.holdout, state_root=Path(args.root))
    for warning in preview.warnings:
        print(f"  ! {warning}", file=sys.stderr)
    if preview.warnings and not args.force:
        print(
            "  refusing to spend a scoring run on this holdout; pass --force to "
            "proceed anyway",
            file=sys.stderr,
        )
        return EXIT_FAILED

    state = {"last": -1}

    def progress(done: int, total: int) -> None:
        pct = int(100 * done / total) if total else 100
        if pct != state["last"]:
            state["last"] = pct
            print(f"\r  scoring {pct:3d}%  ({done}/{total})", end="", flush=True)

    vector = score_checkpoint(
        args.model_dir,
        args.holdout,
        state_root=Path(args.root),
        model_label=args.label or Path(args.model_dir).name,
        model_digest=args.digest or "",
        device_map=args.device_map,
        limit=args.limit,
        progress=None if args.quiet else progress,
        out=Path(args.out) if args.out else None,
    )
    print(f"\r  scored {len(vector)} sequences in {vector.wall_time_s:.1f}s" + " " * 12)
    print(f"  mean loss  {vector.mean:.6f}")
    for name, part in sorted(vector.by_corpus().items()):
        print(f"    {name:22} {len(part):>6}  {part.mean:.6f}")

    if args.record:
        from .evidence import Cost

        store = _store(args)
        entry = store.record(
            vector,
            run_id=args.record,
            kind="king" if args.king else "challenger",
            provenance={"model_dir": str(args.model_dir), "holdout": args.holdout},
            cost=Cost(
                gpu_hours=args.gpu_hours,
                usd_per_gpu_hour=args.usd_per_gpu_hour,
                n_gpus=args.n_gpus,
            ),
            overwrite=args.overwrite,
        )
        print(f"  recorded as {entry.run_id} ({entry.kind}) in {store.root}")
    return EXIT_OK


def cmd_evidence(args) -> int:
    store = _store(args)

    if args.evidence_command == "list":
        records = store.ordered()
        if not records:
            print(f"  no records in {store.root}")
            return EXIT_OK
        print(f"\n  {store.root}  ({len(records)} records)\n")
        for r in records:
            mark = "KING" if r.kind == "king" else "    "
            print(
                f"  {mark} {r.run_id:24} {r.model_label:28} n={r.n:<6} "
                f"mean={r.mean_loss:.6f}  {r.sequence_set}"
            )
        return EXIT_OK

    if args.evidence_command == "spend":
        print(json.dumps(store.spend(), indent=2))
        return EXIT_OK

    board = store.leaderboard(
        king_run_id=args.king_run_id, stats=_stats(args)
    )
    if args.json:
        print(json.dumps(board, indent=2))
        return EXIT_OK

    king = board["king"]
    print(f"\n  against king {king['run_id']} ({king['label']}) mean {_num(king['mean_loss'])}")
    print(f"  {board['challengers']} challengers · {board['accepted']} would be accepted")
    if board["unrankable"]:
        print(f"  {board['unrankable']} not comparable")
    print()
    print(f"  {'run':24} {'mu_hat':>10} {'lcb':>10} {'margin':>10}  {'usd':>8}  verdict")
    for row in board["rows"]:
        verdict = "ACCEPT" if row["accepted"] else (row["reason"] or "reject")
        print(
            f"  {row['run_id']:24} {_num(row['mu_hat']):>10} {_num(row['lcb']):>10} "
            f"{_num(row['margin']):>10}  {row['cost_usd']:>8.2f}  {verdict}"
        )
    print(
        "\n  Vectors are stored, verdicts are not. When the throne turns over, "
        "score\n  the new king once with --record <id> --king and run this again; "
        "no\n  challenger is re-scored."
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
    stats.add_argument(
        "--delta", type=float, default=LIVE_DELTA,
        help="live acceptance bar (0.1 since 2026-08-27)",
    )

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

    corpus = argparse.ArgumentParser(add_help=False)
    corpus.add_argument("--root", default=str(DEFAULT_ROOT), help="state directory")
    corpus.add_argument("--evidence", help="evidence store root")

    p = sub.add_parser("plan", parents=[corpus], help="check a holdout before scoring on it")
    p.add_argument("model_dir")
    p.add_argument("--holdout", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser(
        "score", parents=[corpus], help="score a checkpoint into a loss vector"
    )
    p.add_argument("model_dir")
    p.add_argument("--holdout", required=True)
    p.add_argument("--out", help="write the loss vector here")
    p.add_argument("--label", help="model label recorded in the vector")
    p.add_argument("--digest", help="model digest, if known")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--limit", type=int, help="score only the first N sequences")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--force", action="store_true", help="score despite holdout warnings")
    p.add_argument("--record", help="file the vector in the evidence store under this id")
    p.add_argument("--king", action="store_true", help="record it as the king baseline")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--gpu-hours", type=float, default=0.0)
    p.add_argument("--usd-per-gpu-hour", type=float, default=0.0)
    p.add_argument("--n-gpus", type=int, default=0)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser(
        "evidence", parents=[corpus, stats], help="stored measurements and standings"
    )
    p.add_argument(
        "evidence_command",
        nargs="?",
        default="standings",
        choices=("standings", "list", "spend"),
    )
    p.add_argument("--king-run-id", help="rank against this king instead of the latest")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_evidence)

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
