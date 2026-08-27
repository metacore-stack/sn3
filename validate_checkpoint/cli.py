"""Command-line interface for checkpoint validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .contract import Contract
from .errors import (
    EXIT_CLEAN,
    EXIT_UNDETERMINED,
    EXIT_USAGE,
    EXIT_WOULD_REJECT,
    ContractError,
    KingUnavailableError,
    ValidationError,
)
from .king import KingReference
from .report import Report, Status
from .validator import Options, validate

DEFAULT_CACHE = Path.home() / "Documents" / "sn3" / "state" / "king-reference"


def _resolve_king(args) -> KingReference | None:
    if not args.king and not args.king_digest:
        return None
    return KingReference.resolve(
        directory=args.king, digest=args.king_digest, cache_dir=DEFAULT_CACHE
    )


def _render(report: Report, *, verbose: bool) -> None:
    width = max((len(c.name) for c in report.checks), default=10)
    print(f"\n{report.model_dir}")
    ctx = report.context
    print(
        f"  generation {ctx.get('generation')}  ·  {ctx.get('locked_config_keys')} locked keys"
        f"  ·  king {ctx.get('king_digest') or ctx.get('king') or 'not supplied'}"
    )
    print()
    for check in report.checks:
        mark = check.status.label
        flag = " !" if check.fatal else "  "
        print(f" {flag}{mark}  {check.name.ljust(width)}  [{check.layer.label}]  {check.detail}")
        if check.items and (verbose or check.status is Status.FAIL):
            for item in check.items:
                print(f"          - {item}")

    counts = report.to_dict()["counts"]
    print(
        f"\n  {counts['passed']} passed · {counts['failed']} failed · "
        f"{counts['warned']} warned · {counts['skipped']} skipped"
    )

    if report.failures:
        print(f"\n  {report.verdict}")
        if report.error_codes:
            print(f"  would surface as: {', '.join(report.error_codes)}")
        fatal = report.fatal_failures
        if fatal:
            print(
                f"\n  {len(fatal)} failure(s) occur AFTER 'ready', which permanently\n"
                "  consumes the hotkey. Fix these before submitting."
            )
    elif report.skipped:
        print(f"\n  {report.verdict}")
        print("  Supply --king-digest and pass --hash-shards --finite for a full answer.")
    else:
        print(f"\n  {report.verdict} — every check ran and passed.")

    print(
        "\n  This tool never uploads or submits. 'teutonic-miner ready' remains a\n"
        "  separate, irreversible action you take by hand."
    )


def cmd_check(args) -> int:
    contract = Contract.load(args.chain)
    king = _resolve_king(args)
    from .reuse import SubmissionLedger

    ledger = None if args.no_ledger else SubmissionLedger.load(DEFAULT_CACHE.parent)
    base = (
        Options.thorough(args.name)
        if args.thorough
        else Options(hash_shards=args.hash_shards, finite=args.finite, model_name=args.name)
    )
    options = Options(
        hash_shards=base.hash_shards,
        finite=base.finite,
        model_name=base.model_name,
        ledger=ledger,
    )
    report = validate(args.model_dir, contract=contract, king=king, options=options)

    if args.json:
        print(report.to_json())
    else:
        _render(report, verbose=args.verbose)

    if report.would_reject:
        return EXIT_WOULD_REJECT
    return EXIT_CLEAN if report.determinate else EXIT_UNDETERMINED


def cmd_contract(args) -> int:
    contract = Contract.load(args.chain)
    print(f"chain.toml        {contract.path}")
    print(f"generation        {contract.name}")
    print(f"seed repo         {contract.seed_repo}")
    print(f"repo pattern      {contract.repo_pattern}")
    print(f"arch module       {contract.arch_module}")
    print(f"model type        {contract.model_type}")
    print(f"allowed code      {', '.join(contract.allowed_code_files)}")
    print(f"eval n            {contract.eval_n}")
    print(f"delta threshold   {contract.delta_threshold}")
    print(f"dataset           {contract.dataset_label}")
    print(f"initial uids      {list(contract.initial_weight_uids)}")
    print(f"\ncontract files ({len(contract.contract_files)}):")
    for name, digest in sorted(contract.contract_files.items()):
        print(f"  {name:26} {digest}")
    print(f"\nlocked config keys ({contract.n_locked_keys}):")
    keys = contract.locked_config_keys
    for i in range(0, len(keys), 3):
        print("  " + "".join(k.ljust(30) for k in keys[i : i + 3]))
    return EXIT_CLEAN


def cmd_king(args) -> int:
    king = _resolve_king(args)
    if king is None:
        print("supply --king-digest or --king", file=sys.stderr)
        return EXIT_USAGE
    print(f"source        {king.source}")
    print(f"digest        {king.digest or '(local)'}")
    print(f"model name    {king.model_name or '(unknown)'}")
    print(f"files         {len(king.files)}")
    print(f"total size    {king.total_bytes / 1e9:.2f} GB")
    print(f"shards        {len(king.shard_names)}")
    print(f"tensors       {len(king.weight_map):,}")
    print("\nfiles:")
    for path in king.paths:
        entry = king.files[path]
        print(f"  {path:38} {entry.size:>14,}  {entry.sha256[:16]}…")
    return EXIT_CLEAN


def cmd_download(args) -> int:
    """Fetch the king's weights. 220 GB, so it reports before it starts."""
    from .download import KingDownloader, human

    king = _resolve_king(args)
    if king is None:
        print("supply --king-digest", file=sys.stderr)
        return EXIT_USAGE

    downloader = KingDownloader(king, Path(args.dest), workers=args.workers)
    plan = downloader.plan(verify_present=args.verify_present)

    print(f"\n  king      {king.digest or '(local)'}")
    print(f"  files     {len(plan.present)} present, {len(plan.missing)} missing"
          f"{f', {len(plan.partial)} resumable' if plan.partial else ''}")
    print(f"  to fetch  {human(plan.bytes_needed)} of {human(plan.total_bytes)}")
    print(f"  free      {human(plan.free_bytes)} at {plan.destination}")
    if not plan.enough_disk:
        print(f"\n  NOT ENOUGH DISK — need {human(plan.bytes_needed)} plus margin.")
        if not args.force:
            return EXIT_WOULD_REJECT
    if args.dry_run:
        print("\n  --dry-run: nothing fetched.")
        return EXIT_CLEAN
    if not plan.missing:
        print("\n  Nothing to do; every file is present.")
        return EXIT_CLEAN

    print()
    state = {"last": -1}

    def on_bytes(done: int, total: int) -> None:
        pct = int(100 * done / total) if total else 100
        if pct != state["last"]:
            state["last"] = pct
            print(f"\r  {pct:3d}%  {human(done)} / {human(total)}", end="", flush=True)

    def on_file(progress) -> None:
        mark = "ok  " if progress.complete else "FAIL"
        resumed = f" (resumed at {human(progress.resumed_from)})" if progress.resumed_from else ""
        print(f"\r  {mark} {progress.path}{resumed}{' ' * 20}")

    results = downloader.fetch(
        on_file=on_file, on_bytes=on_bytes, allow_insufficient_disk=args.force
    )
    verified = sum(1 for r in results if r.complete)
    print(f"\n  {verified}/{len(results)} files verified against the manifest.")
    return EXIT_CLEAN


def cmd_verify(args) -> int:
    """Re-hash a local checkpoint against the published manifest."""
    from .download import KingDownloader

    king = _resolve_king(args)
    if king is None:
        print("supply --king-digest", file=sys.stderr)
        return EXIT_USAGE
    downloader = KingDownloader(king, Path(args.dest), workers=args.workers)
    bad = downloader.verify()
    if not bad:
        print(f"  all {len(king.files)} files match the manifest.")
        return EXIT_CLEAN
    print(f"  {len(bad)} file(s) missing or corrupt:")
    for path in bad:
        print(f"    - {path}")
    print("\n  Re-run 'download' to repair; matching files are not re-fetched.")
    return EXIT_WOULD_REJECT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate",
        description="Run every rule the SN3 validator will run, before 'ready' "
        "spends the hotkey.",
    )
    parser.add_argument("--version", action="version", version=f"validate-checkpoint {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--chain", help="path to chain.toml (default: autodetect)")
    common.add_argument("--king", help="local king checkpoint directory")
    common.add_argument("--king-digest", help="fetch the king's small reference files by digest")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", parents=[common], help="validate a checkpoint directory")
    p.add_argument("model_dir")
    p.add_argument("--name", help="intended submission name, checked against repo_pattern")
    p.add_argument("--hash-shards", action="store_true", help="hash weights for copy detection")
    p.add_argument("--finite", action="store_true", help="scan every weight for NaN/Inf (slow)")
    p.add_argument("--thorough", action="store_true", help="everything, including slow checks")
    p.add_argument("--json", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--no-ledger", action="store_true", help="skip the local submission ledger")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("contract", parents=[common], help="print the enforced contract")
    p.set_defaults(func=cmd_contract)

    p = sub.add_parser("king", parents=[common], help="show the king's reference inventory")
    p.set_defaults(func=cmd_king)

    p = sub.add_parser(
        "download", parents=[common], help="fetch the king's weights (resumable, verified)"
    )
    p.add_argument("dest", help="destination directory")
    p.add_argument("--workers", type=int, default=4, help="parallel file downloads")
    p.add_argument("--dry-run", action="store_true", help="report the plan and stop")
    p.add_argument("--force", action="store_true", help="proceed despite a low disk estimate")
    p.add_argument(
        "--verify-present", action="store_true", help="re-hash files already on disk"
    )
    p.set_defaults(func=cmd_download)

    p = sub.add_parser(
        "verify", parents=[common], help="re-hash a local checkpoint against the manifest"
    )
    p.add_argument("dest", help="directory holding the checkpoint")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ContractError, KingUnavailableError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_UNDETERMINED
    except KeyboardInterrupt:
        return EXIT_UNDETERMINED


if __name__ == "__main__":
    sys.exit(main())
