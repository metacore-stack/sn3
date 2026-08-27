"""Command-line interface for the controller."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import STAGES, CampaignConfig
from .errors import EXIT_BLOCKED, EXIT_FAILED, EXIT_OK, EXIT_USAGE, CampaignError
from .runner import Campaign


def cmd_init(args) -> int:
    config = CampaignConfig.starter(args.name)
    if args.king_digest:
        config.king_digest = args.king_digest
    if args.gpus:
        config.hardware.n_gpus = args.gpus
        config.hardware.torchrun = args.gpus > 1
    if args.usd_per_gpu_hour:
        config.hardware.usd_per_gpu_hour = args.usd_per_gpu_hour
    path = Path(args.out or f"{args.name}.json")
    if path.exists() and not args.force:
        print(f"{path} exists; pass --force to overwrite", file=sys.stderr)
        return EXIT_USAGE
    config.save(path)
    print(f"wrote {path}")
    print("edit it, then:  campaign run " + str(path))
    return EXIT_OK


def _render(result) -> None:
    print()
    width = max((len(s.name) for s in result.stages), default=8)
    for stage in result.stages:
        mark = {
            "ok": "  ok  ", "skipped": " skip ", "blocked": "BLOCK ", "failed": "FAIL  "
        }.get(stage.status, stage.status)
        print(f"  {mark} {stage.name.ljust(width)}  {stage.seconds:7.1f}s  {stage.detail}")

    cost = result.cost()
    print(
        f"\n  {cost['billable_hours']:.2f} billable hours on {cost['n_gpus']} GPU(s)"
        f"  ·  {cost['gpu_hours']:.2f} GPU-hours  ·  ${cost['usd']:.2f}"
    )
    print(f"  {cost['wall_hours_per_attempt']:.2f} wall hours for this attempt")

    report = result.stage("report")
    if report and report.data.get("spend"):
        spend = report.data["spend"]
        if spend.get("usd_per_nat"):
            print(
                f"  ${spend['usd_per_nat']:.2f} per nat of improvement across "
                f"{spend['runs']} run(s), ${spend['usd']:.2f} spent"
            )

    state = result.stage("state")
    if state and state.data.get("warning"):
        print(f"\n  ! {state.data['warning']}")

    if result.failed:
        print("\n  " + "BLOCKED — " + ", ".join(s.name for s in result.failed))
        for stage in result.failed:
            for line in (stage.data.get("fatal") or [])[:5]:
                print(f"      fatal: {line}")
    else:
        print("\n  Every stage completed.")

    print(
        "\n  This controller never runs 'teutonic-miner ready'. That call is\n"
        "  irreversible and permanently consumes the hotkey; make it by hand,\n"
        "  after 'sn3 preflight --model-dir ...' is green."
    )


def cmd_run(args) -> int:
    config = CampaignConfig.load(args.config)
    campaign = Campaign(config)
    print(f"\ncampaign '{config.name}' → {config.run_dir}")
    result = campaign.run(
        only=args.only,
        skip=args.skip or (),
        resume=not args.no_resume,
        stop_on_failure=not args.keep_going,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _render(result)
        print(f"\n  journal: {config.journal_path}")

    if result.failed:
        return EXIT_BLOCKED if any(s.status == "blocked" for s in result.failed) else EXIT_FAILED
    return EXIT_OK


def cmd_status(args) -> int:
    config = CampaignConfig.load(args.config)
    path = config.journal_path
    if not path.is_file():
        print(f"no journal at {path}; nothing has run yet")
        return EXIT_OK
    payload = json.loads(path.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(payload, indent=2))
        return EXIT_OK
    print(f"\ncampaign '{payload['campaign']}'  started {payload['started']}")
    for stage in payload["stages"]:
        print(f"  {stage['status']:8} {stage['name']:10} {stage['seconds']:8.1f}s  {stage['detail']}")
    cost = payload.get("cost") or {}
    print(f"\n  ${cost.get('usd', 0):.2f} over {cost.get('gpu_hours', 0):.2f} GPU-hours")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="campaign",
        description="Run one attempt at the throne end to end: state, data, "
        "train, score, validate, report.",
    )
    parser.add_argument("--version", action="version", version=f"campaign {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="write a starter campaign file")
    p.add_argument("--name", default="attempt-001")
    p.add_argument("--out")
    p.add_argument("--king-digest")
    p.add_argument("--gpus", type=int)
    p.add_argument("--usd-per-gpu-hour", type=float)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("run", help="run the campaign")
    p.add_argument("config")
    p.add_argument("--only", action="append", choices=STAGES, help="run only this stage (repeatable)")
    p.add_argument("--skip", action="append", choices=STAGES, help="skip this stage (repeatable)")
    p.add_argument("--no-resume", action="store_true", help="re-run completed stages")
    p.add_argument("--keep-going", action="store_true", help="continue past a failed stage")
    p.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="show the journal for a campaign")
    p.add_argument("config")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except CampaignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("\ninterrupted; the journal records what completed", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
