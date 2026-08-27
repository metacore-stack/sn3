"""Command-line interface for the MiMo routing adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .errors import EXIT_FAILED, EXIT_OK, EXIT_USAGE, AdapterError, ParityError
from .loader import fetch_arch, find_arch_directory, load_arch, read_reference_config
from .miniature import MiniatureSpec, build_miniature, count_parameters, describe
from .verify import run_all

KING_DIGEST_HINT = "c345e657a37e82f785b61762516d6405db48272fa5e023ea09f2147f2616e6d0"


def _spec(args) -> MiniatureSpec:
    overrides = {}
    for field in ("num_hidden_layers", "hidden_size", "n_routed_experts", "num_experts_per_tok"):
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
    return MiniatureSpec(**overrides)


def _prepare(args):
    directory = Path(args.arch) if args.arch else None
    if args.king_digest:
        directory = fetch_arch(args.king_digest)
    arch = load_arch(directory)
    reference = read_reference_config(arch.directory)
    model, config = build_miniature(arch, reference, _spec(args), seed=args.seed)
    return arch, model, config


def cmd_fetch(args) -> int:
    path = fetch_arch(args.king_digest)
    print(f"architecture files in {path}")
    for name in sorted(p.name for p in path.iterdir()):
        size = (path / name).stat().st_size
        print(f"  {name:28} {size:>9,} bytes")
    return EXIT_OK


def cmd_info(args) -> int:
    arch, model, config = _prepare(args)
    print(f"architecture      {arch.directory}")
    print(f"parameters        {count_parameters(model):,}")
    print("\nrouting settings inherited from the king:")
    for key, value in describe(config).items():
        print(f"  {key:24} {value}")
    print("\ngate modules:", len([m for m in model.modules() if type(m).__name__ == "MiMoV2MoEGate"]))
    return EXIT_OK


def cmd_verify(args) -> int:
    arch, model, config = _prepare(args)
    try:
        report = run_all(arch, model, config, include_slow=not args.fast)
    except ParityError as exc:
        print(f"\nPARITY FAILURE: {exc}", file=sys.stderr)
        return EXIT_FAILED

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return EXIT_OK if report.ok else EXIT_FAILED

    width = max(len(c.name) for c in report.checks)
    print(f"\nminiature: {count_parameters(model):,} parameters, "
          f"{config.num_hidden_layers} layers, {config.n_routed_experts} experts, "
          f"top-{config.num_experts_per_tok}")
    print(f"architecture: {arch.directory}\n")
    for check in report.checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"  {mark}  {check.name.ljust(width)}  {check.detail}")

    print()
    if report.ok:
        print("  The patched router is numerically identical in eval, differentiable")
        print("  in training, and produces checkpoints that load under the unpatched")
        print("  code. Verified on a miniature — confirm on real weights before a")
        print("  full run.")
        return EXIT_OK
    print(f"  {len(report.failures)} check(s) failed:")
    for check in report.failures:
        print(f"    - {check.name}: {check.detail}")
    return EXIT_FAILED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mimo-adapter",
        description="Make the locked MiMo noaux_tc router trainable, and prove "
        "the patched version matches the original.",
    )
    parser.add_argument("--version", action="version", version=f"mimo-adapter {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--arch", help="directory holding the two architecture files")
    common.add_argument("--king-digest", help=f"download them by digest (e.g. {KING_DIGEST_HINT[:12]}…)")

    mini = argparse.ArgumentParser(add_help=False)
    mini.add_argument("--num-hidden-layers", type=int)
    mini.add_argument("--hidden-size", type=int)
    mini.add_argument("--n-routed-experts", type=int)
    mini.add_argument("--num-experts-per-tok", type=int)
    mini.add_argument("--seed", type=int, default=0)

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download the architecture files by digest")
    p.add_argument("--king-digest", required=True)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("info", parents=[common, mini], help="describe the miniature")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("verify", parents=[common, mini], help="run every parity and gradient check")
    p.add_argument("--fast", action="store_true", help="skip the save/reload round-trip")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
