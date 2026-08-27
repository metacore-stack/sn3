"""Command-line interface for continued pretraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .checkpoint import latest_checkpoint, prune_checkpoints, verify_locked_files
from .config import STAGES, DataConfig, TrainingConfig
from .errors import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    ConfigError,
    DivergenceError,
    TrainingError,
)
from .sources import resolve_batches
from .trainer import Trainer

STATE_ROOT = Path.home() / "Documents" / "sn3" / "state"


def _load_config(args) -> TrainingConfig:
    config = TrainingConfig.load(args.config) if args.config else TrainingConfig()
    for attr in ("run_name", "max_steps", "stage", "save_every", "log_every"):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(config, attr, value)
    if getattr(args, "learning_rate", None) is not None:
        config.optim.learning_rate = args.learning_rate
    if getattr(args, "batch_size", None) is not None:
        config.data.batch_size = args.batch_size
    if getattr(args, "grad_accum", None) is not None:
        config.data.grad_accum = args.grad_accum
    if getattr(args, "shard", None):
        config.data.shards = tuple(args.shard)
    if getattr(args, "holdout", None):
        config.data.holdouts = tuple(args.holdout)
    if getattr(args, "no_balance", False):
        config.balance.enabled = False
    config.__post_init__()
    return config


def _build_model(args, resume_from: Path | None = None):
    """The miniature by default; a checkpoint when resuming or when one is given.

    Resuming *must* load the weights, not just the optimizer state. Restoring
    one without the other silently restarts training from the initial model
    while pretending to continue, which shows up only as a loss curve that
    jumps backwards.
    """
    from mimo_adapter import MiniatureSpec, build_miniature, load_arch, read_reference_config

    arch = load_arch(args.arch)
    reference = read_reference_config(arch.directory)

    source = resume_from or (Path(args.model_dir) if args.model_dir else None)
    if source is not None:
        import torch

        # local_files_only: the restored config.json carries auto_map, which
        # otherwise sends transformers to the Hub and hangs on a box with no
        # network — or worse, mid-run on a rented node.
        model = arch.causal_lm_cls.from_pretrained(
            source, dtype=torch.float32, local_files_only=True
        )
        king = Path(args.model_dir) if args.model_dir else None
        return arch, model, model.config, king

    spec = (
        MiniatureSpec.real_vocab(num_hidden_layers=args.layers)
        if args.real_vocab
        else MiniatureSpec(num_hidden_layers=args.layers)
    )
    model, config = build_miniature(arch, reference, spec, seed=args.seed)
    return arch, model, config, None


def cmd_train(args) -> int:
    config = _load_config(args)

    # Resolve the resume source before building, so the weights are restored
    # alongside the optimizer state rather than after it.
    resume_from = None
    resume_state = None
    if args.resume:
        output_dir = Path(args.output) if args.output else (
            Path(config.output_dir) / config.run_name
        )
        found = latest_checkpoint(output_dir)
        if not found:
            print(f"nothing to resume in {output_dir}", file=sys.stderr)
            return EXIT_USAGE
        resume_from, resume_state = found

    arch, model, model_config, king_dir = _build_model(args, resume_from)

    seq_len = args.seq_len or (2048 if args.real_vocab or args.model_dir else 64)
    batches, context = resolve_batches(
        config,
        vocab_size=int(model_config.vocab_size),
        seq_len=seq_len,
        synthetic=args.synthetic,
        state_root=STATE_ROOT,
        blend=not args.single_corpus,
    )
    config.manifest_sha256 = context.get("manifest_sha256")

    def log(metrics):
        b = metrics.balance
        print(
            f"  step {metrics.step:>5}  loss {metrics.loss:8.4f}  "
            f"lr {metrics.lr:.2e}  |g| {metrics.grad_norm:7.3f}  "
            f"experts {b.get('experts_touched')}/{b.get('n_experts', '?')}  "
            f"imbalance {b.get('imbalance')}"
        )

    trainer = Trainer(
        model=model,
        arch=arch,
        config=config,
        batches=batches,
        king_dir=Path(args.king) if args.king else king_dir,
        output_dir=Path(args.output) if args.output else None,
        on_log=log,
    )

    if resume_state is not None:
        step = trainer.resume(resume_state)
        print(f"resumed weights from {resume_from} and state from {resume_state} at step {step}")

    print(f"\nrun '{config.run_name}'  stage={config.stage}  source={context['source']}")
    print(
        f"  trainable {trainer.freeze.trainable_params:,} / "
        f"{trainer.freeze.total_params:,} parameters "
        f"({trainer.freeze.trainable_fraction * 100:.1f}%)"
    )
    if context.get("corpora"):
        print(f"  corpora   {', '.join(context['corpora'])}")
        if context.get("missing_corpora"):
            print(
                f"  ! missing {', '.join(context['missing_corpora'])} — the validator "
                "still scores them"
            )
    if context.get("holdouts"):
        print(
            f"  holdouts {', '.join(context['holdouts'])} "
            f"({context.get('excluded_sequences', 0):,} sequences excluded)"
        )
    print()

    try:
        result = trainer.train()
    except DivergenceError as exc:
        print(f"\ndiverged: {exc}", file=sys.stderr)
        return EXIT_FAILED

    report = trainer.write_report(result)
    print(
        f"\n  {result.steps} steps · {result.sequences_seen:,} sequences · "
        f"{result.wall_time_s:.1f}s"
    )
    if result.loss_delta is not None:
        print(
            f"  loss {result.first_loss:.4f} -> {result.last_loss:.4f} "
            f"(Δ {result.loss_delta:+.4f})"
        )
    balance = result.balance_summary
    print(
        f"  routing imbalance {balance.get('first_imbalance')} -> "
        f"{balance.get('last_imbalance')} (1.0 is uniform), "
        f"coverage {balance.get('last_coverage')}"
    )
    for checkpoint in result.checkpoints[-2:]:
        if checkpoint.shape_mismatch:
            flag = "  (miniature: locked files not restored — " + checkpoint.shape_mismatch[0] + ")"
        elif not checkpoint.contract_complete:
            flag = "  ! locked files NOT restored"
        else:
            flag = ""
        print(f"  checkpoint {checkpoint.model_dir}{flag}")
    print(f"  report {report}")

    if args.validate and result.checkpoints:
        return _validate(result.checkpoints[-1].model_dir, args)
    return EXIT_OK


def _validate(model_dir: Path, args) -> int:
    """Run the packaging validator on the checkpoint just written."""
    try:
        from validate_checkpoint import Contract, KingReference, Options, validate
    except ImportError:  # pragma: no cover
        print("validate_checkpoint is not importable", file=sys.stderr)
        return EXIT_OK

    contract = Contract.load(None)
    king = None
    if args.king_digest:
        king = KingReference.from_digest(
            args.king_digest, cache_dir=STATE_ROOT / "king-reference"
        )
    report = validate(
        model_dir, contract=contract, king=king, options=Options(model_name=args.name)
    )
    print(f"\n  packaging: {report.verdict}")
    for check in report.failures:
        print(f"    FAIL {check.name}: {check.detail}")
    return EXIT_FAILED if report.would_reject else EXIT_OK


def cmd_config(args) -> int:
    config = TrainingConfig()
    if args.stage:
        config.stage = args.stage
    config.__post_init__()
    path = Path(args.output or "configs/pilot.json")
    config.save(path)
    print(f"wrote {path}")
    print(json.dumps(config.to_dict(), indent=2))
    return EXIT_OK


def cmd_inspect(args) -> int:
    output_dir = Path(args.run)
    found = latest_checkpoint(output_dir)
    if not found:
        print(f"no checkpoints in {output_dir}", file=sys.stderr)
        return EXIT_USAGE
    model_dir, state_dir = found
    print(f"latest checkpoint  {model_dir}")
    print(f"resume state       {state_dir}")
    files = sorted(p.name for p in model_dir.iterdir() if p.is_file())
    print(f"files ({len(files)}): {', '.join(files)}")

    if args.chain:
        from validate_checkpoint import Contract

        contract = Contract.load(args.chain)
        wrong = verify_locked_files(model_dir, contract.contract_files)
        print(
            "locked files: all match chain.toml"
            if not wrong
            else f"locked files DIFFER: {', '.join(wrong)}"
        )
    return EXIT_OK


def cmd_prune(args) -> int:
    removed = prune_checkpoints(Path(args.run), keep=args.keep)
    print(f"removed {len(removed)} checkpoint(s); kept the newest {args.keep}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train-mimo",
        description="Continued pretraining of a MiMo checkpoint for SN3.",
    )
    parser.add_argument("--version", action="version", version=f"train-mimo {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="run a training job")
    p.add_argument("--config", help="run configuration JSON")
    p.add_argument("--run-name")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--stage", choices=sorted(STAGES))
    p.add_argument("--save-every", type=int)
    p.add_argument("--log-every", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--grad-accum", type=int)
    p.add_argument("--shard", action="append", help="FineWeb shard name (repeatable)")
    p.add_argument("--holdout", action="append", help="holdout to exclude (repeatable)")
    p.add_argument("--synthetic", action="store_true", help="random tokens, no shards")
    p.add_argument(
        "--single-corpus", action="store_true",
        help="train on finewebedu alone (22% of the score); comparison runs only",
    )
    p.add_argument("--no-balance", action="store_true", help="disable the bias rule")
    p.add_argument("--model-dir", help="train a real checkpoint instead of a miniature")
    p.add_argument("--king", help="king directory, for restoring the locked files")
    p.add_argument("--king-digest", help="king digest, for packaging validation")
    p.add_argument("--arch", help="architecture directory")
    p.add_argument("--real-vocab", action="store_true", help="miniature with the real vocabulary")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seq-len", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", help="run output directory")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--validate", action="store_true", help="validate the final checkpoint")
    p.add_argument("--name", help="intended submission name, for validation")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("config", help="write a starter configuration")
    p.add_argument("--output")
    p.add_argument("--stage", choices=sorted(STAGES))
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("inspect", help="show a run's latest checkpoint")
    p.add_argument("run")
    p.add_argument("--chain", help="chain.toml, to verify the locked files")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("prune", help="delete old checkpoints")
    p.add_argument("run")
    p.add_argument("--keep", type=int, default=3)
    p.set_defaults(func=cmd_prune)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except TrainingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:
        print("\ninterrupted; the last checkpoint is on disk", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
