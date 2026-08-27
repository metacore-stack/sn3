"""Save under N ranks, resume under N ranks, and prove nothing was lost.

    python -m torch.distributed.run --standalone --nproc_per_node=2 \
        scripts/dist_resume.py /tmp/dist-resume

Only rank zero writes a checkpoint, but every rank must reach the save -- under
FSDP the state-dict gather is collective. The routing bias is the piece most
easily lost across a restart: it lives outside the model directory, receives no
gradient, and nothing else would notice its absence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mimo_adapter import MiniatureSpec, build_miniature, load_arch, read_reference_config
from train_mimo import TrainingConfig, Trainer
from train_mimo.checkpoint import latest_checkpoint
from train_mimo.config import BalanceConfig, DataConfig, DistributedConfig, OptimConfig
from train_mimo.distributed import assert_bias_synchronised, barrier, process_group
from train_mimo.sources import synthetic_batches


def make(ctx, out: Path, steps: int, save_every: int):
    arch = load_arch()
    reference = read_reference_config(arch.directory)
    model, mcfg = build_miniature(arch, reference, MiniatureSpec(), seed=0)
    config = TrainingConfig(
        run_name="dist-resume",
        max_steps=steps,
        save_every=save_every,
        log_every=0,
        stage="matrices",
        data=DataConfig(batch_size=1, grad_accum=max(1, 4 // ctx.world_size)),
        optim=OptimConfig(learning_rate=0.0, warmup_steps=0, schedule="constant"),
        balance=BalanceConfig(update_rate=1e-3, min_tokens=1),
        distributed=DistributedConfig(
            strategy="ddp" if ctx.is_distributed else "none", bias_check_every=1
        ),
    )
    batches = synthetic_batches(
        vocab_size=int(mcfg.vocab_size), seq_len=32, batch_size=1, steps=400, seed=0
    )
    return Trainer(
        model=model, arch=arch, config=config, batches=batches,
        output_dir=out, context=ctx,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.out)

    with process_group() as ctx:
        # Phase one: train and check-point.
        first = make(ctx, out, args.steps, save_every=args.steps)
        first.train()
        before = first.balancer.bias_state()
        assert_bias_synchronised(first.balancer.gates, ctx)
        barrier(ctx)

        found = latest_checkpoint(out)
        if found is None:
            raise SystemExit(f"no checkpoint under {out}")
        _, state_dir = found

        # Phase two: a fresh process-equivalent trainer picks the state back up.
        second = make(ctx, out, args.steps, save_every=0)
        after_fresh = second.balancer.bias_state()
        step = second.resume(state_dir)
        after = second.balancer.bias_state()
        assert_bias_synchronised(second.balancer.gates, ctx)

        checks = {
            "world_size": ctx.world_size,
            "resumed_step": step,
            "fresh_bias_differs_from_saved": after_fresh != before,
            "resumed_bias_matches_saved": after == before,
            "optimizer_state_restored": bool(second.optimizer.state_dict()["state"]),
        }
        if ctx.is_main:
            print(json.dumps(checks, indent=2))
            if not checks["resumed_bias_matches_saved"]:
                raise SystemExit("routing bias was NOT restored")
            if not checks["fresh_bias_differs_from_saved"]:
                raise SystemExit("test is vacuous: a fresh trainer already matched")
            print("OK — bias, optimizer and step survived the restart on every rank")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
