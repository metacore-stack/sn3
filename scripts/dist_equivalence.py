"""Prove that N ranks reconstruct the single-process token distribution exactly.

    python -m torch.distributed.run --standalone --nproc_per_node=1 \
        scripts/dist_equivalence.py out-w1.json
    python -m torch.distributed.run --standalone --nproc_per_node=2 \
        scripts/dist_equivalence.py out-w2.json

The two JSON files must be identical.

The comparison is only meaningful if every configuration consumes the *same*
batches. ``--micro-per-step`` fixes the global number of micro-batches per
optimizer step; each rank then takes ``micro_per_step / world_size`` of them, so
step *k* always covers global batches ``[k*M, (k+1)*M)`` no matter how many ranks
are running.

The learning rate is zero on purpose. With the weights pinned, the whole
trajectory is a deterministic function of the batches and the routing bias, so
any difference between the runs is a real sharding or all-reduce bug rather than
float-reduction noise. Gradient synchronisation is exercised separately by
``dist_smoke.py``, where bitwise equality across reduction orders is not a
property anyone can promise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mimo_adapter import MiniatureSpec, build_miniature, load_arch, read_reference_config
from train_mimo import TrainingConfig, Trainer
from train_mimo.config import BalanceConfig, DataConfig, DistributedConfig, OptimConfig
from train_mimo.distributed import assert_bias_synchronised, process_group
from train_mimo.sources import synthetic_batches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", help="where rank zero writes the result")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--micro-per-step", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()

    with process_group() as ctx:
        if args.micro_per_step % ctx.world_size:
            raise SystemExit(
                f"--micro-per-step {args.micro_per_step} is not divisible by "
                f"world size {ctx.world_size}; the runs would see different data"
            )

        arch = load_arch()
        reference = read_reference_config(arch.directory)
        model, mcfg = build_miniature(arch, reference, MiniatureSpec(), seed=0)

        config = TrainingConfig(
            run_name="dist-equivalence",
            max_steps=args.steps,
            save_every=0,
            log_every=0,
            stage="matrices",
            data=DataConfig(
                batch_size=1, grad_accum=args.micro_per_step // ctx.world_size
            ),
            # Weights pinned: the run becomes deterministic across world sizes.
            optim=OptimConfig(learning_rate=0.0, warmup_steps=0, schedule="constant"),
            balance=BalanceConfig(update_rate=1e-3, min_tokens=1),
            distributed=DistributedConfig(
                strategy="ddp" if ctx.is_distributed else "none", bias_check_every=1
            ),
        )
        batches = synthetic_batches(
            vocab_size=int(mcfg.vocab_size),
            seq_len=args.seq_len,
            batch_size=1,
            steps=args.steps * args.micro_per_step + 8,
            seed=0,
        )
        trainer = Trainer(
            model=model,
            arch=arch,
            config=config,
            batches=batches,
            output_dir=Path(args.out).parent / "runs",
            context=ctx,
        )
        result = trainer.train()
        assert_bias_synchronised(trainer.balancer.gates, ctx)

        payload = {
            "steps": result.steps,
            "micro_per_step": args.micro_per_step,
            # Not world_size -- that is what differs. Everything below must not.
            "bias": trainer.balancer.bias_state(),
            "balance_history": [s.to_dict() for s in trainer.balancer.history],
        }
        if ctx.is_main:
            Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
            print(f"wrote {args.out} (world_size={ctx.world_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
