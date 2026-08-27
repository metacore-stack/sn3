"""Distributed smoke test. Run with:

    torchrun --standalone --nproc_per_node=4 scripts/dist_smoke.py

Asserts the properties that only appear with more than one rank.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mimo_adapter import MiniatureSpec, build_miniature, load_arch, read_reference_config
from train_mimo import TrainingConfig, Trainer
from train_mimo.config import DataConfig, DistributedConfig
from train_mimo.distributed import assert_bias_synchronised, process_group
from train_mimo.sources import synthetic_batches


def main() -> int:
    with process_group() as ctx:
        arch = load_arch()
        reference = read_reference_config(arch.directory)
        model, mcfg = build_miniature(arch, reference, MiniatureSpec(), seed=0)

        config = TrainingConfig(
            run_name="dist-smoke",
            max_steps=8,
            save_every=0,
            log_every=0,
            stage="matrices",
            data=DataConfig(batch_size=1, grad_accum=1),
            distributed=DistributedConfig(strategy="ddp", bias_check_every=2),
        )
        batches = synthetic_batches(
            vocab_size=int(mcfg.vocab_size), seq_len=32,
            batch_size=1, steps=200, seed=0,
        )
        trainer = Trainer(
            model=model, arch=arch, config=config, batches=batches,
            output_dir=Path("/tmp/dist-smoke"), context=ctx,
        )
        result = trainer.train()

        assert_bias_synchronised(trainer.balancer.gates, ctx)
        biases = trainer.balancer.bias_state()

        if ctx.is_main:
            print(f"world_size      {ctx.world_size}")
            print(f"steps           {result.steps}")
            print(f"trainable       {trainer.freeze.trainable_params:,} / "
                  f"{trainer.freeze.total_params:,}")
            print(f"loss            {result.first_loss:.4f} -> {result.last_loss:.4f}")
            print(f"bias vectors    {len(biases)} gates, synchronised across all ranks")
            print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
