"""Proves the expert-count all-reduce is load-bearing.

    torchrun --standalone --nproc_per_node=3 scripts/dist_divergence.py

Runs the balancer twice on per-rank counts: once with the all-reduce and once
without. With it, every rank ends holding the same bias. Without it, they
diverge -- silently, in a real run.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_mimo.balance import LoadBalancer
from train_mimo.distributed import all_reduce_expert_counts, process_group


class Gate:
    def __init__(self, n):
        self.e_score_correction_bias = torch.nn.Parameter(torch.zeros(n))


def spread(gate) -> list[float]:
    local = gate.e_score_correction_bias.detach().clone()
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return [float((g - gathered[0]).abs().max()) for g in gathered]


def main() -> int:
    N = 8
    with process_group() as ctx:
        # Each rank routes different tokens: rank r favours expert r.
        counts = Counter({ctx.rank: 100, (ctx.rank + 1) % N: 10})

        without = LoadBalancer([Gate(N)], update_rate=0.1, min_tokens=0)
        without.update(type("R", (), {"counts": counts, "n_experts": N, "top_k": 2})())
        drift_without = max(spread(without.gates[0]))

        synced = all_reduce_expert_counts(counts, N, ctx)
        with_ = LoadBalancer([Gate(N)], update_rate=0.1, min_tokens=0)
        with_.update(type("R", (), {"counts": synced, "n_experts": N, "top_k": 2})())
        drift_with = max(spread(with_.gates[0]))

        if ctx.is_main:
            print(f"world_size            {ctx.world_size}")
            print(f"local counts (rank 0) {dict(counts)}")
            print(f"all-reduced counts    {dict(synced)}")
            print()
            print(f"max bias divergence WITHOUT all-reduce  {drift_without:.6f}")
            print(f"max bias divergence WITH    all-reduce  {drift_with:.6f}")
            print()
            assert drift_without > 0, "expected divergence without the all-reduce"
            assert drift_with == 0.0, "ranks must agree once counts are reduced"
            print("OK — the all-reduce is what keeps the ranks consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
