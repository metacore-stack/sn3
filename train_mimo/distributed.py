"""Multi-process training.

The 110B king does not fit in one device's memory, so training it means several
ranks. Almost all of that is testable without a GPU: ``torchrun`` with the gloo
backend on the miniature exercises process-group setup, data sharding, gradient
synchronisation, FSDP parameter sharding, rank-zero saving and resume.

One thing here has no single-process equivalent and is the reason this module
exists as more than a wrapper:

**Expert selection counts must be all-reduced before the routing bias moves.**

``e_score_correction_bias`` receives no gradient, so DDP/FSDP never synchronise
it. It is updated by an explicit rule driven by how many tokens each expert saw.
Under data parallelism each rank sees different tokens, so each rank computes a
different load and would step the bias differently. Within a few hundred steps
the ranks disagree about routing, and the checkpoint saved from rank zero
reflects only rank zero's view.

That failure is invisible in a single-process run and produces no error in a
distributed one -- just a quietly wrong model. :func:`all_reduce_expert_counts`
is what prevents it, and the test suite asserts bias equality across ranks.
"""

from __future__ import annotations

import os
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from .errors import TrainingError

# Wrapping strategies. "none" is the single-process path; "ddp" replicates the
# model per rank; "fsdp" shards parameters, gradients and optimizer state, which
# is what a 110B checkpoint actually requires.
STRATEGIES = ("none", "ddp", "fsdp")


@dataclass(frozen=True)
class DistributedContext:
    """Where this process sits in the group."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = "gloo"
    initialized: bool = False

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """Rank zero. Only this rank writes checkpoints, reports and logs."""
        return self.rank == 0

    def summary(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "backend": self.backend,
            "distributed": self.is_distributed,
        }


def from_environment() -> DistributedContext:
    """Read the group layout torchrun puts in the environment."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = os.environ.get("TEUTONIC_DIST_BACKEND") or default_backend()
    return DistributedContext(
        rank=rank, world_size=world_size, local_rank=local_rank, backend=backend
    )


def default_backend() -> str:
    """nccl on CUDA, gloo otherwise.

    gloo is not a lesser fallback here: it is how the whole synchronisation
    story gets tested on a machine with no GPU.
    """
    try:
        import torch

        return "nccl" if torch.cuda.is_available() else "gloo"
    except ImportError:  # pragma: no cover
        return "gloo"


def initialize(context: DistributedContext | None = None) -> DistributedContext:
    """Join the process group, if there is one."""
    import torch.distributed as dist

    context = context or from_environment()
    if not context.is_distributed:
        return context
    if dist.is_initialized():
        return DistributedContext(
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            local_rank=context.local_rank,
            backend=context.backend,
            initialized=True,
        )
    dist.init_process_group(
        backend=context.backend,
        rank=context.rank,
        world_size=context.world_size,
    )
    if context.backend == "nccl":  # pragma: no cover - GPU only
        import torch

        torch.cuda.set_device(context.local_rank)
    return DistributedContext(
        rank=context.rank,
        world_size=context.world_size,
        local_rank=context.local_rank,
        backend=context.backend,
        initialized=True,
    )


def shutdown() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@contextmanager
def process_group(context: DistributedContext | None = None) -> Iterator[DistributedContext]:
    """Join for the duration of the block, and always leave cleanly."""
    ctx = initialize(context)
    try:
        yield ctx
    finally:
        if ctx.is_distributed:
            shutdown()


# -- the synchronisation that matters ---------------------------------------


def all_reduce_expert_counts(counts: Counter, n_experts: int, context: DistributedContext) -> Counter:
    """Sum expert selection counts across every rank.

    Called before :meth:`LoadBalancer.update` so every rank steps the routing
    bias from the same global load. Without it the ranks diverge silently and
    the saved checkpoint carries only rank zero's routing.

    Returns a new Counter; the input is not modified.
    """
    if not context.is_distributed:
        return Counter(counts)

    import torch
    import torch.distributed as dist

    vector = torch.zeros(n_experts, dtype=torch.float64)
    for expert, value in counts.items():
        if 0 <= int(expert) < n_experts:
            vector[int(expert)] = float(value)
    dist.all_reduce(vector, op=dist.ReduceOp.SUM)
    return Counter(
        {i: int(v) for i, v in enumerate(vector.tolist()) if v}
    )


def all_reduce_mean(value: float, context: DistributedContext) -> float:
    """Average a scalar across ranks, for logging that reflects the whole batch."""
    if not context.is_distributed:
        return value

    import torch
    import torch.distributed as dist

    tensor = torch.tensor([float(value)], dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / context.world_size)


def barrier(context: DistributedContext) -> None:
    if not context.is_distributed:
        return
    import torch.distributed as dist

    dist.barrier()


def assert_bias_synchronised(gates: Sequence[Any], context: DistributedContext) -> None:
    """Verify every rank holds identical routing biases.

    Cheap enough to run periodically. Catches a missing all-reduce immediately
    rather than after a training run has already been paid for.
    """
    if not context.is_distributed:
        return

    import torch
    import torch.distributed as dist

    for index, gate in enumerate(gates):
        bias = getattr(gate, "e_score_correction_bias", None)
        if bias is None:
            continue
        local = bias.detach().to(torch.float64).cpu()
        gathered = [torch.zeros_like(local) for _ in range(context.world_size)]
        dist.all_gather(gathered, local)
        for other_rank, other in enumerate(gathered):
            if not torch.equal(local, other):
                raise TrainingError(
                    f"routing bias diverged: gate {index} differs between rank "
                    f"{context.rank} and rank {other_rank}. Expert counts were "
                    "not all-reduced before the balancer update."
                )


# -- data sharding ----------------------------------------------------------


def shard_stream(stream: Iterator, context: DistributedContext) -> Iterator:
    """Give each rank a disjoint slice of the batch stream.

    Rank *r* of *w* takes every *w*-th batch starting at *r*. Every rank sees
    the same number of batches, which keeps the collectives aligned; a rank that
    ran dry early would hang the others on the next all-reduce.
    """
    if not context.is_distributed:
        yield from stream
        return
    for index, item in enumerate(stream):
        if index % context.world_size == context.rank:
            yield item


# -- model wrapping ---------------------------------------------------------


def wrap_model(model, strategy: str, context: DistributedContext, *, layer_cls_name: str = "MiMoV2DecoderLayer"):
    """Apply the requested parallelism.

    Freezing must already have been applied: both DDP and FSDP inspect
    ``requires_grad`` when they wrap, and FSDP renames parameters, after which
    the stage's substring patterns no longer match.
    """
    if strategy not in STRATEGIES:
        raise TrainingError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    if strategy == "none" or not context.is_distributed:
        return model

    import torch
    import torch.distributed as dist  # noqa: F401  (ensures availability)

    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        device_ids = [context.local_rank] if context.backend == "nccl" else None
        return DistributedDataParallel(
            model,
            device_ids=device_ids,
            # The router bias never receives a gradient, and several stages
            # freeze large parameter groups; without this DDP raises on the
            # first backward.
            find_unused_parameters=True,
        )

    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    layer_cls = _find_layer_class(model, layer_cls_name)
    policy = None
    if layer_cls is not None:
        import functools

        policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}
        )
    return FullyShardedDataParallel(
        model,
        auto_wrap_policy=policy,
        use_orig_params=True,  # keeps named_parameters() stable for reporting
        device_id=context.local_rank if context.backend == "nccl" else None,
    )


def _find_layer_class(model, name: str):
    for module in model.modules():
        if type(module).__name__ == name:
            return type(module)
    return None


def unwrap(model):
    """The underlying module, whatever it was wrapped in."""
    return getattr(model, "module", model)


def gather_full_state_dict(model, context: DistributedContext):
    """Full, unsharded state dict on rank zero.

    FSDP holds a shard per rank; saving that directly would produce a checkpoint
    no one can load. Every rank must enter this context together.
    """
    inner = unwrap(model)
    if not context.is_distributed:
        return inner.state_dict()

    try:
        from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel
        from torch.distributed.fsdp import StateDictType
    except ImportError:  # pragma: no cover
        return inner.state_dict()

    if not isinstance(model, FullyShardedDataParallel):
        return inner.state_dict()

    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FullyShardedDataParallel.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, config
    ):
        return model.state_dict()
