"""A training-capable replacement for MiMo's ``noaux_tc`` routing.

The shipped gate refuses to run under ``model.train()``::

    if self.topk_method == "noaux_tc":
        if self.training:
            raise ValueError("MiMoV2 noaux_tc routing is only implemented for inference.")

``modeling_mimo_v2.py`` is hash-pinned in chain.toml, so the file on disk is
never touched. The class's ``forward`` is replaced in memory for the duration of
a context manager and restored afterwards.

**What the replacement changes:** it removes the guard, and it computes expert
*selection* under ``no_grad``. Nothing else. Selection produces integer indices,
which carry no gradient in either version, so detaching that branch is provably
output-identical while avoiding a dangling autograd graph.

**What stays differentiable:** ``topk_weight = scores.gather(1, topk_idx)``.
``gather`` is differentiable with respect to ``scores``, so gradients reach the
router's ``weight`` through the selected experts' combination weights. This is
why the guard is the only real obstacle -- the rest of the original forward was
already differentiable.

**What is *not* differentiable, by design:** ``e_score_correction_bias`` enters
only ``scores_for_choice``, which is consumed by ``topk`` and discarded. It
receives no gradient in the original implementation and none here. That is the
"auxiliary-loss-free" scheme: the bias is meant to be adjusted by a load-balancing
rule you run yourself, not learned by backprop. Training without such a rule
risks expert collapse, and that -- not a mathematical obstacle -- is the most
likely reason the guard exists.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from .errors import AdapterError

# Set while a patch is active, so the replacement can report routing decisions.
_RECORDER: "RoutingRecorder | None" = None


@dataclass
class RoutingRecorder:
    """Collects which experts the router selects.

    With top-8 of 256 experts, most experts receive no gradient from any given
    token. Knowing the real coverage per batch tells you how large a batch must
    be before expert updates stop being noise.
    """

    counts: Counter = field(default_factory=Counter)
    tokens: int = 0
    calls: int = 0
    n_experts: int = 0
    top_k: int = 0

    def record(self, topk_idx, n_experts: int, top_k: int) -> None:
        self.calls += 1
        self.n_experts = n_experts
        self.top_k = top_k
        flat = topk_idx.detach().reshape(-1).tolist()
        self.counts.update(flat)
        self.tokens += topk_idx.shape[0]

    @property
    def experts_touched(self) -> int:
        return len(self.counts)

    @property
    def coverage(self) -> float:
        """Fraction of experts that received at least one token."""
        return self.experts_touched / self.n_experts if self.n_experts else 0.0

    @property
    def busiest(self) -> list[tuple[int, int]]:
        return self.counts.most_common(5)

    @property
    def imbalance(self) -> float | None:
        """Max/mean selection count. 1.0 is perfectly balanced."""
        if not self.counts or not self.n_experts:
            return None
        mean = sum(self.counts.values()) / self.n_experts
        return (max(self.counts.values()) / mean) if mean else None

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "token_slots": self.tokens,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "experts_touched": self.experts_touched,
            "coverage": round(self.coverage, 4),
            "imbalance": round(self.imbalance, 3) if self.imbalance else None,
            "busiest": self.busiest,
        }


def trainable_gate_forward(self, hidden_states):
    """``MiMoV2MoEGate.forward``, minus the training guard.

    Mirrors the original operation for operation so that eval-mode output is
    bit-comparable; see the module docstring for the two deliberate differences.
    """
    import torch
    import torch.nn.functional as F

    bsz, seq_len, h = hidden_states.shape
    hidden_states = hidden_states.view(-1, h)
    logits = F.linear(
        hidden_states.type(torch.float32), self.weight.type(torch.float32), None
    )
    if self.scoring_func == "sigmoid":
        scores = logits.sigmoid()
    else:
        raise NotImplementedError(
            f"Unsupported scoring function for MoE gating: {self.scoring_func}"
        )

    if self.topk_method != "noaux_tc":
        raise NotImplementedError(
            f"Unsupported TopK function for MoE gating: {self.topk_method}"
        )

    # Selection yields integer indices and carries no gradient in the original
    # either; computing it detached is output-identical and avoids building an
    # autograd graph that would only be discarded.
    with torch.no_grad():
        detached = scores.detach()
        scores_for_choice = detached.view(bsz * seq_len, -1) + (
            self.e_score_correction_bias.unsqueeze(0)
        )
        group_scores = (
            scores_for_choice.view(bsz * seq_len, self.n_group, -1)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(bsz * seq_len, -1)
        )
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)

    # The differentiable path: gather keeps the graph back to self.weight.
    topk_weight = scores.gather(1, topk_idx)

    if self.top_k > 1 and self.norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator
    topk_weight = topk_weight * self.routed_scaling_factor

    if _RECORDER is not None:
        _RECORDER.record(topk_idx, self.n_routed_experts, self.top_k)

    return topk_idx, topk_weight


@contextmanager
def trainable_routing(
    arch, *, recorder: RoutingRecorder | None = None
) -> Iterator[RoutingRecorder | None]:
    """Make ``noaux_tc`` routing trainable for the duration of the block.

    Restores the original ``forward`` on exit, including on exception, so a
    failed run cannot leave a patched class behind.
    """
    global _RECORDER
    gate_cls = arch.gate_cls
    original = gate_cls.forward
    if getattr(original, "_mimo_adapter_patched", False):
        raise AdapterError("routing is already patched; nested patches are not allowed")

    trainable_gate_forward._mimo_adapter_patched = True  # type: ignore[attr-defined]
    gate_cls.forward = trainable_gate_forward
    previous_recorder, _RECORDER = _RECORDER, recorder
    try:
        yield recorder
    finally:
        gate_cls.forward = original
        _RECORDER = previous_recorder


@contextmanager
def recording_routing(arch) -> Iterator[RoutingRecorder]:
    """Patch and collect routing statistics in one step."""
    recorder = RoutingRecorder()
    with trainable_routing(arch, recorder=recorder):
        yield recorder


def is_patched(arch) -> bool:
    return bool(getattr(arch.gate_cls.forward, "_mimo_adapter_patched", False))


def gates(model) -> list[Any]:
    """Every routing gate in a model, in module order."""
    return [
        module
        for module in model.modules()
        if type(module).__name__ == "MiMoV2MoEGate"
    ]


def set_gates_eval(model) -> int:
    """Force gate modules into eval while leaving the rest of the model training.

    This is the cheaper alternative to patching: ``self.training`` becomes False
    on the gates, so the guard does not fire. It is offered for comparison --
    :func:`trainable_routing` is preferred because it states its intent instead
    of relying on a flag the gate also uses for other purposes.
    """
    found = gates(model)
    for gate in found:
        gate.eval()
    return len(found)
