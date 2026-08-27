"""Load balancing for the routing bias.

``e_score_correction_bias`` receives no gradient. It enters only
``scores_for_choice``, which ``topk`` consumes and discards, so backprop never
touches it -- in the shipped code or in the patched version. That is the
auxiliary-loss-free design: the bias is meant to be moved by an explicit rule.

Without one, routing drifts toward whichever experts happened to start slightly
favoured, and the drift compounds: a favoured expert receives more tokens, trains
faster, scores higher, and is favoured more. That collapse is the most plausible
reason the shipped gate refuses training mode at all.

The rule here is the standard one: for each expert, compare its share of routed
tokens against a uniform target, and step its bias against the difference.
Because only the *sign* of the error is used, the step size is bounded and the
rule cannot itself destabilise training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class BalanceStats:
    """What one balancing update saw and did."""

    step: int = 0
    tokens: int = 0
    n_experts: int = 0
    experts_touched: int = 0
    max_share: float = 0.0
    min_share: float = 0.0
    imbalance: float = 0.0
    bias_delta: float = 0.0
    applied: bool = False

    @property
    def coverage(self) -> float:
        return self.experts_touched / self.n_experts if self.n_experts else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tokens": self.tokens,
            "n_experts": self.n_experts,
            "experts_touched": self.experts_touched,
            "coverage": round(self.coverage, 4),
            "max_share": round(self.max_share, 6),
            "min_share": round(self.min_share, 6),
            "imbalance": round(self.imbalance, 4),
            "bias_delta": round(self.bias_delta, 8),
            "applied": self.applied,
        }


class LoadBalancer:
    """Applies the aux-loss-free bias update to every gate in a model."""

    def __init__(
        self,
        gates: Sequence[Any],
        *,
        update_rate: float = 1e-3,
        min_tokens: int = 64,
        enabled: bool = True,
    ):
        self.gates = list(gates)
        self.update_rate = float(update_rate)
        self.min_tokens = int(min_tokens)
        self.enabled = bool(enabled)
        self.history: list[BalanceStats] = []

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.gates) and self.update_rate > 0

    def update(self, recorder, *, step: int = 0) -> BalanceStats:
        """Nudge each expert's bias against its share of the routed tokens.

        ``recorder`` is a :class:`mimo_adapter.RoutingRecorder` holding the
        selection counts from the step just completed.
        """
        import torch

        counts = getattr(recorder, "counts", None)
        n_experts = int(getattr(recorder, "n_experts", 0) or 0)
        total = sum(counts.values()) if counts else 0

        stats = BalanceStats(step=step, tokens=total, n_experts=n_experts)
        if not counts or not n_experts:
            self.history.append(stats)
            return stats

        shares = [counts.get(i, 0) / total for i in range(n_experts)] if total else []
        if shares:
            stats.experts_touched = sum(1 for s in shares if s > 0)
            stats.max_share = max(shares)
            stats.min_share = min(shares)
            stats.imbalance = stats.max_share * n_experts  # 1.0 == uniform

        if not self.active or total < self.min_tokens:
            self.history.append(stats)
            return stats

        target = 1.0 / n_experts
        error = torch.tensor(
            [target - share for share in shares], dtype=torch.float32
        )
        # Sign only: the magnitude of the imbalance sets the direction, not the
        # step, so a single pathological batch cannot move the bias far.
        delta = self.update_rate * torch.sign(error)

        with torch.no_grad():
            for gate in self.gates:
                bias = getattr(gate, "e_score_correction_bias", None)
                if bias is None or bias.numel() != n_experts:
                    continue
                bias.add_(delta.to(dtype=bias.dtype, device=bias.device))

        stats.bias_delta = float(delta.abs().mean().item())
        stats.applied = True
        self.history.append(stats)
        return stats

    # -- state ------------------------------------------------------------

    def bias_state(self) -> list[list[float]]:
        """Current bias vectors, for checkpointing."""
        return [
            gate.e_score_correction_bias.detach().cpu().tolist()
            for gate in self.gates
            if getattr(gate, "e_score_correction_bias", None) is not None
        ]

    def load_bias_state(self, state: Sequence[Sequence[float]]) -> int:
        """Restore bias vectors saved by :meth:`bias_state`."""
        import torch

        restored = 0
        biased = [
            gate
            for gate in self.gates
            if getattr(gate, "e_score_correction_bias", None) is not None
        ]
        if len(state) != len(biased):
            raise ValueError(
                f"bias state has {len(state)} vectors but the model has {len(biased)} gates"
            )
        with torch.no_grad():
            for gate, values in zip(biased, state):
                bias = gate.e_score_correction_bias
                incoming = torch.tensor(values, dtype=bias.dtype, device=bias.device)
                if incoming.shape != bias.shape:
                    raise ValueError(
                        f"bias shape {tuple(incoming.shape)} != {tuple(bias.shape)}"
                    )
                bias.copy_(incoming)
                restored += 1
        return restored

    def summary(self) -> dict[str, Any]:
        if not self.history:
            return {"updates": 0}
        applied = [s for s in self.history if s.applied]
        imbalances = [s.imbalance for s in self.history if s.imbalance]
        return {
            "updates": len(self.history),
            "applied": len(applied),
            "first_imbalance": round(imbalances[0], 4) if imbalances else None,
            "last_imbalance": round(imbalances[-1], 4) if imbalances else None,
            "worst_imbalance": round(max(imbalances), 4) if imbalances else None,
            "last_coverage": round(self.history[-1].coverage, 4),
        }
