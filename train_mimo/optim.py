"""Parameter selection and the learning-rate schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .config import OptimConfig
from .errors import ConfigError


@dataclass(frozen=True)
class FreezeResult:
    trainable: tuple[str, ...]
    frozen: tuple[str, ...]
    trainable_params: int
    frozen_params: int

    @property
    def total_params(self) -> int:
        return self.trainable_params + self.frozen_params

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_params / self.total_params if self.total_params else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "trainable_tensors": len(self.trainable),
            "frozen_tensors": len(self.frozen),
            "trainable_params": self.trainable_params,
            "frozen_params": self.frozen_params,
            "trainable_fraction": round(self.trainable_fraction, 6),
        }


def apply_freeze(model, patterns: Sequence[str]) -> FreezeResult:
    """Enable gradients only for parameters matching ``patterns``.

    An empty pattern list trains everything. Freezing is what makes staged
    unfreezing cheap: optimizer state is allocated only for what is trainable,
    which on a 110B model is the difference between fitting and not fitting.
    """
    trainable: list[str] = []
    frozen: list[str] = []
    trainable_params = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        wanted = not patterns or any(p in name for p in patterns)
        param.requires_grad_(wanted)
        if wanted:
            trainable.append(name)
            trainable_params += param.numel()
        else:
            frozen.append(name)
            frozen_params += param.numel()

    if not trainable:
        raise ConfigError(
            f"stage selectors {list(patterns)} matched no parameters; "
            "the run would have nothing to optimise"
        )

    return FreezeResult(
        trainable=tuple(trainable),
        frozen=tuple(frozen),
        trainable_params=trainable_params,
        frozen_params=frozen_params,
    )


def build_optimizer(model, config: OptimConfig):
    """AdamW over the trainable parameters only."""
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ConfigError("no trainable parameters; call apply_freeze first")
    return torch.optim.AdamW(
        params,
        lr=config.learning_rate,
        betas=tuple(config.betas),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def lr_multiplier(step: int, total: int, config: OptimConfig) -> float:
    """Warmup then decay, as a multiplier on the base learning rate."""
    warmup = max(0, int(config.warmup_steps))
    if warmup and step < warmup:
        return (step + 1) / warmup

    if config.schedule == "constant":
        return 1.0

    remaining_total = max(1, total - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / remaining_total))
    floor = max(0.0, min(1.0, config.min_lr_ratio))

    if config.schedule == "linear":
        return floor + (1.0 - floor) * (1.0 - progress)
    # cosine
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer, total_steps: int, config: OptimConfig):
    import torch

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, total_steps, config)
    )


def clip_gradients(model, max_norm: float) -> float:
    """Clip trainable gradients, returning the pre-clip norm."""
    import torch

    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params or max_norm <= 0:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm))
