"""The training loop.

Every step runs inside ``mimo_adapter.trainable_routing``, so the locked
``modeling_mimo_v2.py`` is never modified on disk. Batches come from
``fineweb_loader``'s contamination-guarded stream rather than from the
filesystem, so held-out sequences cannot leak into training even by accident.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .balance import BalanceStats, LoadBalancer
from .checkpoint import SaveResult, save_checkpoint
from .config import TrainingConfig
from .errors import DivergenceError, TrainingError
from .optim import FreezeResult, apply_freeze, build_optimizer, build_scheduler, clip_gradients


@dataclass
class StepMetrics:
    step: int
    loss: float
    lr: float
    grad_norm: float
    tokens: int
    seconds: float
    balance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "loss": round(self.loss, 6),
            "lr": self.lr,
            "grad_norm": round(self.grad_norm, 6),
            "tokens": self.tokens,
            "seconds": round(self.seconds, 3),
            "balance": self.balance,
        }


@dataclass
class TrainResult:
    run_name: str
    steps: int
    metrics: list[StepMetrics]
    checkpoints: list[SaveResult]
    freeze: FreezeResult
    balance_summary: dict[str, Any]
    sequences_seen: int
    wall_time_s: float

    @property
    def first_loss(self) -> float | None:
        return self.metrics[0].loss if self.metrics else None

    @property
    def last_loss(self) -> float | None:
        return self.metrics[-1].loss if self.metrics else None

    @property
    def loss_delta(self) -> float | None:
        if self.first_loss is None or self.last_loss is None:
            return None
        return self.first_loss - self.last_loss

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "steps": self.steps,
            "sequences_seen": self.sequences_seen,
            "wall_time_s": round(self.wall_time_s, 2),
            "first_loss": self.first_loss,
            "last_loss": self.last_loss,
            "loss_delta": self.loss_delta,
            "freeze": self.freeze.summary(),
            "balance": self.balance_summary,
            "checkpoints": [str(c.model_dir) for c in self.checkpoints],
            "metrics": [m.to_dict() for m in self.metrics],
        }


class Trainer:
    """Continued pretraining of a MiMo checkpoint.

    The model, the architecture handle and a batch source are supplied by the
    caller, so the same loop drives a miniature on CPU and the real 110B on a
    rented node.
    """

    def __init__(
        self,
        *,
        model,
        arch,
        config: TrainingConfig,
        batches: Iterator,
        king_dir: Path | None = None,
        output_dir: Path | None = None,
        on_log: Callable[[StepMetrics], None] | None = None,
    ):
        self.model = model
        self.arch = arch
        self.config = config
        self.batches = batches
        self.king_dir = Path(king_dir) if king_dir else None
        self.output_dir = Path(output_dir or Path(config.output_dir) / config.run_name)
        self.on_log = on_log

        self.freeze = apply_freeze(model, config.trainable_patterns)
        self.optimizer = build_optimizer(model, config.optim)
        self.scheduler = build_scheduler(self.optimizer, config.max_steps, config.optim)

        from mimo_adapter.patch import gates

        self.balancer = LoadBalancer(
            gates(model),
            update_rate=config.balance.update_rate,
            min_tokens=config.balance.min_tokens,
            enabled=config.balance.enabled,
        )

        self.step = 0
        self.sequences_seen = 0
        self.metrics: list[StepMetrics] = []
        self.checkpoints: list[SaveResult] = []
        self._exhausted = False

    # -- helpers -----------------------------------------------------------

    def _next_batch(self):
        """One micro-batch of token ids as a LongTensor, or None when dry."""
        import torch

        try:
            item = next(self.batches)
        except StopIteration:
            self._exhausted = True
            return None
        refs, rows = item if isinstance(item, tuple) else (None, item)
        if refs is not None:
            self.sequences_seen += len(refs)
        # Sources hand back one of three shapes: a torch tensor (synthetic), a
        # numpy array (fineweb_loader with numpy present), or a list of
        # array.array rows (fineweb_loader's stdlib fallback).
        if isinstance(rows, torch.Tensor):
            tensor = rows
        elif hasattr(rows, "astype"):  # numpy array
            tensor = torch.as_tensor(rows.astype("int64"))
        else:
            tensor = torch.as_tensor([list(row) for row in rows], dtype=torch.long)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.long()

    def _run_micro_batches(self, recorder) -> tuple[float, int]:
        """Accumulate gradients over ``grad_accum`` micro-batches."""
        import torch

        from mimo_adapter.patch import trainable_routing

        total_loss = 0.0
        seen = 0
        accum = self.config.data.grad_accum

        with trainable_routing(self.arch, recorder=recorder):
            for _ in range(accum):
                batch = self._next_batch()
                if batch is None:
                    break
                outputs = self.model(input_ids=batch, labels=batch)
                loss = outputs.loss
                if not torch.isfinite(loss):
                    raise DivergenceError(
                        f"loss is {loss.item()} at step {self.step}; stopping"
                    )
                (loss / accum).backward()
                total_loss += float(loss.item())
                seen += 1

        if seen == 0:
            return 0.0, 0
        return total_loss / seen, seen

    # -- the loop ----------------------------------------------------------

    def train(self) -> TrainResult:
        from mimo_adapter.patch import RoutingRecorder

        started = time.monotonic()
        self.model.train()

        while self.step < self.config.max_steps and not self._exhausted:
            step_started = time.monotonic()
            recorder = RoutingRecorder()
            self.optimizer.zero_grad(set_to_none=True)

            loss, micro = self._run_micro_batches(recorder)
            if micro == 0:
                break

            grad_norm = clip_gradients(self.model, self.config.optim.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            balance = self.balancer.update(recorder, step=self.step)

            metrics = StepMetrics(
                step=self.step,
                loss=loss,
                lr=float(self.optimizer.param_groups[0]["lr"]),
                grad_norm=grad_norm,
                tokens=recorder.tokens,
                seconds=time.monotonic() - step_started,
                balance=balance.to_dict(),
            )
            self.metrics.append(metrics)
            if self.on_log and (
                self.config.log_every and self.step % self.config.log_every == 0
            ):
                self.on_log(metrics)

            self.step += 1
            if self.config.save_every and self.step % self.config.save_every == 0:
                self.save(metrics)

        self.model.eval()
        if not self.checkpoints or self.checkpoints[-1].step != self.step:
            self.save(self.metrics[-1] if self.metrics else None)

        return TrainResult(
            run_name=self.config.run_name,
            steps=self.step,
            metrics=self.metrics,
            checkpoints=self.checkpoints,
            freeze=self.freeze,
            balance_summary=self.balancer.summary(),
            sequences_seen=self.sequences_seen,
            wall_time_s=time.monotonic() - started,
        )

    def save(self, metrics: StepMetrics | None = None) -> SaveResult:
        result = save_checkpoint(
            model=self.model,
            step=self.step,
            output_dir=self.output_dir,
            king_dir=self.king_dir,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            balancer=self.balancer,
            metrics=metrics.to_dict() if metrics else {},
        )
        self.checkpoints.append(result)
        return result

    def resume(self, state_dir: Path) -> int:
        """Restore optimizer, scheduler, RNG and routing bias from a state dir."""
        from .checkpoint import load_checkpoint_state

        self.step = load_checkpoint_state(
            state_dir,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            balancer=self.balancer,
        )
        return self.step

    def write_report(self, result: TrainResult, path: Path | None = None) -> Path:
        path = Path(path or (self.output_dir / "run.json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": self.config.to_dict(), "result": result.to_dict()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
