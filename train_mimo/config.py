"""Run configuration.

Everything that defines an experiment lives here so that a run is described by
one file rather than by a command line nobody wrote down. The config is copied
verbatim into the run's provenance record.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .errors import ConfigError

# Parameter-group selectors, matched as substrings against parameter names.
# Named stages exist so staged unfreezing is a config change, not a code change.
STAGES: dict[str, tuple[str, ...]] = {
    "shared": (".shared_experts.",),
    "shared+router": (".shared_experts.", ".gate.weight"),
    "experts": (".shared_experts.", ".experts.", ".gate.weight"),
    "experts+attention": (
        ".shared_experts.",
        ".experts.",
        ".gate.weight",
        ".self_attn.",
    ),
    "all": (),  # empty means everything
}


@dataclass
class BalanceConfig:
    """The routing-bias load-balancing rule.

    ``e_score_correction_bias`` receives no gradient -- it steers selection, and
    selection is not differentiable. Something has to move it, or routing
    collapses onto whichever experts start out slightly favoured.

    The rule is the standard auxiliary-loss-free one: nudge each expert's bias
    against its recent share of the tokens, by a fixed step.
    """

    enabled: bool = True
    update_rate: float = 1e-3
    # Only act once a step has routed enough tokens for the counts to mean
    # something; below this the signal is noise.
    min_tokens: int = 64


@dataclass
class OptimConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 10
    schedule: str = "cosine"  # cosine | linear | constant
    min_lr_ratio: float = 0.1


@dataclass
class DataConfig:
    shards: tuple[str, ...] = ()
    holdouts: tuple[str, ...] = ()
    batch_size: int = 2
    grad_accum: int = 1
    seed: int = 1234
    max_batches: int | None = None


@dataclass
class TrainingConfig:
    """One experiment, in full."""

    run_name: str = "pilot-001"
    max_steps: int = 100
    save_every: int = 50
    eval_every: int = 0  # 0 disables mid-run evaluation
    log_every: int = 10
    stage: str = "all"
    seed: int = 0
    dtype: str = "float32"

    king_dir: str | None = None
    arch_dir: str | None = None
    output_dir: str = "runs"

    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)

    # Provenance, filled in by the runner.
    target_snapshot: str | None = None
    king_digest: str | None = None
    manifest_sha256: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ConfigError(
                f"unknown stage {self.stage!r}; choose from {sorted(STAGES)}"
            )
        if self.max_steps < 1:
            raise ConfigError("max_steps must be at least 1")
        if self.data.grad_accum < 1:
            raise ConfigError("grad_accum must be at least 1")
        if self.data.batch_size < 1:
            raise ConfigError("batch_size must be at least 1")
        if self.optim.schedule not in ("cosine", "linear", "constant"):
            raise ConfigError(f"unknown schedule {self.optim.schedule!r}")
        overlap = set(self.data.shards) & set()
        if overlap:  # pragma: no cover - placeholder for future checks
            raise ConfigError(str(overlap))

    @property
    def trainable_patterns(self) -> tuple[str, ...]:
        return STAGES[self.stage]

    @property
    def tokens_per_step(self) -> int:
        return self.data.batch_size * self.data.grad_accum

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        payload = dict(payload)
        nested = {
            "data": DataConfig,
            "optim": OptimConfig,
            "balance": BalanceConfig,
        }
        for key, klass in nested.items():
            value = payload.get(key)
            if isinstance(value, dict):
                known = {f.name for f in fields(klass)}
                payload[key] = klass(**{k: v for k, v in value.items() if k in known})
        for key in ("notes",):
            if key in payload and payload[key] is not None:
                payload[key] = tuple(payload[key])
        if isinstance(payload.get("data"), DataConfig):
            payload["data"].shards = tuple(payload["data"].shards)
            payload["data"].holdouts = tuple(payload["data"].holdouts)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})

    @classmethod
    def load(cls, path: Path | str) -> "TrainingConfig":
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"{path} does not exist") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
        return cls.from_dict(payload)
