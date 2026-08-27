"""train_mimo — continued pretraining of a MiMo checkpoint for SN3.

Runs inside ``mimo_adapter.trainable_routing`` so the hash-pinned architecture
file is never modified, draws batches only from ``fineweb_loader``'s
contamination-guarded stream, and writes checkpoints whose six locked files are
restored byte-identical from the king.
"""

__version__ = "1.0.0"

from .balance import BalanceStats, LoadBalancer
from .checkpoint import (
    LOCKED_FILES,
    SaveResult,
    latest_checkpoint,
    load_checkpoint_state,
    prune_checkpoints,
    restore_locked_files,
    save_checkpoint,
    verify_locked_files,
)
from .config import STAGES, BalanceConfig, DataConfig, OptimConfig, TrainingConfig
from .errors import CheckpointError, ConfigError, DivergenceError, ResumeError, TrainingError
from .optim import FreezeResult, apply_freeze, build_optimizer, build_scheduler, lr_multiplier
from .sources import fineweb_batches, resolve_batches, synthetic_batches
from .trainer import StepMetrics, Trainer, TrainResult

__all__ = [
    "Trainer", "TrainResult", "StepMetrics",
    "TrainingConfig", "DataConfig", "OptimConfig", "BalanceConfig", "STAGES",
    "LoadBalancer", "BalanceStats",
    "apply_freeze", "build_optimizer", "build_scheduler", "lr_multiplier", "FreezeResult",
    "save_checkpoint", "load_checkpoint_state", "restore_locked_files",
    "verify_locked_files", "latest_checkpoint", "prune_checkpoints",
    "SaveResult", "LOCKED_FILES",
    "resolve_batches", "synthetic_batches", "fineweb_batches",
    "TrainingError", "ConfigError", "CheckpointError", "ResumeError", "DivergenceError",
    "__version__",
]
