"""evaluate_losses — reproduce the SN3 validator's decision offline.

The statistics are never reimplemented: ``paired_bootstrap_verdict`` is loaded
out of the cloned Teutonic repository and called directly.
"""

__version__ = "1.0.0"

from .backends import ReplayBackend, ScoringBackend, TorchBackend
from .compare import Comparison, ShardBreakdown, Verdict, compare
from .evidence import Cost, EvidenceRecord, EvidenceStore, Standing
from .scoring import ScoringPlan, open_blend, plan, score_checkpoint
from .engine import EngineSpec, StatsSpec, describe, n_positions, reduce_per_token
from .errors import AlignmentError, EngineMismatchError, EvaluationError, PolicyUnavailableError
from .lossvec import LossVector
from .policy import available, paired_bootstrap_verdict, policy_source

__all__ = [
    "LossVector", "compare", "Comparison", "Verdict", "ShardBreakdown",
    "ScoringBackend", "ReplayBackend", "TorchBackend",
    "EngineSpec",
    "EvidenceStore", "EvidenceRecord", "Standing", "Cost",
    "score_checkpoint", "plan", "open_blend", "ScoringPlan", "StatsSpec", "describe", "n_positions", "reduce_per_token",
    "paired_bootstrap_verdict", "policy_source", "available",
    "EvaluationError", "AlignmentError", "EngineMismatchError", "PolicyUnavailableError",
    "__version__",
]
