"""mimo_adapter — make the locked MiMo noaux_tc router trainable, provably.

The architecture file is hash-pinned in chain.toml and is never modified. The
gate's forward is replaced in memory for the duration of a context manager, and
the replacement is verified to be numerically identical to the original before
anything is trained against it.
"""

__version__ = "1.0.0"

from .errors import AdapterError, ArchSourceError, GradientError, ParityError
from .loader import ArchModules, fetch_arch, find_arch_directory, load_arch, read_reference_config
from .miniature import (
    MiniatureSpec,
    build_miniature,
    count_parameters,
    describe,
    initialize_attention_sinks,
    initialize_gates,
    initialize_uninitialized,
    uninitialized_parameters,
    miniature_config_dict,
)
from .patch import (
    RoutingRecorder,
    gates,
    is_patched,
    recording_routing,
    set_gates_eval,
    trainable_gate_forward,
    trainable_routing,
)
from .verify import Report, run_all

__all__ = [
    "load_arch", "fetch_arch", "find_arch_directory", "read_reference_config", "ArchModules",
    "build_miniature", "MiniatureSpec", "miniature_config_dict", "describe", "count_parameters",
    "initialize_gates",
    "initialize_attention_sinks",
    "initialize_uninitialized",
    "uninitialized_parameters",
    "trainable_routing", "recording_routing", "trainable_gate_forward",
    "RoutingRecorder", "gates", "set_gates_eval", "is_patched",
    "run_all", "Report",
    "AdapterError", "ArchSourceError", "ParityError", "GradientError",
    "__version__",
]
