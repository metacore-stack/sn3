"""sn3-monitor — know what SN3 expects of a challenger, and notice when it changes.

Read-only. Touches no wallet, loads no model, writes nothing to the chain.
"""

__version__ = "1.0.0"

from .drift import Severity, Verdict, compare  # noqa: F401
from .history import Report, build_report  # noqa: F401
from .preflight import run_preflight  # noqa: F401
from .store import Store  # noqa: F401
from .target import Target  # noqa: F401

__all__ = [
    "Severity",
    "Verdict",
    "compare",
    "Report",
    "build_report",
    "run_preflight",
    "Store",
    "Target",
    "__version__",
]
