"""validate_checkpoint — run every rule the SN3 validator runs, before 'ready'.

Roughly 1 in 11 live submissions died on packaging rather than model quality.
After 'ready' the hotkey is spent permanently, so those checks have to happen
here instead.
"""

__version__ = "1.0.0"

from .contract import Contract
from .download import DownloadPlan, FileProgress, KingDownloader
from .errors import ContractError, KingUnavailableError, SafetensorsFormatError, ValidationError
from .king import KingReference
from .report import Check, Layer, Report, Status
from .reuse import MAX_COMPLETED_EVALS, SubmissionLedger, safetensors_digest_from_file_digests
from .safetensors_io import SafetensorsHeader, TensorInfo, read_header
from .validator import Options, validate

__all__ = [
    "Contract", "KingReference",
    "KingDownloader", "DownloadPlan", "FileProgress", "Report", "Check", "Layer", "Status",
    "SafetensorsHeader", "TensorInfo", "read_header",
    "validate", "Options",
    "SubmissionLedger", "MAX_COMPLETED_EVALS", "safetensors_digest_from_file_digests",
    "ValidationError", "ContractError", "KingUnavailableError", "SafetensorsFormatError",
    "__version__",
]
