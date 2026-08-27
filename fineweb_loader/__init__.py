"""fineweb_loader — reproducible, verified access to the SN3 FineWeb-Edu shards.

numpy is optional: a pure-stdlib .npy reader is used when it is absent.
"""

__version__ = "1.0.0"

from .cache import ShardCache
from .errors import ContaminationError, IntegrityError, LoaderError, ManifestError
from .loader import FineWebLoader
from .manifest import ShardEntry, ShardManifest, canonical_sha256
from .npyio import NUMPY_AVAILABLE, Shard
from .refs import SequenceRef, SequenceSet

__all__ = [
    "ShardCache",
    "ShardManifest",
    "ShardEntry",
    "Shard",
    "SequenceRef",
    "SequenceSet",
    "FineWebLoader",
    "canonical_sha256",
    "ContaminationError",
    "IntegrityError",
    "LoaderError",
    "ManifestError",
    "NUMPY_AVAILABLE",
    "__version__",
]
