"""Exception types and exit codes for the loader."""

from __future__ import annotations


class LoaderError(Exception):
    """Base class for deliberate failures in this package."""


class ManifestError(LoaderError):
    """The shard inventory is missing, malformed, or fails verification."""


class IntegrityError(LoaderError):
    """A downloaded shard does not match its manifest entry."""


class ShardNotFoundError(LoaderError):
    """A shard key does not appear in the manifest."""


class BudgetExceededError(LoaderError):
    """A fetch would exceed the cache's byte budget and could not be evicted for."""


class ContaminationError(LoaderError):
    """A training request overlapped a held-out sequence set.

    This is deliberately fatal. Silently training on your own validation data
    invalidates every measurement taken afterwards, and the failure is invisible
    until the GPU budget is already spent.
    """


class NpyFormatError(LoaderError):
    """A shard file is not a .npy this reader understands."""


EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTEGRITY = 3
