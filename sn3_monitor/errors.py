"""Exception types and process exit codes."""

from __future__ import annotations


class MonitorError(Exception):
    """Base class for every error this package raises deliberately."""


class FetchError(MonitorError):
    """A remote document could not be retrieved from any configured source."""


class StaleDocumentError(MonitorError):
    """A document was retrieved but its own timestamp is too old to trust."""


class TargetNotFoundError(MonitorError):
    """A pinned target was requested by id but does not exist on disk."""


# Exit codes. These are the contract other scripts branch on, so they are
# defined once here and never inlined.
EXIT_FRESH = 0
EXIT_STALE = 1
EXIT_ABORT = 2
EXIT_FETCH_FAILED = 3
EXIT_USAGE = 4
