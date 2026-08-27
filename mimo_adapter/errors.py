"""Exception types and exit codes."""

from __future__ import annotations


class AdapterError(Exception):
    """Base class for deliberate failures in this package."""


class ArchSourceError(AdapterError):
    """The MiMo architecture code could not be located or imported."""


class ParityError(AdapterError):
    """The patched router does not reproduce the original's output.

    Fatal on purpose. A router that is subtly wrong still produces plausible
    losses, so training against it wastes an entire GPU budget before the
    mistake becomes visible.
    """


class GradientError(AdapterError):
    """Backpropagation did not reach the parameters it was expected to."""


EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
