"""Exception types and exit codes."""

from __future__ import annotations


class EvaluationError(Exception):
    """Base class for deliberate failures in this package."""


class PolicyUnavailableError(EvaluationError):
    """The validator's paired_bootstrap_verdict could not be located."""


class AlignmentError(EvaluationError):
    """Two loss vectors do not describe the same sequences in the same order.

    Fatal on purpose. Comparing misaligned vectors yields a plausible number
    that is pure noise, and nothing downstream can detect it.
    """


class BackendUnavailableError(EvaluationError):
    """A scoring backend's dependencies are not installed."""


class EngineMismatchError(EvaluationError):
    """A backend is configured differently from the validator's engine."""


EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_FAILED = 2
EXIT_USAGE = 3
