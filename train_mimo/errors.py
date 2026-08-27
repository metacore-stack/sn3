"""Exception types and exit codes."""

from __future__ import annotations


class TrainingError(Exception):
    """Base class for deliberate failures in this package."""


class ConfigError(TrainingError):
    """A run configuration is missing something or contradicts itself."""


class CheckpointError(TrainingError):
    """A checkpoint could not be written, restored, or made submittable."""


class ResumeError(TrainingError):
    """Saved training state does not match the run being resumed."""


class DivergenceError(TrainingError):
    """Loss became non-finite. Continuing would only waste compute."""


EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
