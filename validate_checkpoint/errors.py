"""Exception types and exit codes."""

from __future__ import annotations


class ValidationError(Exception):
    """Base class for deliberate failures in this package."""


class ContractError(ValidationError):
    """chain.toml is missing, malformed, or does not describe this generation."""


class SafetensorsFormatError(ValidationError):
    """A .safetensors file could not be parsed."""


class KingUnavailableError(ValidationError):
    """Reference files for the current king could not be obtained."""


EXIT_CLEAN = 0
EXIT_WOULD_REJECT = 1
EXIT_UNDETERMINED = 2
EXIT_USAGE = 3
