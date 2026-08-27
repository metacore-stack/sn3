"""Exception types and exit codes for the controller."""

from __future__ import annotations


class CampaignError(Exception):
    """A deliberate failure in the controller."""


class StageFailed(CampaignError):
    """One stage returned non-zero or raised."""


EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_FAILED = 2
EXIT_USAGE = 3
