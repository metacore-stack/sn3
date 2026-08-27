"""campaign — one command for the whole chain, with the bill attached.

Six packages already do the work: sn3_monitor reads the chain, fineweb_loader
serves the corpora, train_mimo trains, evaluate_losses scores, validate_checkpoint
checks packaging. Running them by hand in the right order, with the right
arguments, on a metered machine, is where attempts get lost.

This is the thing that runs them, records how long each took and what that cost,
and refuses to look finished when the artefact is unshippable.
"""

__version__ = "1.0.0"

from .config import STAGES, CampaignConfig, Hardware
from .errors import CampaignError, StageFailed
from .runner import Campaign, CampaignResult, StageResult

__all__ = [
    "Campaign", "CampaignConfig", "CampaignResult", "StageResult", "Hardware",
    "STAGES", "CampaignError", "StageFailed", "__version__",
]
