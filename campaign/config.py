"""What one attempt at the throne consists of."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import CampaignError

STAGES = ("state", "data", "train", "score", "validate", "report")

DEFAULT_STATE_ROOT = Path.home() / "Documents" / "sn3" / "state"


@dataclass
class Hardware:
    """Enough to turn wall-clock into dollars.

    Recorded per campaign rather than per run because the machine is rented for
    the campaign; an experiment that occupies eight GPUs for an hour costs eight
    GPU-hours whether or not it used them well.
    """

    n_gpus: int = 1
    usd_per_gpu_hour: float = 0.0
    torchrun: bool = False
    backend: str = ""

    def gpu_hours(self, seconds: float) -> float:
        return round(seconds / 3600.0 * max(1, self.n_gpus), 4)

    def usd(self, seconds: float) -> float:
        return round(self.gpu_hours(seconds) * self.usd_per_gpu_hour, 2)


@dataclass
class CampaignConfig:
    """One attempt: train something, measure it, decide whether it is shippable."""

    name: str = "attempt-001"
    run_name: str = ""
    train_config: str | None = None
    holdout: str = "blend-a"

    king_digest: str = ""
    king_dir: str | None = None
    model_dir: str | None = None
    submission_name: str | None = None

    output_dir: str = "runs"
    state_root: str = str(DEFAULT_STATE_ROOT)
    evidence_root: str = ""

    hardware: Hardware = field(default_factory=Hardware)
    stages: tuple[str, ...] = STAGES
    train_args: tuple[str, ...] = ()
    score_limit: int | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.hardware, dict):
            self.hardware = Hardware(**self.hardware)
        self.stages = tuple(self.stages)
        self.train_args = tuple(self.train_args)
        self.notes = tuple(self.notes)
        unknown = [s for s in self.stages if s not in STAGES]
        if unknown:
            raise CampaignError(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")
        if not self.run_name:
            self.run_name = self.name

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name

    @property
    def evidence_dir(self) -> Path:
        return Path(self.evidence_root or (Path(self.state_root) / "evidence"))

    @property
    def journal_path(self) -> Path:
        return self.run_dir / "campaign.json"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stages"] = list(self.stages)
        payload["train_args"] = list(self.train_args)
        payload["notes"] = list(self.notes)
        return payload

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "CampaignConfig":
        path = Path(path)
        if not path.is_file():
            raise CampaignError(f"{path} does not exist")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CampaignError(f"{path} is not valid JSON: {exc}") from exc
        known = {k: v for k, v in payload.items() if k in cls.__annotations__}
        return cls(**known)

    @classmethod
    def starter(cls, name: str = "attempt-001") -> "CampaignConfig":
        return cls(
            name=name,
            notes=(
                "Fill in king_digest from 'sn3 status' before running.",
                "This controller never runs 'teutonic-miner ready'.",
            ),
        )
