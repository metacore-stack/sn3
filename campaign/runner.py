"""The controller: one command from live state to a shippable verdict.

Six stages, each of which can be skipped, re-run or resumed:

``state``     what the chain says right now -- king, threshold, corpus mixture
``data``      the corpora and the holdout the measurement will use
``train``     the run itself, under torchrun when the hardware calls for it
``score``     the resulting checkpoint into a loss vector, filed as evidence
``validate``  every packaging rule, before anything irreversible
``report``    standings, spend, and what would still block a submission

Progress is journalled, so a campaign interrupted in ``train`` resumes at
``train`` rather than repeating the three stages before it.

One thing this deliberately does not do: run ``teutonic-miner ready``. That
call is irreversible and permanently consumes the hotkey, so it stays a
decision a person makes by hand, with this report in front of them.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import STAGES, CampaignConfig
from .errors import CampaignError, StageFailed

# Anything matching this must never appear in a command this module runs.
FORBIDDEN = ("teutonic-miner ready", "miner ready")


@dataclass
class StageResult:
    name: str
    status: str = "pending"  # ok | failed | skipped | blocked
    seconds: float = 0.0
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "skipped")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "seconds": round(self.seconds, 2),
            "detail": self.detail,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StageResult":
        return cls(
            name=payload["name"],
            status=payload.get("status", "pending"),
            seconds=float(payload.get("seconds", 0.0)),
            detail=payload.get("detail", ""),
            data=payload.get("data") or {},
        )


@dataclass
class CampaignResult:
    config: CampaignConfig
    stages: list[StageResult] = field(default_factory=list)
    started: str = ""
    finished: str = ""

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.stages)

    @property
    def failed(self) -> list[StageResult]:
        return [s for s in self.stages if s.status in ("failed", "blocked")]

    def stage(self, name: str) -> StageResult | None:
        return next((s for s in self.stages if s.name == name), None)

    @property
    def billable_seconds(self) -> float:
        """Stages that occupy the hardware. Reading a dashboard does not."""
        return sum(s.seconds for s in self.stages if s.name in ("train", "score"))

    def cost(self) -> dict[str, Any]:
        hw = self.config.hardware
        seconds = self.billable_seconds
        per_stage = {
            s.name: {
                "seconds": round(s.seconds, 1),
                "gpu_hours": hw.gpu_hours(s.seconds) if s.name in ("train", "score") else 0.0,
                "usd": hw.usd(s.seconds) if s.name in ("train", "score") else 0.0,
            }
            for s in self.stages
        }
        return {
            "n_gpus": hw.n_gpus,
            "usd_per_gpu_hour": hw.usd_per_gpu_hour,
            "billable_hours": round(seconds / 3600.0, 3),
            "gpu_hours": hw.gpu_hours(seconds),
            "usd": hw.usd(seconds),
            "wall_hours_per_attempt": round(
                sum(s.seconds for s in self.stages) / 3600.0, 3
            ),
            "per_stage": per_stage,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign": self.config.name,
            "run_name": self.config.run_name,
            "started": self.started,
            "finished": self.finished,
            "ok": self.ok,
            "cost": self.cost(),
            "stages": [s.to_dict() for s in self.stages],
            "config": self.config.to_dict(),
        }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Campaign:
    def __init__(
        self,
        config: CampaignConfig,
        *,
        on_log: Callable[[str], None] | None = None,
        python: str | None = None,
    ):
        self.config = config
        self.on_log = on_log or (lambda message: print(message, flush=True))
        self.python = python or sys.executable
        self.result = CampaignResult(config=config)

    # -- journal -----------------------------------------------------------

    def load_journal(self) -> dict[str, StageResult]:
        path = self.config.journal_path
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return {
            entry["name"]: StageResult.from_dict(entry)
            for entry in payload.get("stages") or []
        }

    def save_journal(self) -> Path:
        path = self.config.journal_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.result.finished = _now()
        path.write_text(json.dumps(self.result.to_dict(), indent=2), encoding="utf-8")
        return path

    # -- process helper ----------------------------------------------------

    def run_command(self, argv: Sequence[str], *, stage: str) -> subprocess.CompletedProcess:
        rendered = " ".join(str(a) for a in argv)
        for banned in FORBIDDEN:
            if banned in rendered:
                raise CampaignError(
                    f"refusing to run {banned!r}: it is irreversible and "
                    "permanently consumes the hotkey. Run it by hand."
                )
        self.on_log(f"    $ {rendered}")
        return subprocess.run(
            [str(a) for a in argv], cwd=Path.cwd(), text=True, capture_output=True
        )

    # -- stages ------------------------------------------------------------

    def stage_state(self) -> StageResult:
        """What the chain says right now. Everything downstream depends on it."""
        from sn3_monitor.fetch import fetch_dashboard, fetch_datasets
        from sn3_monitor.target import Target

        dashboard = fetch_dashboard()
        datasets = fetch_datasets()
        live = Target.from_live(dashboard, datasets)
        data = {
            "king_digest": live.king_digest,
            "king_loss": live.king_loss,
            "delta": live.delta,
            "eval_n": live.eval_n,
            "generation": getattr(live, "generation", None),
        }
        detail = f"king {str(live.king_digest)[:12]} · delta {live.delta}"

        if self.config.king_digest and live.king_digest != self.config.king_digest:
            return StageResult(
                "state",
                "ok",
                detail=detail + " — DIFFERENT from the configured king",
                data=dict(
                    data,
                    warning=(
                        f"configured king {self.config.king_digest[:12]} is no longer "
                        f"on the throne; you will be judged against {str(live.king_digest)[:12]} "
                        "or whoever holds it when your evaluation runs"
                    ),
                ),
            )
        return StageResult("state", "ok", detail=detail, data=data)

    def stage_data(self) -> StageResult:
        """The corpora and the holdout. Refuses a holdout that does not match."""
        from evaluate_losses.scoring import plan as scoring_plan

        preview = scoring_plan(
            self.config.model_dir or self.config.run_dir,
            self.config.holdout,
            state_root=Path(self.config.state_root),
        )
        status = "ok"
        detail = (
            f"{preview.sequences} sequences · mixture error "
            f"{preview.max_share_error:.1%}"
        )
        if preview.max_share_error > 0.05:
            status = "blocked"
            detail += " — this holdout does not measure what the validator scores"
        return StageResult("data", status, detail=detail, data=preview.to_dict())

    def stage_train(self) -> StageResult:
        argv: list[str] = [self.python]
        hw = self.config.hardware
        if hw.torchrun and hw.n_gpus > 1:
            argv += [
                "-m", "torch.distributed.run",
                "--standalone", f"--nproc_per_node={hw.n_gpus}",
                "-m", "train_mimo",
            ]
        else:
            argv += ["-m", "train_mimo"]
        argv += ["train", "--run-name", self.config.run_name]
        if self.config.train_config:
            argv += ["--config", self.config.train_config]
        if self.config.model_dir:
            argv += ["--model-dir", self.config.model_dir]
        if self.config.king_dir:
            argv += ["--king", self.config.king_dir]
        if self.config.king_digest:
            argv += ["--king-digest", self.config.king_digest]
        argv += ["--holdout", self.config.holdout]
        argv += ["--output", str(self.config.run_dir)]
        argv += list(self.config.train_args)

        completed = self.run_command(argv, stage="train")
        if completed.returncode != 0:
            return StageResult(
                "train", "failed",
                detail=f"exit {completed.returncode}",
                data={"stderr": completed.stderr[-4000:]},
            )

        from train_mimo.checkpoint import latest_checkpoint

        found = latest_checkpoint(self.config.run_dir)
        if not found:
            return StageResult(
                "train", "failed",
                detail="training reported success but wrote no checkpoint",
            )
        model_dir, state_dir = found
        return StageResult(
            "train", "ok",
            detail=f"checkpoint {model_dir.name}",
            data={"model_dir": str(model_dir), "state_dir": str(state_dir)},
        )

    def resolve_model_dir(self) -> str | None:
        """The checkpoint this campaign produced, however we got here.

        Looks at this session first, then the journal, then the configured
        starting checkpoint, then the run directory itself -- so `--only score`
        or `--only validate` works against a checkpoint an earlier invocation
        trained, which is the whole point of having stages.
        """
        train = self.result.stage("train")
        if train and train.data.get("model_dir"):
            return train.data["model_dir"]

        journalled = self.load_journal().get("train")
        if journalled and journalled.data.get("model_dir"):
            return journalled.data["model_dir"]

        from train_mimo.checkpoint import latest_checkpoint

        found = latest_checkpoint(self.config.run_dir)
        if found:
            return str(found[0])
        return self.config.model_dir

    def stage_score(self) -> StageResult:
        model_dir = self.resolve_model_dir()
        if not model_dir:
            return StageResult(
                "score", "failed", detail="no checkpoint to score; run train first"
            )

        from evaluate_losses.evidence import Cost, EvidenceStore
        from evaluate_losses.scoring import score_checkpoint

        started = time.monotonic()
        vector = score_checkpoint(
            model_dir,
            self.config.holdout,
            state_root=Path(self.config.state_root),
            model_label=self.config.run_name,
            limit=self.config.score_limit,
        )
        seconds = time.monotonic() - started

        store = EvidenceStore(self.config.evidence_dir).load()
        store.record(
            vector,
            run_id=self.config.run_name,
            provenance={
                "campaign": self.config.name,
                "model_dir": str(model_dir),
                "holdout": self.config.holdout,
                "king_digest": self.config.king_digest,
            },
            cost=Cost(
                gpu_hours=self.config.hardware.gpu_hours(seconds),
                usd_per_gpu_hour=self.config.hardware.usd_per_gpu_hour,
                n_gpus=self.config.hardware.n_gpus,
            ),
            overwrite=True,
        )
        return StageResult(
            "score", "ok",
            detail=f"mean {vector.mean:.6f} over {len(vector)} sequences",
            data={
                "mean_loss": vector.mean,
                "n": len(vector),
                "by_corpus": {k: round(v.mean, 6) for k, v in vector.by_corpus().items()},
                "recorded_as": self.config.run_name,
            },
        )

    def stage_validate(self) -> StageResult:
        model_dir = self.resolve_model_dir()
        if not model_dir:
            return StageResult("validate", "failed", detail="no checkpoint to validate")

        from validate_checkpoint import Contract, KingReference, Options, validate

        king = None
        if self.config.king_dir:
            king = KingReference.from_directory(self.config.king_dir)
        elif self.config.king_digest:
            try:
                king = KingReference.from_digest(self.config.king_digest)
            except Exception as exc:  # noqa: BLE001 - reported, never fatal
                self.on_log(f"    king reference unavailable: {exc}")

        report = validate(
            Path(model_dir),
            contract=Contract.load(),
            king=king,
            options=Options.thorough(self.config.submission_name),
        )
        status = "blocked" if report.would_reject else "ok"
        detail = report.verdict
        return StageResult(
            "validate", status, detail=detail,
            data={
                "would_reject": report.would_reject,
                "determinate": report.determinate,
                "failures": [c.name for c in report.failures],
                "fatal": [c.name for c in report.fatal_failures],
                "skipped": [c.name for c in report.skipped],
            },
        )

    def stage_report(self) -> StageResult:
        from evaluate_losses.evidence import EvidenceStore

        store = EvidenceStore(self.config.evidence_dir).load()
        data: dict[str, Any] = {"spend": store.spend()}
        try:
            board = store.leaderboard()
            data["leaderboard"] = board
            mine = next(
                (r for r in board["rows"] if r["run_id"] == self.config.run_name), None
            )
            data["this_run"] = mine
            if mine:
                detail = (
                    f"mu_hat {mine['mu_hat']} lcb {mine['lcb']} — "
                    + ("would be ACCEPTED" if mine["accepted"] else "would be rejected")
                )
            else:
                detail = f"{board['accepted']}/{board['challengers']} would be accepted"
        except Exception as exc:  # noqa: BLE001 - a missing king is not a failure
            detail = f"no standings available: {exc}"
            data["standings_error"] = str(exc)
        return StageResult("report", "ok", detail=detail, data=data)

    # -- the loop ----------------------------------------------------------

    def run(
        self,
        *,
        only: Sequence[str] | None = None,
        skip: Sequence[str] = (),
        resume: bool = True,
        stop_on_failure: bool = True,
        dry_run: bool = False,
    ) -> CampaignResult:
        wanted = [s for s in self.config.stages if s in (only or STAGES) and s not in skip]
        journal = self.load_journal() if resume else {}
        self.result = CampaignResult(config=self.config, started=_now())

        if not wanted:
            # Selecting nothing must not read as "everything passed".
            declared = list(self.config.stages)
            asked = list(only) if only else declared
            raise CampaignError(
                f"no stages to run: asked for {asked}, skipped {list(skip)}, but "
                f"this campaign declares {declared}. Add the stage to the "
                "campaign file's \"stages\" list first."
            )

        if dry_run:
            for name in wanted:
                previous = journal.get(name)
                note = "would re-use" if previous and previous.ok else "would run"
                self.result.stages.append(StageResult(name, "skipped", detail=note))
            return self.result

        handlers = {
            "state": self.stage_state,
            "data": self.stage_data,
            "train": self.stage_train,
            "score": self.stage_score,
            "validate": self.stage_validate,
            "report": self.stage_report,
        }

        for name in wanted:
            previous = journal.get(name)
            if resume and previous is not None and previous.status == "ok":
                self.on_log(f"  [{name}] already done — {previous.detail}")
                self.result.stages.append(
                    StageResult(name, "skipped", previous.seconds, previous.detail, previous.data)
                )
                continue

            self.on_log(f"  [{name}] running")
            started = time.monotonic()
            try:
                outcome = handlers[name]()
            except Exception as exc:  # noqa: BLE001 - a stage failure is data
                outcome = StageResult(name, "failed", detail=f"{type(exc).__name__}: {exc}")
            outcome.seconds = time.monotonic() - started
            self.result.stages.append(outcome)
            self.save_journal()

            mark = {"ok": "ok", "skipped": "--", "blocked": "BLOCKED", "failed": "FAILED"}
            self.on_log(f"  [{name}] {mark.get(outcome.status, outcome.status)} — {outcome.detail}")
            if not outcome.ok and stop_on_failure:
                self.on_log("  stopping: a later stage would be measuring nothing")
                break

        self.save_journal()
        return self.result
