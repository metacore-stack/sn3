"""The controller.

Every stage here is stubbed. What is under test is the orchestration: that a
blocked stage stops the ones that would have measured nothing, that the journal
makes an interrupted campaign resumable, that cost is attributed to the stages
that actually occupy the hardware, and that nothing in this package can be made
to run the one irreversible command.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from campaign.config import STAGES, CampaignConfig, Hardware
from campaign.errors import CampaignError
from campaign.runner import Campaign, CampaignResult, StageResult


class ConfigTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_run_name_defaults_to_the_campaign_name(self):
        self.assertEqual(CampaignConfig(name="attempt-7").run_name, "attempt-7")

    def test_unknown_stage_is_rejected(self):
        with self.assertRaises(CampaignError) as ctx:
            CampaignConfig(stages=("state", "deploy"))
        self.assertIn("deploy", str(ctx.exception))

    def test_round_trips_through_json(self):
        config = CampaignConfig(
            name="a", holdout="blend-a", hardware=Hardware(n_gpus=8, usd_per_gpu_hour=2.4)
        )
        path = config.save(self.tmp / "c.json")
        again = CampaignConfig.load(path)
        self.assertEqual(again.hardware.n_gpus, 8)
        self.assertEqual(again.hardware.usd_per_gpu_hour, 2.4)
        self.assertEqual(again.stages, STAGES)

    def test_loading_a_missing_file_says_so(self):
        with self.assertRaises(CampaignError):
            CampaignConfig.load(self.tmp / "nope.json")

    def test_loading_junk_says_so(self):
        path = self.tmp / "bad.json"
        path.write_text("{not json")
        with self.assertRaises(CampaignError):
            CampaignConfig.load(path)

    def test_evidence_defaults_under_the_state_root(self):
        config = CampaignConfig(state_root="/s")
        self.assertEqual(config.evidence_dir, Path("/s/evidence"))


class HardwareTestCase(unittest.TestCase):
    def test_gpu_hours_multiply_by_device_count(self):
        hw = Hardware(n_gpus=8, usd_per_gpu_hour=2.5)
        self.assertEqual(hw.gpu_hours(3600), 8.0)
        self.assertEqual(hw.usd(3600), 20.0)

    def test_a_single_gpu_is_assumed_when_none_is_declared(self):
        self.assertEqual(Hardware(n_gpus=0).gpu_hours(3600), 1.0)


class StubCampaign(Campaign):
    """A campaign whose stages are whatever the test says they are."""

    def __init__(self, config, outcomes, **kwargs):
        super().__init__(config, on_log=lambda m: None, **kwargs)
        self.outcomes = outcomes
        self.ran: list[str] = []

    def _stub(self, name):
        def handler():
            self.ran.append(name)
            outcome = self.outcomes.get(name)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome or StageResult(name, "ok", detail=f"{name} done")

        return handler

    def run(self, **kwargs):
        for name in STAGES:
            setattr(self, f"stage_{name}", self._stub(name))
        return super().run(**kwargs)


class OrchestrationTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.config = CampaignConfig(
            name="t",
            output_dir=str(self.tmp / "runs"),
            evidence_root=str(self.tmp / "evidence"),
            hardware=Hardware(n_gpus=4, usd_per_gpu_hour=3.0),
        )

    def test_all_stages_run_in_order(self):
        campaign = StubCampaign(self.config, {})
        result = campaign.run()
        self.assertEqual(campaign.ran, list(STAGES))
        self.assertTrue(result.ok)

    def test_a_blocked_stage_stops_the_ones_that_would_measure_nothing(self):
        outcomes = {"data": StageResult("data", "blocked", detail="wrong mixture")}
        campaign = StubCampaign(self.config, outcomes)
        result = campaign.run()
        self.assertEqual(campaign.ran, ["state", "data"])
        self.assertFalse(result.ok)
        self.assertEqual([s.name for s in result.failed], ["data"])

    def test_keep_going_runs_everything_anyway(self):
        outcomes = {"data": StageResult("data", "blocked", detail="wrong mixture")}
        campaign = StubCampaign(self.config, outcomes)
        campaign.run(stop_on_failure=False)
        self.assertEqual(campaign.ran, list(STAGES))

    def test_an_exception_becomes_a_failed_stage_not_a_crash(self):
        campaign = StubCampaign(self.config, {"train": RuntimeError("cuda oom")})
        result = campaign.run()
        stage = result.stage("train")
        self.assertEqual(stage.status, "failed")
        self.assertIn("cuda oom", stage.detail)

    def test_only_and_skip_select_stages(self):
        campaign = StubCampaign(self.config, {})
        campaign.run(only=["state", "train"])
        self.assertEqual(campaign.ran, ["state", "train"])

        campaign = StubCampaign(self.config, {})
        # resume=False, or the journal from the run above would skip 'state'.
        campaign.run(skip=["train", "score"], resume=False)
        self.assertEqual(campaign.ran, ["state", "data", "validate", "report"])

    def test_selecting_nothing_is_an_error_not_a_pass(self):
        """An empty selection reported 'every stage completed', which is a lie."""
        config = CampaignConfig(
            name="t", stages=("state", "data"), output_dir=str(self.tmp / "runs")
        )
        campaign = StubCampaign(config, {})
        with self.assertRaises(CampaignError) as ctx:
            campaign.run(only=["validate"])
        self.assertIn("no stages to run", str(ctx.exception))
        self.assertIn("state", str(ctx.exception))

    def test_dry_run_touches_nothing(self):
        campaign = StubCampaign(self.config, {})
        result = campaign.run(dry_run=True)
        self.assertEqual(campaign.ran, [])
        self.assertTrue(all(s.status == "skipped" for s in result.stages))


class ResumeTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.config = CampaignConfig(
            name="t", output_dir=str(self.tmp / "runs"), evidence_root=str(self.tmp / "e")
        )

    def test_a_completed_stage_is_not_re_run(self):
        StubCampaign(self.config, {}).run()
        again = StubCampaign(self.config, {})
        again.run()
        self.assertEqual(again.ran, [])

    def test_an_interrupted_campaign_resumes_where_it_stopped(self):
        first = StubCampaign(
            self.config, {"train": StageResult("train", "failed", detail="oom")}
        )
        first.run()
        self.assertEqual(first.ran, ["state", "data", "train"])

        second = StubCampaign(self.config, {})
        second.run()
        # state and data are reused; train onwards actually runs.
        self.assertEqual(second.ran, ["train", "score", "validate", "report"])

    def test_no_resume_re_runs_everything(self):
        StubCampaign(self.config, {}).run()
        again = StubCampaign(self.config, {})
        again.run(resume=False)
        self.assertEqual(again.ran, list(STAGES))

    def test_the_journal_is_written_and_readable(self):
        StubCampaign(self.config, {}).run()
        payload = json.loads(self.config.journal_path.read_text())
        self.assertEqual(payload["campaign"], "t")
        self.assertEqual(len(payload["stages"]), len(STAGES))
        self.assertIn("cost", payload)

    def test_the_journal_survives_a_stage_failure(self):
        StubCampaign(self.config, {"data": RuntimeError("boom")}).run()
        payload = json.loads(self.config.journal_path.read_text())
        self.assertEqual(payload["stages"][-1]["status"], "failed")


class ModelResolutionTestCase(unittest.TestCase):
    """`--only validate` must find the checkpoint an earlier invocation trained."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.config = CampaignConfig(name="t", output_dir=str(self.tmp / "runs"))

    def test_this_session_wins(self):
        campaign = Campaign(self.config, on_log=lambda m: None)
        campaign.result.stages.append(
            StageResult("train", "ok", data={"model_dir": "/from/session"})
        )
        self.assertEqual(campaign.resolve_model_dir(), "/from/session")

    def test_the_journal_is_consulted_when_this_session_did_not_train(self):
        first = StubCampaign(
            self.config,
            {"train": StageResult("train", "ok", data={"model_dir": "/from/journal"})},
        )
        first.run(only=["train"])

        second = Campaign(self.config, on_log=lambda m: None)
        self.assertEqual(second.resolve_model_dir(), "/from/journal")

    def test_the_configured_checkpoint_is_the_last_resort(self):
        config = CampaignConfig(
            name="t", output_dir=str(self.tmp / "runs"), model_dir="/configured"
        )
        campaign = Campaign(config, on_log=lambda m: None)
        self.assertEqual(campaign.resolve_model_dir(), "/configured")

    def test_nothing_anywhere_returns_none(self):
        self.assertIsNone(
            Campaign(self.config, on_log=lambda m: None).resolve_model_dir()
        )


class CostTestCase(unittest.TestCase):
    def setUp(self):
        self.config = CampaignConfig(
            name="t", hardware=Hardware(n_gpus=8, usd_per_gpu_hour=2.0)
        )

    def _result(self, **seconds) -> CampaignResult:
        result = CampaignResult(config=self.config)
        for name in STAGES:
            result.stages.append(StageResult(name, "ok", seconds=seconds.get(name, 0.0)))
        return result

    def test_only_the_hardware_stages_are_billed(self):
        """Reading a dashboard does not occupy eight GPUs."""
        result = self._result(state=600.0, train=3600.0, score=1800.0, report=600.0)
        cost = result.cost()
        self.assertEqual(cost["billable_hours"], 1.5)
        self.assertEqual(cost["gpu_hours"], 12.0)
        self.assertEqual(cost["usd"], 24.0)

    def test_wall_hours_count_every_stage(self):
        result = self._result(state=1800.0, train=1800.0)
        self.assertEqual(result.cost()["wall_hours_per_attempt"], 1.0)

    def test_per_stage_attribution_is_reported(self):
        result = self._result(train=3600.0, state=3600.0)
        per_stage = result.cost()["per_stage"]
        self.assertEqual(per_stage["train"]["usd"], 16.0)
        self.assertEqual(per_stage["state"]["usd"], 0.0)


class SafetyTestCase(unittest.TestCase):
    """The one command this package must never be able to run."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.config = CampaignConfig(name="t", output_dir=str(self.tmp))

    def test_the_irreversible_command_is_refused(self):
        campaign = Campaign(self.config, on_log=lambda m: None)
        for argv in (
            ["teutonic-miner", "ready"],
            ["bash", "-c", "teutonic-miner ready --netuid 3"],
        ):
            with self.assertRaises(CampaignError) as ctx:
                campaign.run_command(argv, stage="train")
            self.assertIn("irreversible", str(ctx.exception))

    def test_ordinary_commands_are_allowed(self):
        campaign = Campaign(self.config, on_log=lambda m: None)
        completed = campaign.run_command(["true"], stage="train")
        self.assertEqual(completed.returncode, 0)

    def test_no_stage_is_named_submit(self):
        self.assertNotIn("submit", STAGES)
        self.assertNotIn("ready", STAGES)


if __name__ == "__main__":
    unittest.main()
