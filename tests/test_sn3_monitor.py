"""Tests for sn3_monitor. Run with: python -m unittest discover -s tests -v

Fixtures are real payloads captured from the live dashboard, so the null-heavy
rows and non-contiguous reign numbers exercised here are the genuine article
rather than something invented to be convenient.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sn3_monitor import cli
from sn3_monitor.drift import Severity, compare
from sn3_monitor.errors import EXIT_FETCH_FAILED, EXIT_USAGE, TargetNotFoundError
from sn3_monitor.fetch import Document, load_local
from sn3_monitor.history import build_report
from sn3_monitor.observe import observation, transitions, weight_uptime
from sn3_monitor.preflight import run_preflight
from sn3_monitor.store import Store
from sn3_monitor.target import Target
from sn3_monitor.timeutil import humanize, parse_duration, parse_ts

FIXTURES = Path(__file__).parent / "fixtures"
DASHBOARD = FIXTURES / "dashboard.json"
DATASETS = FIXTURES / "datasets.json"


def live_pair() -> tuple[Document, Document]:
    return (
        load_local(DASHBOARD, timestamp_keys=("updated_at", "generated_at")),
        load_local(DATASETS, timestamp_keys=("generated_at",)),
    )


def make_target() -> Target:
    dashboard, datasets = live_pair()
    return Target.from_live(dashboard, datasets)


class TimeUtilTests(unittest.TestCase):
    def test_parses_trailing_z(self):
        parsed = parse_ts("2026-08-25T19:36:41.375017Z")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.tzinfo, timezone.utc)
        self.assertEqual(parsed.year, 2026)

    def test_null_and_garbage_are_none(self):
        for value in (None, "", "   ", "not a date", 12345):
            self.assertIsNone(parse_ts(value))  # type: ignore[arg-type]

    def test_durations(self):
        self.assertEqual(parse_duration("30m"), timedelta(minutes=30))
        self.assertEqual(parse_duration("24h"), timedelta(hours=24))
        self.assertEqual(parse_duration("7d"), timedelta(days=7))
        with self.assertRaises(ValueError):
            parse_duration("soon")

    def test_humanize(self):
        self.assertEqual(humanize(timedelta(seconds=42)), "42s")
        self.assertEqual(humanize(timedelta(minutes=58)), "58m")
        self.assertEqual(humanize(None), "unknown")


class TargetTests(unittest.TestCase):
    def test_builds_from_real_payloads(self):
        target = make_target()
        self.assertTrue(target.king_digest)
        self.assertEqual(target.netuid, 3)
        self.assertEqual(target.delta, 0.5)
        self.assertEqual(target.eval_n, 2000)
        self.assertTrue(target.sources)
        self.assertEqual(target.sources[0].sequence_length, 2048)
        self.assertEqual(target.sources[0].tokenizer, "XiaomiMiMo/MiMo-V2.5-Pro")

    def test_round_trips_through_json(self):
        target = make_target()
        restored = Target.from_dict(json.loads(json.dumps(target.to_dict())))
        self.assertEqual(restored.king_digest, target.king_digest)
        self.assertEqual(restored.sources, target.sources)
        self.assertEqual(restored.delta, target.delta)

    def test_snapshot_id_embeds_digest(self):
        target = make_target()
        self.assertIn(target.short_digest, target.snapshot_id)


class DriftTests(unittest.TestCase):
    def setUp(self):
        self.live = make_target()

    def test_identical_target_is_fresh(self):
        verdict = compare(self.live, self.live)
        self.assertEqual(verdict.severity, Severity.OK)
        self.assertTrue(verdict.is_actionable)
        self.assertEqual(verdict.exit_code, 0)
        self.assertEqual(verdict.drifts, ())

    def test_changed_king_digest_is_stale(self):
        pinned = replace(self.live, king_digest="0" * 64)
        verdict = compare(pinned, self.live)
        self.assertEqual(verdict.severity, Severity.STALE)
        self.assertEqual(verdict.exit_code, 1)
        self.assertFalse(verdict.is_actionable)
        self.assertTrue(any(d.field == "king.king_digest" for d in verdict.drifts))

    def test_changed_generation_is_abort(self):
        pinned = replace(self.live, generation="teutonic-III-something")
        verdict = compare(pinned, self.live)
        self.assertEqual(verdict.severity, Severity.ABORT)
        self.assertEqual(verdict.exit_code, 2)

    def test_changed_delta_is_stale(self):
        pinned = replace(self.live, delta_from_datasets=0.25)
        verdict = compare(pinned, self.live)
        self.assertEqual(verdict.severity, Severity.STALE)
        self.assertTrue(any(d.field == "delta_threshold" for d in verdict.drifts))

    def test_changed_dataset_version_is_stale(self):
        pinned = replace(self.live, dataset_version="deadbeef")
        verdict = compare(pinned, self.live)
        self.assertEqual(verdict.severity, Severity.STALE)

    def test_changed_tokenizer_is_abort(self):
        source = replace(self.live.sources[0], tokenizer="some/other-tokenizer")
        pinned = replace(self.live, sources=(source,))
        verdict = compare(pinned, self.live)
        self.assertEqual(verdict.severity, Severity.ABORT)

    def test_changed_shard_manifest_is_stale(self):
        source = replace(self.live.sources[0], manifest_sha256="0" * 64)
        pinned = replace(self.live, sources=(source,))
        verdict = compare(pinned, self.live)
        self.assertEqual(verdict.severity, Severity.STALE)

    def test_reign_gap_warns_but_stays_actionable(self):
        reign = self.live.king_reign or 7
        pinned = replace(self.live, king_reign=reign - 5)
        verdict = compare(pinned, self.live)
        self.assertTrue(any(d.field == "king.reign_number" for d in verdict.drifts))
        self.assertTrue(verdict.is_actionable)

    def test_missing_live_value_is_not_treated_as_change(self):
        live = replace(self.live, dataset_version=None)
        verdict = compare(self.live, live)
        self.assertEqual(verdict.severity, Severity.OK)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.board = json.loads(DASHBOARD.read_text(encoding="utf-8"))

    def test_parses_real_history_without_raising(self):
        report = build_report(self.board)
        self.assertGreater(len(report.attempts), 0)

    def test_error_rows_have_null_metrics_and_survive(self):
        report = build_report(self.board)
        errored = [a for a in report.attempts if a.is_error]
        self.assertTrue(errored, "fixture should contain at least one error row")
        for attempt in errored:
            self.assertIsNone(attempt.mu_hat)
            self.assertIsNone(attempt.gap_to_bar)

    def test_aggregates_never_divide_by_null(self):
        report = build_report(self.board)
        self.assertIsNotNone(report.packaging_failure_rate)
        _ = report.median_wall_time_s
        _ = report.error_breakdown
        _ = report.shard_usage
        _ = report.regressions

    def test_best_rejected_excludes_accepted(self):
        report = build_report(self.board)
        best = report.best_rejected
        if best is not None:
            self.assertFalse(best.accepted)
            self.assertIsNotNone(best.mu_hat)

    def test_handles_non_contiguous_reign_numbers(self):
        report = build_report(self.board)
        numbers = [r.reign_number for r in report.reigns if r.reign_number is not None]
        self.assertTrue(numbers)
        # The live chain genuinely skips a reign; nothing may assume otherwise.
        self.assertEqual(len(numbers), len(set(numbers)))
        durations = report.reign_durations()
        self.assertTrue(all(isinstance(d, str) for _, d in durations))

    def test_window_filter_excludes_old_rows(self):
        report = build_report(self.board, since=timedelta(seconds=1))
        self.assertEqual(report.attempts, [])

    def test_empty_history_is_safe(self):
        report = build_report({"history": [], "king_chain": []})
        self.assertEqual(report.attempts, [])
        self.assertIsNone(report.best)
        self.assertIsNone(report.best_rejected)
        self.assertIsNone(report.packaging_failure_rate)
        self.assertIsNone(report.median_wall_time_s)

    def test_missing_keys_are_safe(self):
        report = build_report({})
        self.assertEqual(report.attempts, [])
        self.assertEqual(report.reigns, [])


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.board = json.loads(DASHBOARD.read_text(encoding="utf-8"))
        self.datasets = json.loads(DATASETS.read_text(encoding="utf-8"))

    def test_observation_is_flat_and_json_safe(self):
        record = observation(self.board, self.datasets)
        json.dumps(record)
        self.assertIn("king_digest", record)
        self.assertIn("weight_state", record)
        self.assertIsInstance(record["queue_depth"], int)
        self.assertIsInstance(record["eval_active"], bool)

    def test_no_transitions_against_itself(self):
        record = observation(self.board, self.datasets)
        self.assertEqual(transitions(record, record), [])

    def test_first_observation_reports_nothing(self):
        record = observation(self.board, self.datasets)
        self.assertEqual(transitions(None, record), [])

    def test_new_king_is_announced(self):
        before = observation(self.board, self.datasets)
        after = dict(before)
        after["king_digest"] = "f" * 64
        after["reign_number"] = (before["reign_number"] or 0) + 1
        messages = transitions(before, after)
        self.assertTrue(any("NEW KING" in m for m in messages))

    def test_generation_change_is_announced(self):
        before = observation(self.board, self.datasets)
        after = dict(before, generation="teutonic-III")
        self.assertTrue(any("GENERATION CHANGED" in m for m in transitions(before, after)))

    def test_weight_uptime(self):
        records = [
            {"weight_state": "finalized"},
            {"weight_state": "failed"},
            {"weight_state": "finalized"},
            {"weight_state": "finalized"},
        ]
        self.assertAlmostEqual(weight_uptime(records), 0.75)
        self.assertIsNone(weight_uptime([]))
        self.assertIsNone(weight_uptime([{"other": 1}]))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = Store.open(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_and_load_round_trip(self):
        target = make_target()
        self.store.save_target(target)
        loaded = self.store.load_target(target.snapshot_id)
        self.assertEqual(loaded.king_digest, target.king_digest)

    def test_latest_resolves_to_newest(self):
        first = replace(make_target(), snapshot_id="20260101T000000Z-aaaaaaaa")
        second = replace(make_target(), snapshot_id="20260102T000000Z-bbbbbbbb")
        self.store.save_target(first)
        self.store.save_target(second)
        self.assertEqual(self.store.load_target("latest").snapshot_id, second.snapshot_id)

    def test_existing_snapshot_is_never_overwritten(self):
        target = make_target()
        path = self.store.save_target(target)
        original = path.read_text(encoding="utf-8")
        self.store.save_target(replace(target, king_digest="0" * 64))
        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_missing_target_raises(self):
        with self.assertRaises(TargetNotFoundError):
            self.store.load_target("nope")
        with self.assertRaises(TargetNotFoundError):
            self.store.load_target("latest")

    def test_observations_append_and_filter(self):
        self.store.append_observation({"king_digest": "a"})
        self.store.append_observation({"king_digest": "b"})
        self.assertEqual(len(self.store.read_observations()), 2)
        old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        self.store.append_observation({"ts": old, "king_digest": "c"})
        recent = self.store.read_observations(since=timedelta(hours=1))
        self.assertEqual(len(recent), 2)

    def test_truncated_final_line_is_skipped(self):
        self.store.append_observation({"king_digest": "a"})
        with self.store.observations_path.open("a", encoding="utf-8") as handle:
            handle.write('{"king_digest": "trunc')
        self.assertEqual(len(self.store.read_observations()), 1)


class FakePackaging:
    """Duck-typed stand-in for validate_checkpoint.Report."""

    def __init__(self, would_reject=False, failures=(), fatal=(), skipped=(), determinate=True):
        self.would_reject = would_reject
        self.failures = failures
        self.fatal_failures = fatal
        self.skipped = skipped
        self.determinate = determinate


CLEAN_PACKAGING = FakePackaging()

class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.dashboard, self.datasets = live_pair()
        self.live = Target.from_live(self.dashboard, self.datasets)
        # The fixture is a captured file, so pin freshness open for these tests.
        self.max_age = timedelta(days=3650)

    def test_blocks_when_king_moved(self):
        pinned = replace(self.live, king_digest="0" * 64)
        result = run_preflight(
            pinned, self.live, self.dashboard, offline_lcb=0.9, max_age=self.max_age
        )
        self.assertFalse(result.ok)
        self.assertTrue(any("king digest" in c.name for c in result.blockers))

    def test_blocks_without_offline_lcb(self):
        result = run_preflight(
            self.live, self.live, self.dashboard, offline_lcb=None, max_age=self.max_age
        )
        self.assertFalse(result.ok)
        self.assertTrue(any("offline LCB" in c.name for c in result.blockers))

    def test_blocks_when_margin_too_thin(self):
        delta = self.live.delta or 0.5
        result = run_preflight(
            self.live,
            self.live,
            self.dashboard,
            offline_lcb=delta + 0.001,
            margin=0.02,
            max_age=self.max_age,
        )
        self.assertFalse(result.ok)

    def test_clears_with_healthy_inputs(self):
        delta = self.live.delta or 0.5
        result = run_preflight(
            self.live,
            self.live,
            self.dashboard,
            offline_lcb=delta + 0.25,
            offline_mu=delta + 0.30,
            packaging=CLEAN_PACKAGING,
            max_age=self.max_age,
        )
        self.assertTrue(result.ok, [c.detail for c in result.blockers])

    def test_blocks_without_a_packaging_report(self):
        """Two separate gates were one gate too many: green here must mean
        the artefact is shippable, not merely that the numbers looked good."""
        delta = self.live.delta or 0.5
        result = run_preflight(
            self.live, self.live, self.dashboard,
            offline_lcb=delta + 0.25, offline_mu=delta + 0.30, max_age=self.max_age,
        )
        self.assertFalse(result.ok)
        self.assertTrue(any("packaging validated" in c.name for c in result.blockers))

    def test_a_rejecting_packaging_report_aborts(self):
        delta = self.live.delta or 0.5
        report = FakePackaging(
            would_reject=True,
            failures=("a", "b"),
            fatal=(type("C", (), {"name": "contract files byte-identical"})(),),
        )
        result = run_preflight(
            self.live, self.live, self.dashboard,
            offline_lcb=delta + 0.25, offline_mu=delta + 0.30,
            packaging=report, max_age=self.max_age,
        )
        self.assertFalse(result.ok)
        check = next(c for c in result.checks if c.name == "packaging validated")
        self.assertEqual(check.severity.name, "ABORT")
        self.assertIn("spent the hotkey", check.detail)

    def test_skipped_packaging_rules_warn_but_do_not_block(self):
        delta = self.live.delta or 0.5
        report = FakePackaging(skipped=("a", "b", "c"), determinate=False)
        result = run_preflight(
            self.live, self.live, self.dashboard,
            offline_lcb=delta + 0.25, offline_mu=delta + 0.30,
            packaging=report, max_age=self.max_age,
        )
        self.assertTrue(result.ok)
        self.assertTrue(any("fully determined" in c.name for c in result.warnings))

    def test_thin_mu_warns_but_does_not_block(self):
        """The queue argument is strong but not certain; it should not veto."""
        delta = self.live.delta or 0.5
        result = run_preflight(
            self.live, self.live, self.dashboard,
            offline_lcb=delta + 0.25, offline_mu=0.01,
            packaging=CLEAN_PACKAGING, max_age=self.max_age,
        )
        self.assertTrue(result.ok)
        self.assertTrue(any("mu_hat" in c.name for c in result.warnings))

    def test_stale_dashboard_blocks(self):
        result = run_preflight(
            self.live,
            self.live,
            self.dashboard,
            offline_lcb=1.0,
            max_age=timedelta(seconds=1),
        )
        self.assertFalse(result.ok)
        self.assertTrue(any("freshness" in c.name for c in result.blockers))


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base = [
            "--root",
            str(self.root),
            "--local-dashboard",
            str(DASHBOARD),
            "--local-datasets",
            str(DATASETS),
            "--allow-stale",
        ]

    def tearDown(self):
        self._tmp.cleanup()

    def test_snapshot_then_check_is_fresh(self):
        self.assertEqual(cli.main(["snapshot", *self.base]), 0)
        self.assertEqual(cli.main(["check", *self.base]), 0)

    def test_check_detects_hand_edited_digest(self):
        cli.main(["snapshot", *self.base])
        store = Store.open(self.root)
        snapshot_id = store.list_targets()[-1]
        path = store.targets_dir / f"{snapshot_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["king_digest"] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(cli.main(["check", *self.base]), 1)

    def test_check_detects_hand_edited_generation(self):
        cli.main(["snapshot", *self.base])
        store = Store.open(self.root)
        snapshot_id = store.list_targets()[-1]
        path = store.targets_dir / f"{snapshot_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["generation"] = "teutonic-III-110B"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(cli.main(["check", *self.base]), 2)

    def test_check_without_target_is_usage_error(self):
        self.assertEqual(cli.main(["check", *self.base]), EXIT_USAGE)

    def test_unreachable_source_returns_fetch_failed(self):
        # Pin a valid target first, so the only thing that can fail is the fetch.
        cli.main(["snapshot", *self.base])
        code = cli.main(
            [
                "check",
                "--root",
                str(self.root),
                "--local-dashboard",
                str(self.root / "missing.json"),
                "--local-datasets",
                str(DATASETS),
                "--allow-stale",
            ]
        )
        self.assertEqual(code, EXIT_FETCH_FAILED)

    def test_fetch_failure_leaves_state_intact(self):
        cli.main(["snapshot", *self.base])
        store = Store.open(self.root)
        before = store.list_targets()
        cli.main(
            [
                "check",
                "--root",
                str(self.root),
                "--local-dashboard",
                str(self.root / "missing.json"),
                "--local-datasets",
                str(DATASETS),
                "--allow-stale",
            ]
        )
        self.assertEqual(Store.open(self.root).list_targets(), before)

    def test_status_history_and_targets_run(self):
        cli.main(["snapshot", *self.base])
        self.assertEqual(cli.main(["targets", *self.base]), 0)
        self.assertEqual(cli.main(["history", *self.base, "--shards"]), 0)
        self.assertIn(cli.main(["status", *self.base]), (0, 1, 2))

    def test_watch_once_writes_an_observation(self):
        cli.main(["snapshot", *self.base])
        self.assertEqual(cli.main(["watch", *self.base, "--once"]), 0)
        store = Store.open(self.root)
        self.assertEqual(len(store.read_observations()), 1)

    def test_preflight_blocks_without_lcb(self):
        cli.main(["snapshot", *self.base])
        self.assertEqual(cli.main(["preflight", *self.base]), 1)

    def test_check_json_output_is_parseable(self):
        import contextlib
        import io

        cli.main(["snapshot", *self.base])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            cli.main(["check", *self.base, "--json"])
        payload = json.loads(buffer.getvalue())
        self.assertIn("severity", payload)
        self.assertIn("drifts", payload)


if __name__ == "__main__":
    unittest.main()


class WeightPublicationTests(unittest.TestCase):
    """Publication cycles through in-flight states; only a true failure is bad."""

    def setUp(self):
        self.dashboard, self.datasets = live_pair()
        self.live = Target.from_live(self.dashboard, self.datasets)
        self.max_age = timedelta(days=3650)

    def _run(self, weight_status):
        board = dict(self.dashboard.data, weight_status=weight_status)
        doc = Document(
            data=board,
            source="test",
            fetched_at=self.dashboard.fetched_at,
            reported_at=self.dashboard.reported_at,
        )
        delta = self.live.delta or 0.5
        return run_preflight(
            self.live, self.live, doc,
            offline_lcb=delta + 0.25, offline_mu=delta + 0.30,
            packaging=CLEAN_PACKAGING, max_age=self.max_age,
        )

    def _weight_check(self, result):
        return next(c for c in result.checks if "weight publication" in c.name)

    def test_finalized_is_healthy(self):
        result = self._run({"state": "finalized", "finalized_at": "2026-08-26T21:12:54Z"})
        self.assertTrue(self._weight_check(result).passed)

    def test_in_flight_after_a_success_is_healthy(self):
        result = self._run({"state": "claimed", "finalized_at": "2026-08-26T21:12:54Z"})
        self.assertTrue(self._weight_check(result).passed)

    def test_in_flight_having_never_finalized_is_flagged(self):
        result = self._run({"state": "claimed", "finalized_at": None})
        self.assertFalse(self._weight_check(result).passed)

    def test_failed_is_flagged_but_never_blocks(self):
        result = self._run(
            {
                "state": "failed",
                "error_code": "weight_publication_failed",
                "finalized_at": None,
            }
        )
        check = self._weight_check(result)
        self.assertFalse(check.passed)
        self.assertFalse(check.blocking)
        self.assertTrue(result.ok, "a payout warning must not block a submission")
