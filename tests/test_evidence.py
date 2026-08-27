"""The evidence store, and the re-baselining it exists to make cheap."""

from __future__ import annotations

import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from evaluate_losses.engine import EngineSpec, StatsSpec
from evaluate_losses.errors import EvaluationError
from evaluate_losses.evidence import Cost, EvidenceStore, Standing
from evaluate_losses.lossvec import LossVector

REFS = tuple(f"finewebedu__000-of-100.npy#{i}" for i in range(400))


def vector(offset: float, *, label: str, seed: int = 0, refs=REFS, **kwargs) -> LossVector:
    """Losses around 2.5, shifted by ``offset``. Lower is better."""
    rng = random.Random(seed)
    losses = tuple(2.5 + offset + rng.gauss(0.0, 0.35) for _ in refs)
    payload = {
        "sequence_set": "blend-a",
        "manifest_sha256": "m" * 64,
        "engine": EngineSpec().to_dict(),
    }
    payload.update(kwargs)
    return LossVector(refs=tuple(refs), losses=losses, model_label=label, **payload)


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = EvidenceStore(self.tmp / "evidence")

    def test_recording_writes_a_vector_and_an_index(self):
        entry = self.store.record(vector(0.0, label="king-r9"), run_id="king-r9", kind="king")
        self.assertTrue((self.store.root / entry.vector_path).is_file())
        self.assertTrue(self.store.index_path.is_file())
        self.assertEqual(entry.n, len(REFS))

    def test_no_verdict_is_ever_stored(self):
        """The whole design: conclusions are derived, never persisted."""
        self.store.record(vector(0.0, label="king"), run_id="king", kind="king")
        self.store.record(vector(-0.2, label="a", seed=1), run_id="a")
        blob = self.store.index_path.read_text()
        for word in ("accepted", "mu_hat", "lcb", "verdict"):
            self.assertNotIn(word, blob)

    def test_index_round_trips(self):
        self.store.record(vector(0.0, label="king"), run_id="king", kind="king")
        self.store.record(
            vector(-0.2, label="a", seed=1),
            run_id="a",
            provenance={"stage": "matrices", "steps": 400},
            cost=Cost(gpu_hours=8.0, usd_per_gpu_hour=2.5, n_gpus=8),
        )
        reopened = EvidenceStore(self.store.root).load()
        self.assertEqual({r.run_id for r in reopened.ordered()}, {"king", "a"})
        self.assertEqual(reopened.get("a").provenance["stage"], "matrices")
        self.assertEqual(Cost.from_dict(reopened.get("a").cost).usd, 20.0)

    def test_duplicate_run_id_is_refused_unless_overwriting(self):
        self.store.record(vector(0.0, label="a"), run_id="a")
        with self.assertRaises(EvaluationError):
            self.store.record(vector(0.0, label="a"), run_id="a")
        self.store.record(vector(-0.5, label="a2"), run_id="a", overwrite=True)
        self.assertEqual(self.store.get("a").model_label, "a2")

    def test_forget_removes_the_record_and_optionally_the_vector(self):
        entry = self.store.record(vector(0.0, label="a"), run_id="a")
        path = self.store.root / entry.vector_path
        self.assertTrue(self.store.forget("a", delete_vector=True))
        self.assertFalse(path.exists())
        self.assertFalse(self.store.forget("a"))

    def test_standings_without_a_king_says_so(self):
        self.store.record(vector(-0.2, label="a"), run_id="a")
        with self.assertRaises(EvaluationError) as ctx:
            self.store.standings()
        self.assertIn("no king vector", str(ctx.exception))


class RebaselineTestCase(unittest.TestCase):
    """A king change must cost one scoring run, not N."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = EvidenceStore(self.tmp / "evidence")
        self.store.record(vector(0.0, label="king-r9"), run_id="king-r9", kind="king")
        # Three challengers, increasingly good, each measured once and never again.
        for i, offset in enumerate((-0.05, -0.20, -0.45)):
            self.store.record(
                vector(offset, label=f"run-{i}", seed=100 + i),
                run_id=f"run-{i}",
                cost=Cost(gpu_hours=4.0 * (i + 1), usd_per_gpu_hour=2.0),
            )

    def test_standings_rank_challengers_against_the_king(self):
        rows = self.store.standings()
        self.assertEqual([s.record.run_id for s in rows], ["run-2", "run-1", "run-0"])
        self.assertTrue(all(s.comparison is not None for s in rows))
        self.assertGreater(rows[0].mu_hat, rows[-1].mu_hat)

    def test_a_new_king_re_ranks_everything_without_rescoring_anyone(self):
        before = {s.record.run_id: s.accepted for s in self.store.standings()}
        self.assertTrue(before["run-2"])

        vector_files = sorted(p.name for p in (self.store.root / "vectors").iterdir())

        # The throne turns over to something far stronger.
        self.store.record(
            vector(-0.40, label="king-r10", seed=7), run_id="king-r10", kind="king"
        )
        after = {s.record.run_id: s.accepted for s in self.store.standings()}

        # Every challenger was re-judged, and none of them was re-run: the only
        # new file is the new king's vector.
        now = sorted(p.name for p in (self.store.root / "vectors").iterdir())
        self.assertEqual(set(now) - set(vector_files), {"king-r10.json"})

        self.assertFalse(after["run-0"])
        self.assertFalse(after["run-1"])
        self.assertNotEqual(before, after)

    def test_an_explicit_king_can_be_chosen(self):
        self.store.record(
            vector(-0.40, label="king-r10", seed=7), run_id="king-r10", kind="king"
        )
        old = self.store.standings(king_run_id="king-r9")
        new = self.store.standings(king_run_id="king-r10")
        self.assertGreater(old[0].mu_hat, new[0].mu_hat)

    def test_latest_king_is_the_most_recent_one(self):
        self.store.record(
            vector(-0.40, label="king-r10", seed=7), run_id="king-r10", kind="king"
        )
        self.assertEqual(self.store.latest_king().run_id, "king-r10")

    def test_leaderboard_is_serialisable_and_counts_acceptances(self):
        board = self.store.leaderboard()
        json.dumps(board)
        self.assertEqual(board["king"]["run_id"], "king-r9")
        self.assertEqual(board["challengers"], 3)
        self.assertEqual(board["unrankable"], 0)
        self.assertEqual(
            board["accepted"], sum(1 for r in board["rows"] if r["accepted"])
        )

    def test_margin_reports_distance_from_the_threshold(self):
        rows = {s.record.run_id: s for s in self.store.standings()}
        best = rows["run-2"]
        self.assertAlmostEqual(
            best.margin, best.comparison.overall.lcb - best.comparison.overall.delta, places=9
        )
        # The live bar is 0.1 and acceptance is strict.
        self.assertEqual(best.comparison.overall.delta, StatsSpec().delta_threshold)
        self.assertEqual(best.accepted, best.comparison.overall.lcb > 0.1)

    def test_spend_reports_dollars_per_nat(self):
        spend = self.store.spend()
        self.assertEqual(spend["runs"], 3)
        self.assertEqual(spend["gpu_hours"], 24.0)
        self.assertEqual(spend["usd"], 48.0)
        self.assertGreater(spend["best_mu_hat"], 0)
        self.assertAlmostEqual(
            spend["usd_per_nat"], round(48.0 / spend["best_mu_hat"], 2), places=2
        )


class ComparabilityTestCase(unittest.TestCase):
    """Refusing to pair vectors that are not measuring the same thing."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = EvidenceStore(self.tmp / "evidence")
        self.store.record(vector(0.0, label="king"), run_id="king", kind="king")

    def test_a_different_holdout_is_unrankable_not_silently_wrong(self):
        other = tuple(f"dclm-baseline-1.0__001-of-100.npy#{i}" for i in range(400))
        self.store.record(
            vector(-0.5, label="elsewhere", refs=other, sequence_set="val-b"),
            run_id="elsewhere",
        )
        row = self.store.standings()[0]
        self.assertIsNone(row.comparison)
        self.assertIn("different holdouts", row.reason)
        self.assertFalse(row.accepted)

    def test_a_changed_corpus_manifest_blocks_the_pairing(self):
        self.store.record(
            vector(-0.5, label="restated", manifest_sha256="n" * 64), run_id="restated"
        )
        row = self.store.standings()[0]
        self.assertIsNone(row.comparison)
        self.assertIn("manifests differ", row.reason)

    def test_a_different_engine_shape_blocks_the_pairing(self):
        spec = EngineSpec(lm_head_chunk=512).to_dict()
        self.store.record(vector(-0.5, label="other-engine", engine=spec), run_id="other")
        row = self.store.standings()[0]
        self.assertIsNone(row.comparison)
        self.assertIn("engine shape", row.reason)

    def test_unrankable_rows_sort_last(self):
        self.store.record(vector(-0.1, label="ok", seed=3), run_id="ok")
        self.store.record(
            vector(-0.9, label="bad-basis", sequence_set="val-b"), run_id="bad"
        )
        rows = self.store.standings()
        self.assertEqual(rows[0].record.run_id, "ok")
        self.assertEqual(rows[-1].record.run_id, "bad")


if __name__ == "__main__":
    unittest.main()
