"""The scoring entry point: checkpoint plus holdout, in; loss vector, out.

The planner is the part worth testing hard. Its job is to refuse before a GPU
hour is spent, and the failure it exists to catch is the quiet one -- a holdout
that looks better because it is larger, while measuring 22% of what the
validator scores.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from evaluate_losses.backends import manifest_digest
from evaluate_losses.errors import EvaluationError
from evaluate_losses.scoring import (
    MIN_USEFUL_SEQUENCES,
    ScoringPlan,
    load_holdout,
    plan,
    score_checkpoint,
)

STATE = Path.home() / "Documents" / "sn3" / "state"
HAS_STATE = (STATE / "dataset-config.json").is_file()


class PlanShapeTestCase(unittest.TestCase):
    """Arithmetic on a hand-built plan, independent of any local state."""

    def make(self, per_corpus, expected) -> ScoringPlan:
        return ScoringPlan(
            model_dir=Path("/m"),
            holdout_name="h",
            sequences=sum(per_corpus.values()),
            per_corpus=per_corpus,
            expected_share=expected,
        )

    def test_a_matching_mixture_has_no_error(self):
        p = self.make(
            {"finewebedu": 440, "automathtext-v2": 520, "dclm-baseline-1.0": 1040},
            {"finewebedu": 0.22, "automathtext-v2": 0.26, "dclm-baseline-1.0": 0.52},
        )
        self.assertAlmostEqual(p.max_share_error, 0.0, places=9)

    def test_a_single_corpus_holdout_is_off_by_the_missing_share(self):
        p = self.make(
            {"finewebedu": 5120},
            {"finewebedu": 0.22, "automathtext-v2": 0.26, "dclm-baseline-1.0": 0.52},
        )
        self.assertAlmostEqual(p.max_share_error, 0.78, places=9)

    def test_an_empty_plan_does_not_divide_by_zero(self):
        p = self.make({}, {"finewebedu": 1.0})
        self.assertEqual(p.actual_share, {})
        self.assertEqual(p.max_share_error, 0.0)

    def test_summary_is_serialisable(self):
        p = self.make({"finewebedu": 10}, {"finewebedu": 1.0})
        json.dumps(p.to_dict())


@unittest.skipUnless(HAS_STATE, "no local corpus state")
class PlanAgainstRealStateTestCase(unittest.TestCase):
    def test_the_blended_holdout_matches_the_validator_exactly(self):
        result = plan("/nonexistent", "blend-a", state_root=STATE)
        self.assertEqual(result.sequences, 2000)
        # Largest-remainder apportionment of 2000 across 22/26/52.
        self.assertEqual(
            result.per_corpus,
            {"automathtext-v2": 520, "dclm-baseline-1.0": 1040, "finewebedu": 440},
        )
        self.assertAlmostEqual(result.max_share_error, 0.0, places=9)
        self.assertEqual(result.warnings, ())

    def test_a_finewebedu_only_holdout_is_flagged_despite_being_larger(self):
        """The trap: more sequences, less contract."""
        blend = plan("/nonexistent", "blend-a", state_root=STATE)
        single = plan("/nonexistent", "val-a", state_root=STATE)
        self.assertGreater(single.sequences, blend.sequences)
        self.assertTrue(single.warnings)
        self.assertTrue(any("mixture is off" in w for w in single.warnings))
        self.assertTrue(any("covers only part" in w for w in single.warnings))

    def test_a_missing_holdout_names_the_path_and_the_fix(self):
        with self.assertRaises(EvaluationError) as ctx:
            plan("/nonexistent", "no-such-holdout", state_root=STATE)
        message = str(ctx.exception)
        self.assertIn("no-such-holdout", message)
        self.assertIn("fineweb holdout build", message)

    def test_loading_a_holdout_returns_its_refs(self):
        holdout = load_holdout("blend-a", state_root=STATE)
        self.assertEqual(len(holdout.refs), 2000)


class ScoreGuardTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_missing_model_directory_is_refused_before_anything_is_opened(self):
        with self.assertRaises(EvaluationError) as ctx:
            score_checkpoint(self.tmp / "nope", "blend-a", state_root=STATE)
        self.assertIn("not a directory", str(ctx.exception))

    def test_min_useful_sequences_is_a_stated_number(self):
        self.assertEqual(MIN_USEFUL_SEQUENCES, 200)


class ManifestDigestTestCase(unittest.TestCase):
    """A loss vector must record which data produced it, blend or not."""

    def test_a_blended_loader_reports_one_digest_for_the_whole_blend(self):
        class Corpus:
            def __init__(self, name, digest):
                self.name = name
                self.manifest = type("M", (), {"digest": digest})()

        class Blend:
            corpora = [Corpus("a", "1" * 64), Corpus("b", "2" * 64)]

        from fineweb_loader.loader import BlendedLoader

        loader = BlendedLoader.__new__(BlendedLoader)
        loader.corpora = Blend.corpora
        digest = manifest_digest(loader)
        self.assertEqual(len(digest), 64)

        # Changing any source changes the blend's identity.
        loader.corpora = [Corpus("a", "1" * 64), Corpus("b", "3" * 64)]
        self.assertNotEqual(manifest_digest(loader), digest)

    def test_a_single_corpus_loader_still_reports_its_own_digest(self):
        loader = type("L", (), {"manifest": type("M", (), {"digest": "abc"})()})()
        self.assertEqual(manifest_digest(loader), "abc")

    def test_a_loader_with_neither_does_not_raise(self):
        self.assertEqual(manifest_digest(object()), "")


if __name__ == "__main__":
    unittest.main()
