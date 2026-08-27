"""Tests for evaluate_losses. Requires numpy and the cloned Teutonic repo.

Run with: .venv/bin/python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from evaluate_losses import cli
from evaluate_losses.backends import ReplayBackend
from evaluate_losses.compare import compare
from evaluate_losses.engine import (
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP_B,
    DEFAULT_BOOTSTRAP_SEED,
    ENGINE_FALLBACK_DELTA,
    EngineSpec,
    StatsSpec,
    n_positions,
    reduce_per_token,
)
from evaluate_losses.errors import (
    AlignmentError,
    EngineMismatchError,
    EvaluationError,
    PolicyUnavailableError,
)
from evaluate_losses.lossvec import LossVector, shard_of
from evaluate_losses.parity import tier1_statistics, tier2_sampler
from evaluate_losses.policy import load_policy, paired_bootstrap_verdict

SHARD_A = "finewebedu__CC-MAIN-2019-43__part1__shard_000076.npy"
SHARD_B = "finewebedu__CC-MAIN-2013-20__part0__shard_000000.npy"


def vec(label, refs, losses, **kw) -> LossVector:
    return LossVector(tuple(refs), tuple(losses), model_label=label, **kw)


def paired(n_a=6, n_b=6, gap=0.6, seed=1):
    """A king/challenger pair spanning two shards."""
    rng = random.Random(seed)
    refs, king, chall = [], [], []
    for shard, count in ((SHARD_A, n_a), (SHARD_B, n_b)):
        for i in range(count):
            refs.append(f"{shard}#{i}")
            k = 3.0 + rng.gauss(0, 0.2)
            king.append(k)
            chall.append(k - gap + rng.gauss(0, 0.02))
    return vec("king", refs, king), vec("challenger", refs, chall)


class EngineTests(unittest.TestCase):
    def test_constants_match_the_validator_source(self):
        self.assertEqual(DEFAULT_ALPHA, 0.001)
        self.assertEqual(DEFAULT_BOOTSTRAP_B, 10000)
        self.assertEqual(DEFAULT_BOOTSTRAP_SEED, 0xB007)
        self.assertEqual(ENGINE_FALLBACK_DELTA, 0.0015)

    def test_n_positions_is_seq_len_minus_one(self):
        self.assertEqual(n_positions(2048), 2047)
        self.assertEqual(n_positions(2), 1)
        with self.assertRaises(ValueError):
            n_positions(1)

    def test_reduce_divides_by_2047(self):
        losses = [2.0] * 2047
        self.assertAlmostEqual(reduce_per_token(losses), 2.0)
        # Dividing by seq_len instead of seq_len-1 would give 1.99951…
        self.assertNotAlmostEqual(sum(losses) / 2048, 2.0, places=4)

    def test_reduce_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            reduce_per_token([1.0] * 2048)

    def test_reduce_is_chunk_order_invariant(self):
        rng = random.Random(7)
        losses = [rng.random() for _ in range(2047)]
        whole = reduce_per_token(losses)
        chunked = sum(sum(losses[i : i + 1024]) for i in range(0, 2047, 1024)) / 2047
        self.assertAlmostEqual(whole, chunked, places=12)

    def test_engine_spec_defaults_are_clean(self):
        self.assertEqual(EngineSpec().check(), [])
        self.assertEqual(EngineSpec().n_positions, 2047)

    def test_engine_spec_flags_deviations(self):
        self.assertTrue(EngineSpec(batch_size=8).check())
        self.assertTrue(EngineSpec(attn_implementation="sdpa").check())
        self.assertTrue(EngineSpec(use_cache=True).check())
        with self.assertRaises(EngineMismatchError):
            EngineSpec(batch_size=4).require()


class PolicyTests(unittest.TestCase):
    def test_loads_the_validators_function(self):
        module = load_policy()
        self.assertTrue(hasattr(module, "paired_bootstrap_verdict"))

    def test_missing_repo_raises_with_guidance(self):
        with self.assertRaises(PolicyUnavailableError) as ctx:
            load_policy("/nonexistent/teutonic/repo")
        self.assertIn("could not locate", str(ctx.exception))

    def test_acceptance_is_strict(self):
        result = paired_bootstrap_verdict(
            [1.0] * 64, [0.5] * 64,
            bootstrap_seed=0xB007, n_bootstrap=500, alpha=0.001, delta_threshold=0.5,
        )
        self.assertEqual(result["lcb"], 0.5)
        self.assertFalse(result["accepted"])


class LossVectorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_rejects_malformed_input(self):
        with self.assertRaises(EvaluationError):
            vec("m", ["a#0", "a#1"], [1.0])
        with self.assertRaises(EvaluationError):
            vec("m", [], [])
        with self.assertRaises(EvaluationError):
            vec("m", ["a#0", "a#0"], [1.0, 2.0])
        with self.assertRaises(EvaluationError):
            vec("m", ["a#0"], [float("nan")])
        with self.assertRaises(EvaluationError):
            vec("m", ["a#0"], [float("inf")])

    def test_shard_of(self):
        self.assertEqual(shard_of(f"{SHARD_A}#417"), SHARD_A)

    def test_by_shard_partitions(self):
        king, _ = paired(n_a=4, n_b=3)
        groups = king.by_shard()
        self.assertEqual(sorted(groups), sorted([SHARD_A, SHARD_B]))
        self.assertEqual(len(groups[SHARD_A]), 4)
        self.assertEqual(len(groups[SHARD_B]), 3)

    def test_round_trip(self):
        king, _ = paired()
        path = king.save(self.root / "king.json")
        restored = LossVector.load(path)
        self.assertEqual(restored.refs, king.refs)
        self.assertEqual(restored.losses, king.losses)
        self.assertEqual(restored.engine, king.engine)

    def test_subset_reorders(self):
        king, _ = paired(n_a=4, n_b=0)
        picked = [king.refs[2], king.refs[0]]
        sub = king.subset(picked)
        self.assertEqual(sub.refs, tuple(picked))
        self.assertEqual(sub.losses[0], king.losses[2])

    def test_subset_rejects_unknown_ref(self):
        king, _ = paired()
        with self.assertRaises(AlignmentError):
            king.subset(["missing#0"])

    def test_alignment_accepts_identical_refs(self):
        king, challenger = paired()
        king.assert_aligned(challenger)  # must not raise

    def test_alignment_rejects_length_mismatch(self):
        king, challenger = paired()
        with self.assertRaises(AlignmentError):
            king.assert_aligned(challenger.subset(challenger.refs[:-1]))

    def test_alignment_rejects_reordering(self):
        king, challenger = paired()
        shuffled = challenger.subset(list(reversed(challenger.refs)))
        with self.assertRaises(AlignmentError) as ctx:
            king.assert_aligned(shuffled)
        self.assertIn("first divergence", str(ctx.exception))

    def test_alignment_rejects_different_manifest(self):
        king, challenger = paired()
        a = vec("king", king.refs, king.losses, manifest_sha256="a" * 64)
        b = vec("challenger", challenger.refs, challenger.losses, manifest_sha256="b" * 64)
        with self.assertRaises(AlignmentError):
            a.assert_aligned(b)

    def test_engine_differences_reported(self):
        king, challenger = paired()
        a = vec("k", king.refs, king.losses, engine=EngineSpec().to_dict())
        b = vec("c", challenger.refs, challenger.losses,
                engine=EngineSpec(attn_implementation="sdpa").to_dict())
        self.assertTrue(any("attn_implementation" in d for d in a.engine_differences(b)))


class CompareTests(unittest.TestCase):
    def test_matches_the_policy_function_directly(self):
        king, challenger = paired(n_a=100, n_b=100, gap=0.6)
        stats = StatsSpec()
        result = compare(king, challenger, stats=stats, per_shard=False)
        direct = paired_bootstrap_verdict(
            king.losses, challenger.losses,
            bootstrap_seed=stats.bootstrap_seed, n_bootstrap=stats.n_bootstrap,
            alpha=stats.alpha, delta_threshold=stats.delta_threshold,
        )
        self.assertEqual(result.overall.mu_hat, direct["mu_hat"])
        self.assertEqual(result.overall.lcb, direct["lcb"])
        self.assertEqual(result.overall.accepted, direct["accepted"])

    def test_refuses_misaligned_vectors(self):
        king, challenger = paired()
        reordered = challenger.subset(list(reversed(challenger.refs)))
        with self.assertRaises(AlignmentError):
            compare(king, reordered)

    def test_per_shard_breakdown(self):
        king, challenger = paired(n_a=60, n_b=60)
        result = compare(king, challenger)
        self.assertEqual(len(result.by_shard.shards), 2)
        self.assertIsNotNone(result.by_shard.spread)
        self.assertIsNotNone(result.by_shard.stdev)
        self.assertIsNotNone(result.by_shard.median)

    def test_clearing_fraction_reflects_single_draw_odds(self):
        # One shard clears the bar, one does not.
        refs = [f"{SHARD_A}#{i}" for i in range(40)] + [f"{SHARD_B}#{i}" for i in range(40)]
        king = vec("king", refs, [3.0] * 80)
        chall = vec("challenger", refs, [2.3] * 40 + [2.8] * 40)
        result = compare(king, chall, stats=StatsSpec(delta_threshold=0.5))
        self.assertAlmostEqual(result.by_shard.clearing_fraction(0.5), 0.5)
        self.assertLess(result.by_shard.worst.mu_hat, 0.5)
        self.assertGreater(result.by_shard.best.mu_hat, 0.5)

    def test_bootstrap_penalty_is_small_for_paired_data(self):
        king, challenger = paired(n_a=250, n_b=250, gap=0.6)
        overall = compare(king, challenger, per_shard=False).overall
        self.assertGreater(overall.bootstrap_penalty, 0)
        self.assertLess(overall.bootstrap_penalty / overall.mu_hat, 0.05)

    def test_regression_is_rejected(self):
        refs = [f"{SHARD_A}#{i}" for i in range(50)]
        king = vec("king", refs, [3.0] * 50)
        worse = vec("challenger", refs, [3.4] * 50)
        result = compare(king, worse, per_shard=False)
        self.assertFalse(result.overall.accepted)
        self.assertLess(result.overall.mu_hat, 0)

    def test_single_shard_skipped_when_too_small(self):
        refs = [f"{SHARD_A}#0"]
        king = vec("king", refs, [3.0])
        chall = vec("challenger", refs, [2.0])
        result = compare(king, chall, min_shard_n=2)
        self.assertEqual(result.by_shard.shards, ())


class ReplayBackendTests(unittest.TestCase):
    def test_replays_saved_losses(self):
        king, _ = paired()
        backend = ReplayBackend(king)
        out = backend.score(list(king.refs), model_label="replayed-king")
        self.assertEqual(out.losses, king.losses)
        self.assertEqual(out.model_label, "replayed-king")

    def test_replays_a_subset_in_order(self):
        king, _ = paired(n_a=5, n_b=0)
        backend = ReplayBackend(king)
        picked = [king.refs[3], king.refs[1]]
        out = backend.score(picked, model_label="k")
        self.assertEqual(out.refs, tuple(picked))
        self.assertEqual(out.losses[0], king.losses[3])

    def test_missing_ref_raises(self):
        king, _ = paired()
        with self.assertRaises(EvaluationError):
            ReplayBackend(king).score(["nope#0"], model_label="k")

    def test_loads_from_a_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            king, _ = paired()
            path = king.save(Path(tmp) / "k.json")
            out = ReplayBackend(path).score(list(king.refs), model_label="k")
            self.assertEqual(out.losses, king.losses)


class ParityTests(unittest.TestCase):
    def test_tier1_passes(self):
        report = tier1_statistics()
        self.assertTrue(report.ok, [c.name for c in report.checks if not c.passed])

    def test_tier2_passes(self):
        report = tier2_sampler()
        self.assertTrue(report.ok, [c.name for c in report.checks if not c.passed])


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        king, challenger = paired(n_a=40, n_b=40)
        self.king = king.save(self.root / "king.json")
        self.challenger = challenger.save(self.root / "chal.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_contract(self):
        self.assertEqual(cli.main(["contract"]), 0)

    def test_compare_accepts(self):
        code = cli.main(["compare", "--king", str(self.king),
                         "--challenger", str(self.challenger), "--delta", "0.1"])
        self.assertEqual(code, 0)

    def test_compare_rejects_with_exit_1(self):
        code = cli.main(["compare", "--king", str(self.king),
                         "--challenger", str(self.challenger), "--delta", "5.0"])
        self.assertEqual(code, 1)

    def test_compare_json_is_parseable(self):
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["compare", "--king", str(self.king),
                      "--challenger", str(self.challenger), "--json", "--delta", "0.1"])
        payload = json.loads(buf.getvalue())
        self.assertIn("overall", payload)
        self.assertIn("by_shard", payload)

    def test_compare_misaligned_fails_cleanly(self):
        king = LossVector.load(self.king)
        bad = king.subset(list(reversed(king.refs)))
        path = bad.save(self.root / "bad.json")
        self.assertEqual(
            cli.main(["compare", "--king", str(self.king), "--challenger", str(path)]), 2
        )

    def test_show(self):
        self.assertEqual(cli.main(["show", str(self.king)]), 0)

    def test_parity_offline(self):
        self.assertEqual(cli.main(["parity"]), 0)

    def test_missing_file_is_usage_error(self):
        self.assertEqual(
            cli.main(["compare", "--king", "/nope.json", "--challenger", str(self.challenger)]), 3
        )


if __name__ == "__main__":
    unittest.main()
