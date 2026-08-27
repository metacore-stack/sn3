"""Tests for train_mimo.

The pure pieces -- config, schedule, freezing, balance arithmetic, checkpoint
layout -- run without torch. The loop tests need torch and the architecture
files and skip cleanly without them.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch  # noqa: F401

    TORCH = True
except ImportError:  # pragma: no cover
    TORCH = False

from train_mimo.balance import BalanceStats, LoadBalancer
from train_mimo.checkpoint import (
    LOCKED_FILES,
    SHAPE_KEYS,
    architecture_matches,
    latest_checkpoint,
    prune_checkpoints,
    restore_locked_files,
    verify_locked_files,
)
from train_mimo.config import STAGES, DataConfig, OptimConfig, TrainingConfig
from train_mimo.errors import ConfigError
from train_mimo.optim import apply_freeze, lr_multiplier

try:
    from mimo_adapter.loader import find_arch_directory

    find_arch_directory()
    ARCH = TORCH
except Exception:  # noqa: BLE001
    ARCH = False

SKIP = "needs torch and the architecture files"


# -- config -----------------------------------------------------------------


class ConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        config = TrainingConfig()
        self.assertIn(config.stage, STAGES)
        self.assertEqual(config.tokens_per_step, 2)

    def test_unknown_stage_rejected(self):
        with self.assertRaises(ConfigError):
            TrainingConfig(stage="nonsense")

    def test_degenerate_values_rejected(self):
        with self.assertRaises(ConfigError):
            TrainingConfig(max_steps=0)
        with self.assertRaises(ConfigError):
            TrainingConfig(data=DataConfig(grad_accum=0))
        with self.assertRaises(ConfigError):
            TrainingConfig(data=DataConfig(batch_size=0))
        with self.assertRaises(ConfigError):
            TrainingConfig(optim=OptimConfig(schedule="magic"))

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = TrainingConfig(
                run_name="x",
                stage="experts",
                data=DataConfig(shards=("a.npy",), holdouts=("val-a",)),
            )
            path = config.save(Path(tmp) / "c.json")
            restored = TrainingConfig.load(path)
            self.assertEqual(restored.stage, "experts")
            self.assertEqual(tuple(restored.data.shards), ("a.npy",))
            self.assertEqual(tuple(restored.data.holdouts), ("val-a",))
            self.assertEqual(restored.optim.learning_rate, config.optim.learning_rate)

    def test_missing_file(self):
        with self.assertRaises(ConfigError):
            TrainingConfig.load("/nonexistent/config.json")

    def test_stage_selectors_are_ordered_by_breadth(self):
        sizes = [len(STAGES[s]) for s in ("shared", "shared+router", "experts")]
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(STAGES["all"], ())  # empty means everything


# -- schedule ---------------------------------------------------------------


class ScheduleTests(unittest.TestCase):
    def test_warmup_ramps_from_zero(self):
        config = OptimConfig(warmup_steps=10)
        self.assertAlmostEqual(lr_multiplier(0, 100, config), 0.1)
        self.assertAlmostEqual(lr_multiplier(9, 100, config), 1.0)

    def test_cosine_decays_to_floor(self):
        config = OptimConfig(warmup_steps=0, schedule="cosine", min_lr_ratio=0.1)
        self.assertAlmostEqual(lr_multiplier(0, 100, config), 1.0)
        self.assertAlmostEqual(lr_multiplier(100, 100, config), 0.1, places=6)
        self.assertLess(lr_multiplier(50, 100, config), lr_multiplier(10, 100, config))

    def test_linear_decays_to_floor(self):
        config = OptimConfig(warmup_steps=0, schedule="linear", min_lr_ratio=0.0)
        self.assertAlmostEqual(lr_multiplier(100, 100, config), 0.0)
        self.assertAlmostEqual(lr_multiplier(50, 100, config), 0.5, places=6)

    def test_constant_stays_flat(self):
        config = OptimConfig(warmup_steps=0, schedule="constant")
        self.assertEqual(lr_multiplier(0, 100, config), 1.0)
        self.assertEqual(lr_multiplier(99, 100, config), 1.0)

    def test_never_negative(self):
        for schedule in ("cosine", "linear", "constant"):
            config = OptimConfig(warmup_steps=5, schedule=schedule)
            for step in range(0, 200, 7):
                self.assertGreaterEqual(lr_multiplier(step, 100, config), 0.0)


# -- balance ----------------------------------------------------------------


class FakeGate:
    def __init__(self, n: int):
        import torch

        self.e_score_correction_bias = torch.nn.Parameter(torch.zeros(n))


def recorder_for(counts: dict[int, int], n_experts: int, top_k: int = 2):
    from collections import Counter

    return SimpleNamespace(counts=Counter(counts), n_experts=n_experts, top_k=top_k)


@unittest.skipUnless(TORCH, "needs torch")
class BalanceTests(unittest.TestCase):
    def test_uniform_load_is_imbalance_one(self):
        balancer = LoadBalancer([FakeGate(4)], min_tokens=0)
        stats = balancer.update(recorder_for({0: 5, 1: 5, 2: 5, 3: 5}, 4))
        self.assertAlmostEqual(stats.imbalance, 1.0)
        self.assertEqual(stats.experts_touched, 4)
        self.assertAlmostEqual(stats.coverage, 1.0)

    def test_collapse_shows_as_high_imbalance(self):
        balancer = LoadBalancer([FakeGate(4)], min_tokens=0)
        stats = balancer.update(recorder_for({0: 20}, 4))
        self.assertAlmostEqual(stats.imbalance, 4.0)  # one expert takes everything
        self.assertEqual(stats.experts_touched, 1)

    def test_bias_moves_against_the_load(self):
        gate = FakeGate(4)
        balancer = LoadBalancer([gate], update_rate=0.1, min_tokens=0)
        balancer.update(recorder_for({0: 17, 1: 1, 2: 1, 3: 1}, 4))
        bias = gate.e_score_correction_bias.detach()
        self.assertLess(bias[0].item(), 0.0)  # overused expert pushed down
        self.assertGreater(bias[1].item(), 0.0)  # underused pushed up

    def test_step_size_is_bounded_by_update_rate(self):
        gate = FakeGate(4)
        balancer = LoadBalancer([gate], update_rate=0.05, min_tokens=0)
        balancer.update(recorder_for({0: 999999, 1: 1}, 4))
        self.assertAlmostEqual(
            gate.e_score_correction_bias.detach().abs().max().item(), 0.05
        )

    def test_disabled_balancer_records_but_does_not_act(self):
        gate = FakeGate(4)
        balancer = LoadBalancer([gate], enabled=False, min_tokens=0)
        stats = balancer.update(recorder_for({0: 10, 1: 2}, 4))
        self.assertFalse(stats.applied)
        self.assertGreater(stats.imbalance, 1.0)  # still measured
        self.assertEqual(gate.e_score_correction_bias.detach().abs().max().item(), 0.0)

    def test_min_tokens_suppresses_noisy_updates(self):
        gate = FakeGate(4)
        balancer = LoadBalancer([gate], min_tokens=100)
        stats = balancer.update(recorder_for({0: 3}, 4))
        self.assertFalse(stats.applied)

    def test_empty_recorder_is_safe(self):
        balancer = LoadBalancer([FakeGate(4)], min_tokens=0)
        stats = balancer.update(SimpleNamespace(counts=None, n_experts=0))
        self.assertFalse(stats.applied)
        self.assertEqual(stats.coverage, 0.0)

    def test_bias_state_round_trip(self):
        gate = FakeGate(4)
        balancer = LoadBalancer([gate], update_rate=0.1, min_tokens=0)
        balancer.update(recorder_for({0: 17, 1: 1, 2: 1, 3: 1}, 4))
        state = balancer.bias_state()

        other = FakeGate(4)
        restored = LoadBalancer([other])
        restored.load_bias_state(state)
        self.assertTrue(
            torch.equal(gate.e_score_correction_bias, other.e_score_correction_bias)
        )

    def test_mismatched_bias_state_rejected(self):
        balancer = LoadBalancer([FakeGate(4)])
        with self.assertRaises(ValueError):
            balancer.load_bias_state([[0.0] * 4, [0.0] * 4])
        with self.assertRaises(ValueError):
            balancer.load_bias_state([[0.0] * 8])

    def test_summary_tracks_worst(self):
        balancer = LoadBalancer([FakeGate(4)], min_tokens=0)
        balancer.update(recorder_for({0: 5, 1: 5, 2: 5, 3: 5}, 4))
        balancer.update(recorder_for({0: 20}, 4))
        summary = balancer.summary()
        self.assertAlmostEqual(summary["worst_imbalance"], 4.0)
        self.assertEqual(summary["updates"], 2)


# -- checkpoint layout ------------------------------------------------------


class CheckpointLayoutTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.king = self.root / "king"
        self.king.mkdir()
        for name in LOCKED_FILES:
            (self.king / name).write_text(f"king:{name}", encoding="utf-8")
        (self.king / "config.json").write_text(
            json.dumps({"hidden_size": 3072, "num_hidden_layers": 45, "vocab_size": 152576}),
            encoding="utf-8",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_restore_overwrites_everything_locked(self):
        model_dir = self.root / "ckpt"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("rewritten by save_pretrained")
        restored, missing = restore_locked_files(model_dir, self.king)
        self.assertEqual(set(restored), set(LOCKED_FILES))
        self.assertEqual(missing, ())
        # Every locked file must come back byte-identical, config.json included
        # — that is the whole defence against GenesisContractMismatch.
        for name in LOCKED_FILES:
            self.assertEqual(
                (model_dir / name).read_bytes(), (self.king / name).read_bytes(), name
            )

    def test_missing_king_reports_every_file(self):
        restored, missing = restore_locked_files(self.root / "ckpt", None)
        self.assertEqual(restored, ())
        self.assertEqual(set(missing), set(LOCKED_FILES))

    def test_shape_mismatch_detected(self):
        mini = SimpleNamespace(hidden_size=128, num_hidden_layers=4, vocab_size=256)
        differences = architecture_matches(mini, self.king)
        self.assertTrue(any("hidden_size" in d for d in differences))

    def test_matching_shape_reports_nothing(self):
        real = SimpleNamespace(hidden_size=3072, num_hidden_layers=45, vocab_size=152576)
        self.assertEqual(architecture_matches(real, self.king), [])

    def test_shape_keys_cover_the_decisive_dimensions(self):
        for key in ("hidden_size", "num_hidden_layers", "vocab_size"):
            self.assertIn(key, SHAPE_KEYS)

    def test_verify_locked_files_detects_a_rewrite(self):
        import hashlib

        model_dir = self.root / "ckpt"
        model_dir.mkdir()
        restore_locked_files(model_dir, self.king)
        expected = {
            name: hashlib.sha256((self.king / name).read_bytes()).hexdigest()
            for name in LOCKED_FILES
        }
        self.assertEqual(verify_locked_files(model_dir, expected), [])
        (model_dir / "config.json").write_text("tampered")
        self.assertEqual(verify_locked_files(model_dir, expected), ["config.json"])

    def test_latest_checkpoint_requires_state(self):
        output = self.root / "run"
        (output / "checkpoint-000010").mkdir(parents=True)
        self.assertIsNone(latest_checkpoint(output))
        state = output / "state-000010"
        state.mkdir()
        (state / "training_state.pt").write_bytes(b"x")
        found = latest_checkpoint(output)
        self.assertIsNotNone(found)
        self.assertEqual(found[0].name, "checkpoint-000010")

    def test_latest_checkpoint_picks_the_newest(self):
        output = self.root / "run"
        for step in (5, 20, 100):
            (output / f"checkpoint-{step:06d}").mkdir(parents=True)
            state = output / f"state-{step:06d}"
            state.mkdir()
            (state / "training_state.pt").write_bytes(b"x")
        self.assertEqual(latest_checkpoint(output)[0].name, "checkpoint-000100")

    def test_prune_keeps_the_newest(self):
        output = self.root / "run"
        for step in range(1, 6):
            (output / f"checkpoint-{step:06d}").mkdir(parents=True)
            (output / f"state-{step:06d}").mkdir(parents=True)
        removed = prune_checkpoints(output, keep=2)
        self.assertEqual(len(removed), 3)
        self.assertEqual(len(list(output.glob("checkpoint-*"))), 2)


# -- the loop ---------------------------------------------------------------


@unittest.skipUnless(ARCH, SKIP)
class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mimo_adapter import MiniatureSpec, build_miniature, load_arch, read_reference_config

        cls.arch = load_arch()
        cls.reference = read_reference_config(cls.arch.directory)
        cls.spec = MiniatureSpec()
        cls._build = staticmethod(
            lambda: build_miniature(cls.arch, cls.reference, cls.spec, seed=0)
        )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _trainer(self, **overrides):
        from train_mimo import Trainer
        from train_mimo.sources import synthetic_batches

        model, config = self._build()
        training = TrainingConfig(
            run_name="t",
            max_steps=overrides.pop("max_steps", 4),
            save_every=overrides.pop("save_every", 0),
            log_every=0,
            **overrides,
        )
        batches = synthetic_batches(
            vocab_size=int(config.vocab_size),
            seq_len=32,
            batch_size=training.data.batch_size,
            steps=training.max_steps * training.data.grad_accum + 2,
        )
        return Trainer(
            model=model,
            arch=self.arch,
            config=training,
            batches=batches,
            output_dir=self.root / "run",
        )

    def test_loss_moves_and_checkpoint_is_written(self):
        trainer = self._trainer(max_steps=6)
        result = trainer.train()
        self.assertEqual(result.steps, 6)
        self.assertEqual(len(result.metrics), 6)
        self.assertTrue(all(m.loss > 0 for m in result.metrics))
        self.assertTrue(result.checkpoints)
        self.assertTrue(result.checkpoints[-1].model_dir.is_dir())

    def test_training_state_stays_out_of_the_model_directory(self):
        trainer = self._trainer(max_steps=2)
        result = trainer.train()
        model_dir = result.checkpoints[-1].model_dir
        names = {p.name for p in model_dir.iterdir()}
        # An undeclared object in the uploaded tree is ArtifactIntegrityError.
        self.assertNotIn("training_state.pt", names)
        self.assertNotIn("routing_bias.json", names)
        self.assertNotIn("checkpoint.json", names)
        self.assertTrue((result.checkpoints[-1].state_dir / "training_state.pt").is_file())

    def test_freeze_stage_limits_trainable_parameters(self):
        trainer = self._trainer(stage="shared")
        self.assertLess(trainer.freeze.trainable_fraction, 0.5)
        self.assertTrue(all(".shared_experts." in n for n in trainer.freeze.trainable))

    def test_grad_accumulation_consumes_more_batches(self):
        from train_mimo.config import DataConfig

        trainer = self._trainer(max_steps=3, data=DataConfig(grad_accum=2, batch_size=1))
        result = trainer.train()
        self.assertEqual(result.sequences_seen, 6)

    def test_balancer_runs_every_step(self):
        trainer = self._trainer(max_steps=5)
        trainer.train()
        self.assertEqual(len(trainer.balancer.history), 5)

    def test_resume_restores_step_and_bias(self):
        trainer = self._trainer(max_steps=4, save_every=4)
        trainer.train()
        state_dir = trainer.checkpoints[-1].state_dir
        before = trainer.balancer.bias_state()

        fresh = self._trainer(max_steps=8)
        step = fresh.resume(state_dir)
        self.assertEqual(step, 4)
        self.assertEqual(fresh.balancer.bias_state(), before)

    def test_report_is_written_and_parseable(self):
        trainer = self._trainer(max_steps=2)
        result = trainer.train()
        path = trainer.write_report(result)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("config", payload)
        self.assertIn("metrics", payload["result"])

    def test_patch_is_not_left_active(self):
        from mimo_adapter.patch import is_patched

        trainer = self._trainer(max_steps=2)
        trainer.train()
        self.assertFalse(is_patched(self.arch))


@unittest.skipUnless(ARCH, SKIP)
class CliTests(unittest.TestCase):
    def test_synthetic_run(self):
        from train_mimo import cli

        with tempfile.TemporaryDirectory() as tmp:
            code = cli.main(
                [
                    "train", "--synthetic", "--max-steps", "3", "--log-every", "0",
                    "--run-name", "cli", "--output", str(Path(tmp) / "run"),
                ]
            )
            self.assertEqual(code, 0)

    def test_config_command(self):
        from train_mimo import cli

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            self.assertEqual(cli.main(["config", "--output", str(path)]), 0)
            self.assertTrue(path.is_file())

    def test_inspect_without_checkpoints(self):
        from train_mimo import cli

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(cli.main(["inspect", tmp]), 2)


if __name__ == "__main__":
    unittest.main()
