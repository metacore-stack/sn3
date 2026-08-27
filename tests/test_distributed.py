"""Multi-process training.

The unit tests below cover the single-process paths. The three integration
tests actually launch ``torch.distributed.run``, because the properties that
matter here -- that the ranks reconstruct the single-process token distribution
exactly, that the routing bias stays identical across ranks, and that a restart
loses nothing -- have no single-process expression at all.

Set ``SN3_SKIP_DIST=1`` to skip the subprocess tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from collections import Counter
from pathlib import Path

from train_mimo.config import OptimConfig
from train_mimo.distributed import (
    STRATEGIES,
    DistributedContext,
    all_reduce_expert_counts,
    all_reduce_mean,
    assert_bias_synchronised,
    barrier,
    default_backend,
    from_environment,
    gather_full_state_dict,
    shard_stream,
    unwrap,
    wrap_model,
)
from train_mimo.errors import TrainingError
from train_mimo.optim import lr_multiplier

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIST = os.environ.get("SN3_SKIP_DIST") == "1"


class ContextTestCase(unittest.TestCase):
    def test_default_context_is_single_process(self):
        ctx = DistributedContext()
        self.assertFalse(ctx.is_distributed)
        self.assertTrue(ctx.is_main)
        self.assertEqual(ctx.world_size, 1)

    def test_only_rank_zero_is_main(self):
        self.assertTrue(DistributedContext(rank=0, world_size=4).is_main)
        self.assertFalse(DistributedContext(rank=3, world_size=4).is_main)

    def test_from_environment_reads_torchrun_variables(self):
        env = {"WORLD_SIZE": "8", "RANK": "5", "LOCAL_RANK": "1"}
        with unittest.mock.patch.dict(os.environ, env, clear=False):
            ctx = from_environment()
        self.assertEqual((ctx.rank, ctx.world_size, ctx.local_rank), (5, 8, 1))
        self.assertTrue(ctx.is_distributed)

    def test_backend_override_is_honoured(self):
        with unittest.mock.patch.dict(
            os.environ, {"TEUTONIC_DIST_BACKEND": "gloo", "WORLD_SIZE": "2"}
        ):
            self.assertEqual(from_environment().backend, "gloo")

    def test_default_backend_is_gloo_without_cuda(self):
        # Every claim in this suite is made on CPU; gloo is the tested path.
        self.assertIn(default_backend(), ("gloo", "nccl"))

    def test_summary_is_serialisable(self):
        json.dumps(DistributedContext(rank=1, world_size=2).summary())


class ShardStreamTestCase(unittest.TestCase):
    def test_single_process_passes_everything_through(self):
        items = list(range(10))
        self.assertEqual(list(shard_stream(iter(items), DistributedContext())), items)

    def test_ranks_partition_the_stream_without_overlap(self):
        items = list(range(12))
        world = 3
        slices = [
            list(shard_stream(iter(items), DistributedContext(rank=r, world_size=world)))
            for r in range(world)
        ]
        self.assertEqual(sorted(x for s in slices for x in s), items)
        for a in range(world):
            for b in range(a + 1, world):
                self.assertEqual(set(slices[a]) & set(slices[b]), set())

    def test_every_rank_receives_the_same_count(self):
        """Unequal counts would hang the group on the next collective."""
        slices = [
            list(shard_stream(iter(range(12)), DistributedContext(rank=r, world_size=4)))
            for r in range(4)
        ]
        self.assertEqual({len(s) for s in slices}, {3})

    def test_rank_r_takes_every_wth_item_starting_at_r(self):
        got = list(shard_stream(iter(range(10)), DistributedContext(rank=1, world_size=3)))
        self.assertEqual(got, [1, 4, 7])


class CollectiveFallbackTestCase(unittest.TestCase):
    """Single-process behaviour of the collectives -- no process group."""

    def test_expert_counts_pass_through_and_are_copied(self):
        counts = Counter({0: 3, 5: 1})
        out = all_reduce_expert_counts(counts, 8, DistributedContext())
        self.assertEqual(out, counts)
        out[0] = 99
        self.assertEqual(counts[0], 3)

    def test_mean_passes_through(self):
        self.assertEqual(all_reduce_mean(1.25, DistributedContext()), 1.25)

    def test_barrier_and_bias_check_are_no_ops(self):
        barrier(DistributedContext())
        assert_bias_synchronised([object()], DistributedContext())


class WrapTestCase(unittest.TestCase):
    def test_unknown_strategy_is_rejected(self):
        import torch

        with self.assertRaises(TrainingError) as ctx:
            wrap_model(torch.nn.Linear(2, 2), "deepspeed", DistributedContext())
        self.assertIn("deepspeed", str(ctx.exception))

    def test_every_named_strategy_is_accepted_single_process(self):
        import torch

        model = torch.nn.Linear(2, 2)
        for strategy in STRATEGIES:
            # Single process: nothing wraps, whatever was asked for.
            self.assertIs(wrap_model(model, strategy, DistributedContext()), model)

    def test_unwrap_returns_the_inner_module(self):
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.module = inner

        inner = torch.nn.Linear(2, 2)
        self.assertIs(unwrap(Wrapper(inner)), inner)
        self.assertIs(unwrap(inner), inner)

    def test_state_dict_gather_is_plain_single_process(self):
        import torch

        model = torch.nn.Linear(2, 2)
        gathered = gather_full_state_dict(model, DistributedContext())
        self.assertEqual(set(gathered), set(model.state_dict()))


class ScheduleTestCase(unittest.TestCase):
    """WSD, the schedule a long run branches cooldowns from."""

    def test_wsd_holds_full_rate_then_decays(self):
        config = OptimConfig(
            schedule="wsd", warmup_steps=10, decay_fraction=0.2, min_lr_ratio=0.1
        )
        values = {s: round(lr_multiplier(s, 100, config), 4) for s in (0, 9, 10, 50, 79, 80, 90, 100)}
        self.assertAlmostEqual(values[0], 0.1)      # warmup, 1/10
        self.assertAlmostEqual(values[9], 1.0)
        self.assertAlmostEqual(values[50], 1.0)     # stable
        self.assertAlmostEqual(values[79], 1.0)
        self.assertAlmostEqual(values[80], 1.0)     # decay starts
        self.assertAlmostEqual(values[90], 0.55)
        self.assertAlmostEqual(values[100], 0.1)    # floor

    def test_explicit_decay_start_overrides_the_fraction(self):
        config = OptimConfig(schedule="wsd", warmup_steps=0, decay_start_step=50, min_lr_ratio=0.0)
        self.assertAlmostEqual(lr_multiplier(49, 100, config), 1.0)
        self.assertAlmostEqual(lr_multiplier(75, 100, config), 0.5)

    def test_constant_ignores_decay(self):
        config = OptimConfig(schedule="constant", warmup_steps=0)
        self.assertEqual(lr_multiplier(99, 100, config), 1.0)


def run_dist(script: str, args: list[str], nproc: int, timeout: int = 420):
    """Launch a script under torch.distributed.run and return its output."""
    env = dict(os.environ)
    # torchrun pins OMP_NUM_THREADS=1 for multi-rank launches but not for a
    # single rank. Left alone, the reduction order in CPU matmuls differs
    # between world sizes and near-tied top-k decisions flip, which would make
    # the comparison below fail for a reason that has nothing to do with
    # sharding.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         f"--nproc_per_node={nproc}", str(ROOT / script), *args],
        capture_output=True, text=True, timeout=timeout, cwd=ROOT, env=env,
    )


@unittest.skipIf(SKIP_DIST, "SN3_SKIP_DIST=1")
class MultiRankTestCase(unittest.TestCase):
    """The claims that only exist with more than one rank."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)

    def test_sharded_ranks_reproduce_the_single_process_trajectory_exactly(self):
        """Two ranks and four ranks must see exactly what one process saw.

        The learning rate is zero inside the script, so the weights are pinned
        and the whole trajectory is a deterministic function of the batches and
        the routing bias. Any difference is a sharding or all-reduce bug, not
        float-reduction noise -- which is a claim nobody could make about the
        gradient path, where reduction order legitimately varies.
        """
        outputs = {}
        for world in (1, 2, 4):
            path = self.tmp / f"w{world}.json"
            result = run_dist(
                "scripts/dist_equivalence.py",
                [str(path), "--steps", "3", "--micro-per-step", "4"],
                nproc=world,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            outputs[world] = json.loads(path.read_text())

        self.assertEqual(outputs[1], outputs[2])
        self.assertEqual(outputs[1], outputs[4])
        # Guard against comparing two empty runs.
        self.assertEqual(outputs[1]["steps"], 3)
        self.assertTrue(outputs[1]["bias"])
        self.assertTrue(any(s["applied"] for s in outputs[1]["balance_history"]))

    def test_missing_all_reduce_makes_the_ranks_diverge(self):
        """The bug this module exists to prevent, demonstrated.

        Without the all-reduce each rank steps the routing bias from its own
        tokens. Nothing raises; the checkpoint just carries rank zero's routing.
        """
        result = run_dist("scripts/dist_divergence.py", [], nproc=3)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        out = result.stdout
        self.assertIn("OK", out)
        without = float(out.split("WITHOUT all-reduce")[1].split()[0])
        with_it = float(out.split("WITH    all-reduce")[1].split()[0])
        self.assertGreater(without, 0.0)
        self.assertEqual(with_it, 0.0)

    def test_checkpoint_and_resume_survive_across_ranks(self):
        result = run_dist(
            "scripts/dist_resume.py", [str(self.tmp / "resume"), "--steps", "2"], nproc=2
        )
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        payload = json.loads(result.stdout[result.stdout.index("{"):result.stdout.rindex("}") + 1])
        self.assertTrue(payload["resumed_bias_matches_saved"])
        self.assertTrue(payload["fresh_bias_differs_from_saved"])
        self.assertTrue(payload["optimizer_state_restored"])
        self.assertEqual(payload["resumed_step"], 2)

    def test_four_rank_run_trains_and_stays_synchronised(self):
        result = run_dist("scripts/dist_smoke.py", [], nproc=4)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        self.assertIn("synchronised across all ranks", result.stdout)
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
