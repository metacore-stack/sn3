"""Tests for mimo_adapter.

These need torch, transformers and the two architecture files. Fetch them with:

    .venv/bin/python -m mimo_adapter fetch --king-digest <digest>

Everything is skipped cleanly when they are absent.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401

    TORCH = True
except ImportError:  # pragma: no cover
    TORCH = False

from mimo_adapter.errors import AdapterError, ArchSourceError
from mimo_adapter.loader import find_arch_directory, load_arch, read_reference_config
from mimo_adapter.miniature import (
    PER_LAYER_KEYS,
    ROUTING_KEYS,
    MiniatureSpec,
    build_miniature,
    count_parameters,
    initialize_attention_sinks,
    initialize_gates,
    uninitialized_parameters,
    miniature_config_dict,
)
from mimo_adapter.patch import (
    RoutingRecorder,
    gates,
    is_patched,
    set_gates_eval,
    trainable_routing,
)
from mimo_adapter.verify import group_of, run_all

REFERENCE = {
    "num_hidden_layers": 45,
    "hidden_size": 3072,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "n_group": 1,
    "topk_group": 1,
    "routed_scaling_factor": None,
    "moe_layer_freq": [0] + [1] * 44,
    "hybrid_layer_pattern": [0, 1] * 22 + [0],
    "auto_map": {"AutoConfig": "x"},
    "architectures": ["MiMoV2ForCausalLM"],
    "transformers_version": "5.8.1",
    "initializer_range": 0.02,
}


def arch_available() -> bool:
    try:
        find_arch_directory()
    except ArchSourceError:
        return False
    return True


ARCH = TORCH and arch_available()
SKIP = "needs torch, transformers and the architecture files"


# -- config shrinking (no torch needed) -------------------------------------


class MiniatureConfigTests(unittest.TestCase):
    def test_routing_keys_survive_shrinking(self):
        payload = miniature_config_dict(REFERENCE, MiniatureSpec())
        for key in ROUTING_KEYS:
            self.assertEqual(payload[key], REFERENCE[key], key)

    def test_sizes_are_overridden(self):
        spec = MiniatureSpec(num_hidden_layers=4, hidden_size=128, n_routed_experts=8)
        payload = miniature_config_dict(REFERENCE, spec)
        self.assertEqual(payload["num_hidden_layers"], 4)
        self.assertEqual(payload["hidden_size"], 128)
        self.assertEqual(payload["n_routed_experts"], 8)

    def test_per_layer_lists_are_truncated_to_layer_count(self):
        payload = miniature_config_dict(REFERENCE, MiniatureSpec(num_hidden_layers=4))
        for key in PER_LAYER_KEYS:
            if key in REFERENCE:
                self.assertEqual(len(payload[key]), 4, key)

    def test_per_layer_lists_are_extended_when_too_short(self):
        reference = dict(REFERENCE, moe_layer_freq=[0, 1])
        payload = miniature_config_dict(reference, MiniatureSpec(num_hidden_layers=6))
        self.assertEqual(payload["moe_layer_freq"], [0, 1, 1, 1, 1, 1])

    def test_dense_first_layer_is_preserved(self):
        payload = miniature_config_dict(REFERENCE, MiniatureSpec(num_hidden_layers=4))
        self.assertEqual(payload["moe_layer_freq"][0], 0)
        self.assertTrue(all(v == 1 for v in payload["moe_layer_freq"][1:]))

    def test_checkpoint_only_fields_are_dropped(self):
        payload = miniature_config_dict(REFERENCE, MiniatureSpec())
        for key in ("auto_map", "architectures", "transformers_version", "dtype"):
            self.assertNotIn(key, payload)

    def test_use_cache_disabled(self):
        self.assertFalse(miniature_config_dict(REFERENCE, MiniatureSpec())["use_cache"])


class GroupingTests(unittest.TestCase):
    def test_parameter_grouping(self):
        cases = {
            "model.layers.1.mlp.gate.weight": "router",
            "model.layers.1.mlp.gate.e_score_correction_bias": "router_bias",
            "model.layers.1.mlp.experts.3.up_proj.weight": "routed_experts",
            "model.layers.1.mlp.shared_experts.up_proj.weight": "shared_expert",
            "model.layers.1.self_attn.q_proj.weight": "attention",
            "lm_head.weight": "lm_head",
            "model.embed_tokens.weight": "embedding",
        }
        for name, expected in cases.items():
            self.assertEqual(group_of(name), expected, name)


class LoaderTests(unittest.TestCase):
    def test_missing_directory_raises(self):
        with self.assertRaises(ArchSourceError):
            load_arch("/nonexistent/arch/dir")

    def test_bad_digest_rejected(self):
        from mimo_adapter.loader import fetch_arch

        with self.assertRaises(ArchSourceError):
            fetch_arch("tooshort")


# -- the real architecture --------------------------------------------------


@unittest.skipUnless(ARCH, SKIP)
class ArchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arch = load_arch()
        cls.reference = read_reference_config(cls.arch.directory)

    def _model(self, **kw):
        return build_miniature(self.arch, self.reference, MiniatureSpec(), **kw)

    def test_reference_config_is_the_real_one(self):
        self.assertEqual(self.reference["topk_method"], "noaux_tc")
        self.assertEqual(self.reference["n_group"], 1)
        self.assertEqual(self.reference["n_routed_experts"], 256)

    def test_miniature_builds_and_is_small(self):
        model, config = self._model()
        self.assertLess(count_parameters(model), 20_000_000)
        self.assertEqual(config.topk_method, "noaux_tc")
        self.assertEqual(len(gates(model)), config.num_hidden_layers - 1)

    def test_gates_are_initialised_sanely(self):
        model, config = self._model()
        for gate in gates(model):
            weight = gate.weight.detach()
            self.assertTrue(torch.isfinite(weight).all())
            # The architecture allocates with torch.empty(); without explicit
            # initialisation these reach ~1e38 and saturate the sigmoid.
            self.assertLess(weight.abs().max().item(), 1.0)
            self.assertEqual(gate.e_score_correction_bias.abs().max().item(), 0.0)

    def test_initialize_gates_is_deterministic_and_bounded(self):
        # Uninitialised values come from torch.empty(), so what they contain is
        # by definition unpredictable — sometimes 1e38, sometimes recycled and
        # tame. Only the initialised state is worth asserting on.
        model, config = self._model(initialize=False)
        count = initialize_gates(model, config, seed=3)
        self.assertEqual(count, len(gates(model)))
        std = float(getattr(config, "initializer_range", 0.02))
        for gate in gates(model):
            weight = gate.weight.detach()
            self.assertTrue(torch.isfinite(weight).all())
            self.assertLess(weight.abs().max().item(), 10 * std)
            self.assertEqual(gate.e_score_correction_bias.abs().max().item(), 0.0)

        again, _ = self._model(initialize=False)
        initialize_gates(again, config, seed=3)
        for a, b in zip(gates(model), gates(again)):
            self.assertTrue(torch.equal(a.weight, b.weight))

    def test_no_parameter_is_left_uninitialised(self):
        """The architecture allocates three kinds of bare Parameter and fills none.

        An unwritten parameter is read as whatever was in that memory, so two
        processes with the same seed disagree and every local comparison carries
        noise it did not earn. Measured on this miniature before the fix: 0.0096
        nats between identical runs, a tenth of the live delta threshold.
        """
        self.assertEqual(
            uninitialized_parameters(self.arch, self.reference, MiniatureSpec(),
                                     initialize=True),
            [],
        )

    def test_the_detector_actually_detects(self):
        """A regression test that cannot pass vacuously."""
        missed = uninitialized_parameters(
            self.arch, self.reference, MiniatureSpec(), initialize=False
        )
        self.assertTrue(missed)
        self.assertTrue(any("gate.weight" in n for n in missed))
        self.assertTrue(any("attention_sink_bias" in n for n in missed))

    def test_build_is_reproducible_from_a_seed(self):
        a, _ = self._model(seed=7)
        b, _ = self._model(seed=7)
        sa, sb = a.state_dict(), b.state_dict()
        self.assertEqual(set(sa), set(sb))
        differing = [n for n in sa if not torch.equal(sa[n], sb[n])]
        self.assertEqual(differing, [])

    def test_attention_sinks_are_zeroed(self):
        model, _ = self._model()
        sinks = [
            m.attention_sink_bias
            for m in model.modules()
            if getattr(m, "attention_sink_bias", None) is not None
        ]
        self.assertTrue(sinks, "the miniature should have SWA layers with sinks")
        for sink in sinks:
            self.assertEqual(sink.abs().max().item(), 0.0)
            # Frozen in the shipped architecture, and the reign 6->7 diff shows
            # the king's operator left every 1-D vector frozen too.
            self.assertFalse(sink.requires_grad)

    def test_initialize_attention_sinks_reports_what_it_touched(self):
        model, _ = self._model(initialize=False)
        touched = initialize_attention_sinks(model)
        self.assertGreater(touched, 0)
        self.assertEqual(touched, sum(
            1 for m in model.modules()
            if getattr(m, "attention_sink_bias", None) is not None
        ))

    def test_shipped_gate_refuses_training(self):
        model, config = self._model()
        model.train()
        ids = torch.randint(0, config.vocab_size, (1, 8))
        with self.assertRaises(ValueError) as ctx:
            model(input_ids=ids)
        self.assertIn("noaux_tc", str(ctx.exception))

    def test_patched_gate_is_numerically_identical_in_eval(self):
        model, config = self._model()
        model.eval()
        ids = torch.randint(0, config.vocab_size, (2, 12))
        with torch.no_grad():
            baseline = model(input_ids=ids).logits.clone()
        with trainable_routing(self.arch):
            with torch.no_grad():
                patched = model(input_ids=ids).logits.clone()
        self.assertEqual((baseline - patched).abs().max().item(), 0.0)

    def test_patch_permits_backward(self):
        model, config = self._model()
        model.train()
        with trainable_routing(self.arch):
            ids = torch.randint(0, config.vocab_size, (2, 12))
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for gate in gates(model):
            self.assertIsNotNone(gate.weight.grad)
            self.assertGreater(gate.weight.grad.norm().item(), 0.0)

    def test_router_bias_receives_no_gradient(self):
        model, config = self._model()
        model.train()
        with trainable_routing(self.arch):
            ids = torch.randint(0, config.vocab_size, (2, 12))
            model(input_ids=ids, labels=ids).loss.backward()
        for gate in gates(model):
            grad = gate.e_score_correction_bias.grad
            self.assertTrue(grad is None or grad.abs().sum().item() == 0.0)

    def test_patch_restores_forward_even_on_exception(self):
        original = self.arch.gate_cls.forward
        with self.assertRaises(RuntimeError):
            with trainable_routing(self.arch):
                raise RuntimeError("boom")
        self.assertIs(self.arch.gate_cls.forward, original)
        self.assertFalse(is_patched(self.arch))

    def test_nested_patch_refused(self):
        with trainable_routing(self.arch):
            self.assertTrue(is_patched(self.arch))
            with self.assertRaises(AdapterError):
                with trainable_routing(self.arch):
                    pass
        self.assertFalse(is_patched(self.arch))

    def test_recorder_collects_routing(self):
        model, config = self._model()
        recorder = RoutingRecorder()
        model.eval()
        with trainable_routing(self.arch, recorder=recorder):
            with torch.no_grad():
                model(input_ids=torch.randint(0, config.vocab_size, (2, 16)))
        self.assertEqual(recorder.n_experts, config.n_routed_experts)
        self.assertEqual(recorder.top_k, config.num_experts_per_tok)
        self.assertGreater(recorder.experts_touched, 0)
        self.assertLessEqual(recorder.coverage, 1.0)
        self.assertIn("imbalance", recorder.summary())

    def test_gates_to_eval_is_an_alternative(self):
        model, config = self._model()
        model.train()
        n = set_gates_eval(model)
        self.assertEqual(n, len(gates(model)))
        ids = torch.randint(0, config.vocab_size, (1, 8))
        model(input_ids=ids, labels=ids).loss.backward()  # must not raise

    def test_full_verification_passes(self):
        model, config = self._model()
        report = run_all(self.arch, model, config, include_slow=True)
        self.assertTrue(report.ok, [c.name + ": " + c.detail for c in report.failures])

    def test_trained_checkpoint_loads_under_unpatched_code(self):
        import tempfile

        model, config = self._model()
        model.train()
        with trainable_routing(self.arch):
            ids = torch.randint(0, config.vocab_size, (2, 12))
            model(input_ids=ids, labels=ids).loss.backward()
        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
        model.eval()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt"
            model.save_pretrained(path, safe_serialization=True)
            self.assertFalse(is_patched(self.arch))
            reloaded = self.arch.causal_lm_cls.from_pretrained(path)
            reloaded.eval()
            ids = torch.randint(0, config.vocab_size, (1, 8))
            with torch.no_grad():
                a = model(input_ids=ids).logits
                b = reloaded(input_ids=ids).logits
            self.assertLess((a - b).abs().max().item(), 1e-5)


@unittest.skipUnless(ARCH, SKIP)
class CliTests(unittest.TestCase):
    def test_info_and_verify(self):
        from mimo_adapter import cli

        self.assertEqual(cli.main(["info"]), 0)
        self.assertEqual(cli.main(["verify", "--fast"]), 0)


if __name__ == "__main__":
    unittest.main()
