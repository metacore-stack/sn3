"""A miniature MiMo model that keeps the routing semantics but fits on a laptop.

The routing question -- can ``noaux_tc`` be made trainable, and does a patched
gate reproduce the original numerically -- does not depend on model size. It
depends on ``topk_method``, ``scoring_func``, ``norm_topk_prob``, ``n_group`` and
``topk_group``, which are copied from the real king's config verbatim.

Everything else is shrunk. The result is a few million random parameters that
exercise the same code path as the 110B checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Fields that define routing behaviour. Never overridden by a miniature.
ROUTING_KEYS = (
    "topk_method",
    "scoring_func",
    "norm_topk_prob",
    "n_group",
    "topk_group",
    "routed_scaling_factor",
    "n_shared_experts",
)

# Config entries that are lists with one element per layer.
PER_LAYER_KEYS = ("moe_layer_freq", "hybrid_layer_pattern", "layer_types")


@dataclass(frozen=True)
class MiniatureSpec:
    """Size overrides for a scaled-down model.

    Defaults are chosen so the model builds in about a second on CPU while
    keeping a dense first layer plus several MoE layers, which is the real
    layout (``moe_layer_freq`` starts ``[0, 1, 1, …]``).
    """

    num_hidden_layers: int = 4
    hidden_size: int = 128
    intermediate_size: int = 256
    moe_intermediate_size: int = 64
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 48
    v_head_dim: int = 32
    swa_num_attention_heads: int = 4
    swa_num_key_value_heads: int = 2
    swa_head_dim: int = 48
    swa_v_head_dim: int = 32
    vocab_size: int = 256
    max_position_embeddings: int = 512
    sliding_window: int = 32
    sliding_window_size: int = 32
    attention_chunk_size: int = 32
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def real_vocab(cls, **kw: Any) -> "MiniatureSpec":
        """A miniature that speaks the real tokenizer.

        Keeping ``vocab_size`` at the king's 152,576 costs about 39M parameters
        in embeddings and the LM head -- still trainable on CPU -- and is what
        lets real FineWeb-Edu shards be fed through the loop end to end. With a
        toy vocabulary the token ids are out of range and nothing downstream is
        being tested honestly.
        """
        defaults = {"vocab_size": 152576, "max_position_embeddings": 4096}
        defaults.update(kw)
        return cls(**defaults)

    def overrides(self) -> dict[str, Any]:
        payload = {
            k: v
            for k, v in self.__dict__.items()
            if k != "extra" and not k.startswith("_")
        }
        payload.update(self.extra)
        return payload


def miniature_config_dict(
    reference: dict[str, Any], spec: MiniatureSpec | None = None
) -> dict[str, Any]:
    """Shrink a real config while preserving every routing decision.

    Per-layer lists are truncated (or extended by repeating their tail) so their
    length always matches ``num_hidden_layers``; a mismatch there is a confusing
    failure deep inside model construction.
    """
    spec = spec or MiniatureSpec()
    config = dict(reference)

    for key in ROUTING_KEYS:
        if key in reference:
            config[key] = reference[key]

    config.update(spec.overrides())

    layers = int(config["num_hidden_layers"])
    for key in PER_LAYER_KEYS:
        value = reference.get(key)
        if not isinstance(value, list) or not value:
            continue
        if len(value) >= layers:
            config[key] = list(value[:layers])
        else:
            config[key] = list(value) + [value[-1]] * (layers - len(value))

    # A miniature is never submitted, so remove the fields that only make sense
    # for a real checkpoint and would mislead anyone reading it.
    for key in ("auto_map", "architectures", "transformers_version", "dtype"):
        config.pop(key, None)

    config["use_cache"] = False
    return config


def initialize_gates(model, config, *, seed: int = 0) -> int:
    """Initialise routing parameters, which the architecture leaves uninitialised.

    ``MiMoV2MoEGate.__init__`` creates both of its parameters with
    ``torch.empty()``::

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.e_score_correction_bias = nn.Parameter(torch.empty((self.n_routed_experts)))

    and ``_init_weights`` does not cover them. For the real king this never
    bites, because the values arrive from the checkpoint. Any model built from a
    config alone gets whatever was in that memory -- observed here as gate
    weights of 2e+38, which saturate the sigmoid to exactly 1.0 and produce
    exactly-zero gradients that look like a broken router rather than a broken
    fixture.

    Returns the number of gates initialised.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    std = float(getattr(config, "initializer_range", 0.02) or 0.02)
    count = 0
    for module in model.modules():
        if type(module).__name__ != "MiMoV2MoEGate":
            continue
        with torch.no_grad():
            module.weight.normal_(mean=0.0, std=std, generator=generator)
            bias = getattr(module, "e_score_correction_bias", None)
            if bias is not None:
                # Zero is the neutral starting point: selection then follows the
                # scores alone, which is what an untrained router should do.
                bias.zero_()
        count += 1
    return count


def build_miniature(
    arch,
    reference: dict[str, Any],
    spec: MiniatureSpec | None = None,
    *,
    seed: int = 0,
    dtype: str = "float32",
    init_gates: bool = True,
):
    """Instantiate a miniature ``MiMoV2ForCausalLM`` with random weights.

    float32 by default: the point is exact numerical comparison between two
    routing implementations, and bf16 rounding would mask real differences.

    ``init_gates`` is on by default because the architecture leaves routing
    parameters uninitialised; see :func:`initialize_gates`. Pass ``False`` only
    to reproduce that failure deliberately.
    """
    import torch

    payload = miniature_config_dict(reference, spec)
    config = arch.config_cls(**payload)

    torch.manual_seed(seed)
    model = arch.causal_lm_cls(config)
    if init_gates:
        initialize_gates(model, config, seed=seed)
    model = model.to(getattr(torch, dtype))
    model.eval()
    return model, config


def describe(config) -> dict[str, Any]:
    """The routing-relevant summary of a built config."""
    return {
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "n_routed_experts": getattr(config, "n_routed_experts", None),
        "num_experts_per_tok": getattr(config, "num_experts_per_tok", None),
        "n_shared_experts": getattr(config, "n_shared_experts", None),
        "topk_method": getattr(config, "topk_method", None),
        "scoring_func": getattr(config, "scoring_func", None),
        "norm_topk_prob": getattr(config, "norm_topk_prob", None),
        "n_group": getattr(config, "n_group", None),
        "topk_group": getattr(config, "topk_group", None),
        "routed_scaling_factor": getattr(config, "routed_scaling_factor", None),
        "vocab_size": getattr(config, "vocab_size", None),
    }


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
