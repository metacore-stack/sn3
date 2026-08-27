"""The validator's evaluation contract, transcribed from its source.

Every constant here was read out of ``teutonic/evaluator/engine.py`` rather than
inferred. The comments record where, because if any of these drift your offline
number stops being comparable to the validator's and nothing will warn you.

Reference, engine.py:1237-1255 -- the per-sequence loss::

    n_pos = labels_full.size(1) - 1                  # 2047, not 2048
    for start in range(0, n_pos, chunk_size):        # chunk_size = 1024
        logits = model.lm_head(hidden[:, start:end, :])
        labels = labels_full[:, start + 1 : end + 1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        )
        per_token_losses.append(loss.reshape(batch, -1).float())
    total = torch.cat(per_token_losses, dim=1).sum(dim=1)
    result = (total / n_pos).float().cpu().tolist()
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .errors import EngineMismatchError

# engine.py:85-89 -- statistical and shape constants.
DEFAULT_BATCH_SIZE = 1
DEFAULT_ALPHA = 0.001
DEFAULT_SEQ_LEN = 2048
DEFAULT_BOOTSTRAP_B = 10000

# engine.py:89. This is the engine's *fallback* threshold, matching the
# Teutonic-I generation. The live bar arrives per request in
# request.limits["delta_threshold"] and is currently 0.5. Never plan against
# this value; read the live one from the dashboard or chain.toml.
ENGINE_FALLBACK_DELTA = 0.0015

# engine.py:148-150 -- seeds.
DEFAULT_BOOTSTRAP_SEED = 0xB007  # 45063
DEFAULT_DATASET_SEED = 0xE1A  # 3610, overridden by the block-hash derivation

# engine.py:86, 106 -- fixed execution shape.
DEFAULT_ATTN_IMPLEMENTATION = "eager"
DEFAULT_LM_HEAD_CHUNK = 1024
DEFAULT_DTYPE = "bfloat16"

# engine.py:92-95 -- version strings echoed in every history row.
EVALUATION_POLICY_VERSION = "paired-bootstrap-v1"
EVALUATOR_VERSION = "pair-evaluator-v2"

# engine.py:103 -- server-side cap on requested sequence count.
EVAL_N_CAP = 25000


@dataclass(frozen=True)
class EngineSpec:
    """The execution shape a backend must reproduce.

    ``batch_size`` and ``attn_implementation`` are typed as ``Literal`` in the
    validator, so they are not merely defaults -- a request cannot vary them.
    """

    seq_len: int = DEFAULT_SEQ_LEN
    batch_size: int = DEFAULT_BATCH_SIZE
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION
    dtype: str = DEFAULT_DTYPE
    lm_head_chunk: int = DEFAULT_LM_HEAD_CHUNK
    use_cache: bool = False

    @property
    def n_positions(self) -> int:
        """Predictions per sequence: ``seq_len - 1``."""
        return n_positions(self.seq_len)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["n_positions"] = self.n_positions
        return payload

    def check(self) -> list[str]:
        """Differences from the validator's enforced configuration."""
        problems: list[str] = []
        if self.batch_size != DEFAULT_BATCH_SIZE:
            problems.append(
                f"batch_size {self.batch_size} != {DEFAULT_BATCH_SIZE}; the validator "
                "types this as Literal[1], so a request cannot vary it"
            )
        if self.attn_implementation != DEFAULT_ATTN_IMPLEMENTATION:
            problems.append(
                f"attn_implementation {self.attn_implementation!r} != "
                f"{DEFAULT_ATTN_IMPLEMENTATION!r}; sdpa and flash differ numerically"
            )
        if self.seq_len != DEFAULT_SEQ_LEN:
            problems.append(f"seq_len {self.seq_len} != {DEFAULT_SEQ_LEN}")
        if self.use_cache:
            problems.append("use_cache must be False")
        if self.dtype != DEFAULT_DTYPE:
            problems.append(
                f"dtype {self.dtype!r} != {DEFAULT_DTYPE!r}; the king ships in bf16"
            )
        return problems

    def require(self) -> None:
        problems = self.check()
        if problems:
            raise EngineMismatchError("; ".join(problems))


@dataclass(frozen=True)
class StatsSpec:
    """Arguments passed to ``paired_bootstrap_verdict``.

    ``alpha`` and ``n_bootstrap`` are engine defaults that the validator may
    override via ``request.limits``; the dashboard does not publish the values
    actually used. They live in config rather than inline so a discovered
    mismatch is a one-line change.
    """

    alpha: float = DEFAULT_ALPHA
    n_bootstrap: int = DEFAULT_BOOTSTRAP_B
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    delta_threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def n_positions(seq_len: int = DEFAULT_SEQ_LEN) -> int:
    """Number of next-token predictions in a sequence.

    ``seq_len - 1``: position *i* predicts token *i+1*, so the final token is a
    label only and never a prediction. Dividing by ``seq_len`` instead inflates
    every loss by a factor of 2048/2047.
    """
    if seq_len < 2:
        raise ValueError(f"seq_len must be at least 2, got {seq_len}")
    return seq_len - 1


def reduce_per_token(
    per_token: Sequence[float], *, seq_len: int = DEFAULT_SEQ_LEN
) -> float:
    """Collapse per-token losses to the one number the validator records.

    Sums first and divides once, mirroring the engine's concatenate-then-sum.
    The engine does this explicitly so that ``lm_head_chunk`` cannot change the
    result; summing per chunk and averaging the chunk means would not match.
    """
    expected = n_positions(seq_len)
    if len(per_token) != expected:
        raise ValueError(
            f"expected {expected} per-token losses for seq_len={seq_len}, "
            f"got {len(per_token)}"
        )
    return float(sum(float(x) for x in per_token) / expected)


def describe() -> dict[str, Any]:
    """The full contract, for embedding in a saved loss vector."""
    return {
        "engine": EngineSpec().to_dict(),
        "stats": StatsSpec().to_dict(),
        "evaluator_version": EVALUATOR_VERSION,
        "policy_version": EVALUATION_POLICY_VERSION,
        "engine_fallback_delta": ENGINE_FALLBACK_DELTA,
        "eval_n_cap": EVAL_N_CAP,
    }
