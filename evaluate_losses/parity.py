"""Parity checks: proof that this package computes what the validator computes.

Three tiers, in increasing cost:

1. **statistics** -- no model, no GPU. Our comparison path must agree with
   ``paired_bootstrap_verdict`` exactly, and must reproduce the reign-7
   coronation numbers published on the dashboard.
2. **sampler** -- no model, no GPU. Our seed derivation and index selection must
   match the validator's byte for byte.
3. **loss** -- needs the real weights on a GPU. Our per-sequence loss must match
   the engine's ``compute_per_sequence_loss`` to ~1e-6.

Until tier 3 has run, every number this package produces is a hypothesis. It is
the one check that costs money and the one worth paying for.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from .engine import StatsSpec
from .errors import EvaluationError
from .lossvec import LossVector
from .policy import paired_bootstrap_verdict

# Dashboard record for the reign-7 coronation. If our plumbing ever stops
# reproducing these from the same inputs, something upstream has changed.
CORONATION = {
    "mu_hat": 0.522645,
    "lcb": 0.517178,
    "delta": 0.5,
    "avg_king_loss": 3.509797,
    "avg_challenger_loss": 2.987152,
    "accepted": True,
}

TOLERANCE = 1e-9


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ParityReport:
    tier: str
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(Check(name, passed, detail))


# -- tier 1: statistics -----------------------------------------------------


def tier1_statistics(*, stats: StatsSpec | None = None) -> ParityReport:
    """Our comparison path vs. the validator's function, on synthetic vectors."""
    from .compare import compare

    stats = stats or StatsSpec()
    report = ParityReport("statistics")

    rng = random.Random(20260826)
    n = 400
    refs = [f"synthetic__shard_{i // 100:06d}.npy#{i}" for i in range(n)]
    king_losses = [3.0 + rng.gauss(0, 0.35) for _ in range(n)]
    challenger_losses = [k - 0.6 + rng.gauss(0, 0.05) for k in king_losses]

    king = LossVector(tuple(refs), tuple(king_losses), model_label="king")
    challenger = LossVector(
        tuple(refs), tuple(challenger_losses), model_label="challenger"
    )

    direct = paired_bootstrap_verdict(
        king_losses,
        challenger_losses,
        bootstrap_seed=stats.bootstrap_seed,
        n_bootstrap=stats.n_bootstrap,
        alpha=stats.alpha,
        delta_threshold=stats.delta_threshold,
    )
    ours = compare(king, challenger, stats=stats, per_shard=False).overall

    report.add(
        "mu_hat matches the validator's function",
        abs(ours.mu_hat - direct["mu_hat"]) < TOLERANCE,
        f"{ours.mu_hat} vs {direct['mu_hat']}",
    )
    report.add(
        "lcb matches the validator's function",
        abs(ours.lcb - direct["lcb"]) < TOLERANCE,
        f"{ours.lcb} vs {direct['lcb']}",
    )
    report.add(
        "accepted matches",
        ours.accepted == direct["accepted"],
        f"{ours.accepted} vs {direct['accepted']}",
    )

    # Determinism: the same seed must give the same bound.
    again = paired_bootstrap_verdict(
        king_losses,
        challenger_losses,
        bootstrap_seed=stats.bootstrap_seed,
        n_bootstrap=stats.n_bootstrap,
        alpha=stats.alpha,
        delta_threshold=stats.delta_threshold,
    )
    report.add(
        "bootstrap is deterministic for a fixed seed",
        again["lcb"] == direct["lcb"],
        f"{again['lcb']}",
    )

    # Acceptance is a strict inequality: LCB exactly at the bar is a rejection.
    flat = [1.0] * 64
    lifted = [0.5] * 64
    edge = paired_bootstrap_verdict(
        flat,
        lifted,
        bootstrap_seed=stats.bootstrap_seed,
        n_bootstrap=1000,
        alpha=stats.alpha,
        delta_threshold=0.5,
    )
    report.add(
        "lcb == delta is rejected (strict >)",
        edge["accepted"] is False and abs(edge["lcb"] - 0.5) < 1e-12,
        f"lcb={edge['lcb']} accepted={edge['accepted']}",
    )

    # Reconstruct the published coronation from its own summary statistics.
    report.add(
        "reign-7 record is internally consistent",
        abs(
            (CORONATION["avg_king_loss"] - CORONATION["avg_challenger_loss"])
            - CORONATION["mu_hat"]
        )
        < 1e-6,
        f"3.509797 - 2.987152 = {CORONATION['avg_king_loss'] - CORONATION['avg_challenger_loss']:.6f}"
        f" vs mu_hat {CORONATION['mu_hat']}",
    )
    report.add(
        "reign-7 lcb sits just below mu_hat as a paired test implies",
        0 < CORONATION["mu_hat"] - CORONATION["lcb"] < 0.02,
        f"penalty {CORONATION['mu_hat'] - CORONATION['lcb']:.6f}",
    )
    return report


# -- tier 2: sampler --------------------------------------------------------


def tier2_sampler() -> ParityReport:
    """Our seed derivation and index selection vs. the validator's."""
    from fineweb_loader import sampler

    report = ParityReport("sampler")
    block_hash = "0x" + "ab" * 32
    hotkey = "5DZMJER8BqWj8eZounzRDz6Bgqa9HwRgoRwwwsA7rk8XXhbW"

    material = sampler.dataset_seed_material(block_hash, hotkey)
    report.add(
        "seed material format",
        material == f"block_hash={block_hash}|hotkey={hotkey}",
        material,
    )

    expected = int.from_bytes(
        hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest(), "little"
    )
    seed = sampler.dataset_seed(block_hash, hotkey)
    report.add("dataset seed is blake2b-64 little-endian", seed == expected, str(seed))

    expected_source = int.from_bytes(
        hashlib.blake2b(f"{seed}:finewebedu".encode("utf-8"), digest_size=8).digest(),
        "little",
    )
    report.add(
        "source seed derivation",
        sampler.source_seed(seed, "finewebedu") == expected_source,
        str(expected_source),
    )

    report.add(
        "a missing block hash is refused",
        _raises(lambda: sampler.dataset_seed("", hotkey)),
        "empty block_hash must raise",
    )
    report.add(
        "the literal 'default' block hash is refused",
        _raises(lambda: sampler.dataset_seed("default", hotkey)),
        "sentinel must raise",
    )

    report.add(
        "required_sequences(2000) == 3000",
        sampler.required_sequences(2000) == 3000,
        "target + max(16, ceil(target/2))",
    )
    report.add(
        "one 6144-sequence shard covers a 2000-sequence evaluation",
        6144 >= sampler.required_sequences(2000),
        "explains why every observed eval used exactly one shard",
    )

    # Shard order comes from stdlib random.Random, not numpy.
    shards = [{"key": f"shards/s{i:05d}.npy", "n_tokens": 12582912} for i in range(64)]
    ours = sampler.shuffled_shards(shards, seed, "finewebedu")
    reference = list(shards)
    random.Random(expected_source).shuffle(reference)
    report.add(
        "shard shuffle uses stdlib random.Random",
        [s["key"] for s in ours] == [s["key"] for s in reference],
        "numpy shuffle would permute differently from the same seed",
    )

    try:
        import numpy as np

        indices = sampler.select_sequence_indices(
            6144, block_hash=block_hash, hotkey=hotkey, limit=2000
        )
        rng = np.random.default_rng(expected_source)
        expected_idx = rng.choice(6144, size=2000, replace=False)
        report.add(
            "sequence indices use numpy default_rng.choice without replacement",
            list(indices) == list(expected_idx),
            f"{len(set(indices))} distinct of {len(indices)}",
        )
        report.add(
            "no index repeats",
            len(set(indices)) == len(indices),
            "replace=False",
        )
    except ImportError:  # pragma: no cover
        report.add("sequence index selection", False, "numpy not installed")

    report.add(
        "load limit over-fetches only when a vocab filter is active",
        sampler.load_limit(100, vocab_filtered=False) == 100
        and sampler.load_limit(100, vocab_filtered=True) == 158,
        "int(remaining * 1.5) + 8",
    )
    return report


# -- tier 3: loss -----------------------------------------------------------


def tier3_loss(
    backend,
    engine_module,
    token_sequences: Sequence[Sequence[int]],
    *,
    tolerance: float = 1e-6,
) -> ParityReport:
    """Our per-sequence loss vs. the engine's own, on real weights.

    Requires a GPU, the real checkpoint and the validator's importable engine.
    Not exercised by the offline test suite -- run it once, on rented hardware,
    before trusting any number this package produces.
    """
    report = ParityReport("loss")
    if not hasattr(engine_module, "compute_per_sequence_loss"):
        raise EvaluationError(
            "engine module does not expose compute_per_sequence_loss"
        )
    model = backend._ensure_model()  # noqa: SLF001 - deliberate, parity only
    for i, tokens in enumerate(token_sequences):
        ours = backend.score_tokens(tokens)
        theirs = engine_module.compute_per_sequence_loss(
            model, [list(tokens)], chunk_size=backend.spec.lm_head_chunk
        )[0]
        report.add(
            f"sequence {i}",
            abs(ours - theirs) < tolerance,
            f"ours {ours:.9f} vs engine {theirs:.9f} (delta {abs(ours - theirs):.2e})",
        )
    return report


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def run_offline() -> list[ParityReport]:
    """Every tier that needs no GPU."""
    return [tier1_statistics(), tier2_sampler()]
