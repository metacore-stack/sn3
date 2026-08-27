"""The validator's shard and sequence sampler, transcribed from its source.

This module previously held guesses. It no longer does -- every step below was
read out of the cloned repository:

* ``teutonic/evaluation/configuration.py`` builds the request: it derives the
  dataset seed from the finalized block hash and the miner's hotkey, shuffles
  the shard list with **stdlib** ``random.Random``, and walks that order until
  enough sequences are available.
* ``teutonic/evaluator/sources.py`` and ``engine.py`` then pick sequences inside
  each shard with **numpy** ``default_rng``.

Two different RNGs seeded from the same integer. Using one where the other
belongs silently produces a different sample.

This still cannot predict your own evaluation: the block hash is created when
your submission finalizes, after you commit. Its use is the reverse -- given a
known ``(block_hash, hotkey)``, reproduce what the validator did and confirm you
understand the mechanism.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

try:  # numpy is required only for sequence selection
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

DIGEST_SIZE = 8

# configuration.py:_source_targets -- headroom over the requested count.
OVERSAMPLE_FLOOR = 16
OVERSAMPLE_FRACTION = 0.5

# sources.py:sample_pretokenized_sequences -- per-shard load headroom when a
# vocab filter is active.
LOAD_LIMIT_FACTOR = 1.5
LOAD_LIMIT_PAD = 8


def dataset_seed_material(block_hash: str, hotkey: str) -> str:
    """``engine.py:dataset_seed_material`` / ``configuration.py:_dataset_seed``."""
    block_hash = (block_hash or "").strip()
    hotkey = (hotkey or "").strip()
    if not block_hash or block_hash == "default":
        raise ValueError(
            "pretokenized evaluation requires a finalized block hash"
        )
    return f"block_hash={block_hash}|hotkey={hotkey}"


def dataset_seed(block_hash: str, hotkey: str) -> int:
    """64-bit dataset seed. This is ``blake2b-64-block-hash-hotkey-v1``."""
    material = dataset_seed_material(block_hash, hotkey)
    digest = hashlib.blake2b(material.encode("utf-8"), digest_size=DIGEST_SIZE).digest()
    return int.from_bytes(digest, "little")


def source_seed(seed: int, source_name: str) -> int:
    """Per-corpus seed: ``blake2b(f"{seed}:{name}")``, little-endian."""
    digest = hashlib.blake2b(
        f"{seed}:{source_name}".encode("utf-8"), digest_size=DIGEST_SIZE
    ).digest()
    return int.from_bytes(digest, "little")


def required_sequences(target: int) -> int:
    """Sequences a source must be able to supply for ``target`` to be met.

    ``target + max(16, ceil(target * 0.5))``. For the live ``eval_n`` of 2000
    that is 3000 -- comfortably inside one 6144-sequence shard, which is why
    every observed evaluation drew from exactly one shard.
    """
    return target + max(OVERSAMPLE_FLOOR, math.ceil(target * OVERSAMPLE_FRACTION))


def shuffled_shards(
    shards: Sequence[dict[str, Any]], seed: int, source_name: str
) -> list[dict[str, Any]]:
    """Shard order for one evaluation.

    Uses stdlib ``random.Random`` exactly as ``configuration.py`` does. numpy's
    shuffle would give a different permutation from the same seed.
    """
    ordered = list(shards)
    random.Random(source_seed(seed, source_name)).shuffle(ordered)
    return ordered


def select_shards(
    shards: Sequence[dict[str, Any]],
    *,
    block_hash: str,
    hotkey: str,
    target_sequences: int,
    source_name: str = "finewebedu",
    seq_len: int = 2048,
) -> list[dict[str, Any]]:
    """Shards the validator would walk, in order, until the target is covered."""
    seed = dataset_seed(block_hash, hotkey)
    ordered = shuffled_shards(shards, seed, source_name)
    needed = required_sequences(target_sequences)

    chosen: list[dict[str, Any]] = []
    available = 0
    for shard in ordered:
        if available >= needed:
            break
        chosen.append(shard)
        available += int(shard["n_tokens"]) // seq_len
    return chosen


def load_limit(remaining: int, *, vocab_filtered: bool) -> int:
    """Sequences pulled from a shard for ``remaining`` still wanted.

    With a vocab filter active the engine over-fetches, because sequences
    containing an out-of-range token id are dropped after loading.
    """
    if not vocab_filtered:
        return remaining
    return int(remaining * LOAD_LIMIT_FACTOR) + LOAD_LIMIT_PAD


def shuffled_indices(rng, size: int, limit: int | None = None):
    """``engine.py:shuffled_indices`` verbatim."""
    if _np is None:  # pragma: no cover
        raise RuntimeError("numpy is required for sequence selection parity")
    if limit is None or limit >= size:
        indices = _np.arange(size)
        rng.shuffle(indices)
        return indices
    return rng.choice(size, size=limit, replace=False)


def select_sequence_indices(
    n_sequences: int,
    *,
    block_hash: str,
    hotkey: str,
    limit: int | None = None,
    source_name: str = "finewebedu",
):
    """Sequence indices the validator would take from one shard.

    Selection is without replacement via ``numpy.random.Generator.choice``.
    """
    if _np is None:  # pragma: no cover
        raise RuntimeError("numpy is required for sequence selection parity")
    seed = source_seed(dataset_seed(block_hash, hotkey), source_name)
    rng = _np.random.default_rng(seed)
    return shuffled_indices(rng, n_sequences, limit)


@dataclass(frozen=True)
class Observation:
    """A known ``(block_hash, hotkey) -> shard`` outcome to check against."""

    block_hash: str
    hotkey: str
    shard_name: str


def verify(
    manifest_shards: Sequence[dict[str, Any]],
    observations: Iterable[Observation],
    *,
    target_sequences: int = 2000,
    source_name: str = "finewebedu",
    seq_len: int = 2048,
) -> list[tuple[Observation, bool, str]]:
    """Check this implementation against recorded outcomes.

    Returns ``(observation, matched, predicted_first_shard)`` per observation.
    The dashboard does not publish per-attempt block hashes, so the triples have
    to be assembled from chain data before this can be run.
    """
    results = []
    for observed in observations:
        chosen = select_shards(
            manifest_shards,
            block_hash=observed.block_hash,
            hotkey=observed.hotkey,
            target_sequences=target_sequences,
            source_name=source_name,
            seq_len=seq_len,
        )
        predicted = ""
        if chosen:
            reference = next(
                str(chosen[0][key]).strip()
                for key in ("url", "href", "uri", "key", "path", "name")
                if chosen[0].get(key)
            )
            predicted = reference.rsplit("/", 1)[-1]
        results.append((observed, predicted == observed.shard_name, predicted))
    return results
