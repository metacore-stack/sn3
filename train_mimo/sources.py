"""Batch sources.

Training data always arrives through ``fineweb_loader``'s contamination-guarded
stream. The synthetic source exists only so the loop itself can be exercised
without any shards on disk; it is never a substitute for real data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

from .config import TrainingConfig
from .errors import ConfigError


def synthetic_batches(
    *,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    steps: int,
    seed: int = 0,
) -> Iterator[tuple[list[str], object]]:
    """Random token ids. Exercises the loop, teaches the model nothing."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    for step in range(steps):
        rows = torch.randint(
            0, vocab_size, (batch_size, seq_len), generator=generator, dtype=torch.long
        )
        refs = [f"synthetic#{step * batch_size + i}" for i in range(batch_size)]
        yield refs, rows


def fineweb_batches(
    config: TrainingConfig,
    *,
    state_root: Path | None = None,
    budget_gib: float = 4.0,
) -> tuple[Iterator, dict]:
    """Real FineWeb-Edu batches, with every declared holdout excluded.

    Returns ``(iterator, context)``; the context carries the manifest digest and
    holdout names for the run's provenance record.
    """
    from fineweb_loader import FineWebLoader, SequenceSet, ShardCache, ShardManifest

    root = Path(state_root or (Path.home() / "Documents" / "sn3" / "state"))
    manifest = ShardManifest.load(root / "fineweb-manifest.json")
    cache = ShardCache(root / "cache", manifest, budget_bytes=int(budget_gib * 1024**3))

    holdouts = []
    for name in config.data.holdouts:
        path = root / "holdouts" / f"{name}.json"
        if not path.is_file():
            raise ConfigError(f"holdout {name!r} not found at {path}")
        holdouts.append(SequenceSet.load(path))

    if not config.data.shards:
        raise ConfigError("config.data.shards is empty; nothing to train on")

    loader = FineWebLoader(manifest, cache, holdouts=holdouts)
    stream = loader.training_stream(
        seed=config.data.seed,
        shards=list(config.data.shards),
        batch_size=config.data.batch_size,
        max_batches=config.data.max_batches,
    )
    context = {
        "manifest_sha256": manifest.digest,
        "holdouts": [h.name for h in holdouts],
        "excluded_sequences": loader.excluded_count,
        "shards": list(config.data.shards),
        "seq_len": manifest.seq_len,
    }
    return stream, context


def resolve_batches(
    config: TrainingConfig,
    *,
    vocab_size: int,
    seq_len: int,
    synthetic: bool = False,
    state_root: Path | None = None,
) -> tuple[Iterator, dict]:
    """Pick a source, and report what it is."""
    if synthetic:
        steps = config.max_steps * config.data.grad_accum + 1
        return (
            synthetic_batches(
                vocab_size=vocab_size,
                seq_len=seq_len,
                batch_size=config.data.batch_size,
                steps=steps,
                seed=config.data.seed,
            ),
            {"source": "synthetic", "seq_len": seq_len, "vocab_size": vocab_size},
        )
    stream, context = fineweb_batches(config, state_root=state_root)
    context["source"] = "finewebedu"
    return stream, context
