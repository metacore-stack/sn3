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


def blended_batches(
    config: TrainingConfig,
    *,
    state_root: Path | None = None,
    budget_gib: float = 4.0,
    match_proportions: bool = True,
) -> tuple[Iterator, dict]:
    """Batches drawn across every evaluation corpus.

    Since 2026-08-27 the validator scores a fixed 22/26/52 blend of
    finewebedu, automathtext-v2 and dclm-baseline-1.0. Training on one source
    trains on that source's share of the score, and the live history shows what
    the mismatch costs: challengers built for the single-corpus contract scored
    -0.5 to -1.05 within hours of the change.

    ``match_proportions`` sizes the pool to the validator's shares rather than
    to whatever happens to be cached, so a lopsided local cache does not quietly
    become a lopsided training mixture.
    """
    from fineweb_loader import BlendedLoader, CorpusSet, DatasetConfig, SequenceSet

    root = Path(state_root or (Path.home() / "Documents" / "sn3" / "state"))
    dataset = DatasetConfig.load(root / "dataset-config.json")
    corpora = CorpusSet.open(
        dataset, root, budget_bytes=int(budget_gib * 1024**3)
    )

    holdouts = []
    for name in config.data.holdouts:
        path = root / "holdouts" / f"{name}.json"
        if not path.is_file():
            raise ConfigError(f"holdout {name!r} not found at {path}")
        holdouts.append(SequenceSet.load(path))

    if not config.data.shards:
        raise ConfigError("config.data.shards is empty; nothing to train on")

    loader = BlendedLoader(corpora, holdouts=holdouts)
    proportions = (
        {s.name: s.proportion for s in dataset.sources} if match_proportions else None
    )
    stream = loader.training_stream(
        seed=config.data.seed,
        shards=list(config.data.shards),
        batch_size=config.data.batch_size,
        max_batches=config.data.max_batches,
        proportions=proportions,
    )
    present = sorted({corpora.corpus_of(s).name for s in config.data.shards})
    missing = [n for n in dataset.names if n not in present]
    context = {
        "source": "blend",
        "dataset_label": dataset.dataset_label,
        "config_version": dataset.config_version,
        "delta_threshold": dataset.delta_threshold,
        "corpora": present,
        "missing_corpora": missing,
        "proportions": proportions,
        "holdouts": [h.name for h in holdouts],
        "excluded_sequences": loader.excluded_count,
        "shards": list(config.data.shards),
        "seq_len": next(iter(corpora)).manifest.seq_len,
    }
    return stream, context


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
    blend: bool = True,
) -> tuple[Iterator, dict]:
    """Pick a source, and report what it is.

    Defaults to the blend, because that is what the validator scores. Pass
    ``blend=False`` for the single-corpus path, which now covers 22% of the
    score and is kept only for comparison runs.
    """
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
    if blend:
        root = Path(state_root or (Path.home() / "Documents" / "sn3" / "state"))
        if (root / "dataset-config.json").is_file():
            return blended_batches(config, state_root=state_root)
        raise ConfigError(
            "no dataset-config.json; run 'fineweb corpus sync' so training "
            "matches the validator's 22/26/52 blend, or pass --single-corpus"
        )
    stream, context = fineweb_batches(config, state_root=state_root)
    context["source"] = "finewebedu"
    return stream, context
