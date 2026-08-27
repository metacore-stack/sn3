"""The loader itself: refs in, token sequences out, holdouts strictly enforced.

The contamination guard is the point of this class. Training on your own
validation sequences inflates every number produced afterwards and stays
invisible until the compute budget is gone, so it raises rather than warns.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .cache import ShardCache
from .errors import ContaminationError, LoaderError
from .manifest import ShardManifest
from .npyio import NUMPY_AVAILABLE, Shard
from .refs import SequenceRef, SequenceSet


@dataclass
class LoaderStats:
    shards_opened: int = 0
    sequences_read: int = 0
    bytes_downloaded: int = 0


class FineWebLoader:
    """Reads sequences by ref, grouping by shard so each is opened once."""

    def __init__(
        self,
        manifest: ShardManifest,
        cache: ShardCache,
        *,
        holdouts: Sequence[SequenceSet] = (),
        prefer_numpy: bool = True,
    ):
        self.manifest = manifest
        self.cache = cache
        self.prefer_numpy = prefer_numpy
        self.stats = LoaderStats()
        self._holdouts: list[SequenceSet] = list(holdouts)
        self._banned: set[SequenceRef] = set()
        for holdout in self._holdouts:
            self._banned |= holdout.as_set()
        self._open: dict[str, Shard] = {}

    # -- holdout management ------------------------------------------------

    @property
    def holdouts(self) -> tuple[SequenceSet, ...]:
        return tuple(self._holdouts)

    @property
    def excluded_count(self) -> int:
        return len(self._banned)

    def add_holdout(self, holdout: SequenceSet) -> None:
        """Register a set whose sequences may never be served for training."""
        if holdout.manifest_sha256 != self.manifest.digest:
            raise LoaderError(
                f"holdout {holdout.name!r} was built against manifest "
                f"{holdout.manifest_sha256[:12]}… but this manifest is "
                f"{self.manifest.digest[:12]}…"
            )
        self._holdouts.append(holdout)
        self._banned |= holdout.as_set()

    def contamination(self, refs: Iterable[SequenceRef]) -> set[SequenceRef]:
        """Which of these refs are held out."""
        return {r for r in refs if r in self._banned}

    def assert_clean(self, refs: Iterable[SequenceRef]) -> None:
        """Raise if any ref is held out. Never downgrade this to a warning."""
        overlap = self.contamination(refs)
        if overlap:
            sample = ", ".join(str(r) for r in sorted(overlap)[:5])
            raise ContaminationError(
                f"{len(overlap)} requested sequence(s) are held out (e.g. {sample}). "
                "Training on them would invalidate every subsequent measurement."
            )

    # -- reading -----------------------------------------------------------

    def open_shard(self, shard_name: str) -> Shard:
        """Fetch if needed, then memory-map. Shards stay open until closed."""
        if shard_name in self._open:
            return self._open[shard_name]
        entry = self.manifest.lookup(shard_name)
        had = self.cache.has(entry)
        path = self.cache.ensure(entry)
        if not had:
            self.stats.bytes_downloaded += entry.size_bytes
        shard = Shard(path, seq_len=self.manifest.seq_len, prefer_numpy=self.prefer_numpy)
        if shard.n_sequences < 1:
            shard.close()
            raise LoaderError(f"{shard_name} contains no whole sequences")
        self._open[shard_name] = shard
        self.stats.shards_opened += 1
        return shard

    def sequences(self, refs: Sequence[SequenceRef], *, allow_holdout: bool = True):
        """Read the given refs, grouped by shard.

        ``allow_holdout`` is True here because evaluation legitimately reads
        holdout sequences; only the training path forbids it.
        """
        if not allow_holdout:
            self.assert_clean(refs)
        grouped: dict[str, list[int]] = defaultdict(list)
        order: list[tuple[str, int]] = []
        for ref in refs:
            grouped[ref.shard].append(ref.index)
            order.append((ref.shard, ref.index))

        collected: dict[tuple[str, int], object] = {}
        for shard_name, indices in grouped.items():
            shard = self.open_shard(shard_name)
            for index in indices:
                collected[(shard_name, index)] = shard.sequence(index)
        self.stats.sequences_read += len(order)

        rows = [collected[key] for key in order]
        if NUMPY_AVAILABLE and self.prefer_numpy and rows:
            import numpy as np

            return np.stack(rows)
        return rows

    def batches(
        self,
        refs: Sequence[SequenceRef],
        batch_size: int = 8,
        *,
        allow_holdout: bool = True,
    ) -> Iterator[tuple[list[SequenceRef], object]]:
        """Yield ``(refs, data)`` in fixed-size batches, preserving order."""
        if batch_size < 1:
            raise LoaderError("batch_size must be at least 1")
        if not allow_holdout:
            self.assert_clean(refs)
        window = list(refs)
        for start in range(0, len(window), batch_size):
            chunk = window[start : start + batch_size]
            yield chunk, self.sequences(chunk, allow_holdout=True)

    def training_stream(
        self,
        *,
        seed: int,
        shards: Sequence[str],
        batch_size: int = 8,
        max_batches: int | None = None,
    ) -> Iterator[tuple[list[SequenceRef], object]]:
        """Shuffled training batches drawn only from non-held-out sequences.

        Holdout refs are filtered out up front and the result is asserted clean,
        so contamination cannot occur even if a caller passes a shard that a
        holdout also samples from.
        """
        import random

        rng = random.Random(seed)
        pool: list[SequenceRef] = []
        for shard_name in sorted(set(shards)):
            entry = self.manifest.lookup(shard_name)
            available = entry.sequences(self.manifest.seq_len)
            pool.extend(
                ref
                for i in range(available)
                if (ref := SequenceRef(entry.name, i)) not in self._banned
            )
        if not pool:
            raise LoaderError("no trainable sequences remain after holdout exclusion")

        rng.shuffle(pool)
        self.assert_clean(pool)

        emitted = 0
        for start in range(0, len(pool), batch_size):
            if max_batches is not None and emitted >= max_batches:
                return
            chunk = pool[start : start + batch_size]
            yield chunk, self.sequences(chunk, allow_holdout=True)
            emitted += 1

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        for shard in self._open.values():
            shard.close()
        self._open.clear()

    def __enter__(self) -> "FineWebLoader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"<FineWebLoader shards_open={len(self._open)} "
            f"holdouts={len(self._holdouts)} excluded={len(self._banned)}>"
        )
