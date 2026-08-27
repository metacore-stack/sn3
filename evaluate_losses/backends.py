"""Scoring backends: the only part of this package that needs a GPU.

Everything else -- alignment, statistics, per-shard reporting, persistence --
runs on CPU against a :class:`ReplayBackend`, so the whole pipeline is testable
today and GPU time is spent only filling in loss vectors.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Sequence

from .engine import EngineSpec, n_positions
from .errors import BackendUnavailableError, EvaluationError
from .lossvec import LossVector


class ScoringBackend(ABC):
    """Produces one loss per sequence."""

    spec: EngineSpec

    @abstractmethod
    def score(
        self,
        refs: Sequence[str],
        *,
        model_label: str,
        progress: Callable[[int, int], None] | None = None,
    ) -> LossVector:
        """Return a loss vector aligned to ``refs``, in that order."""

    def close(self) -> None:  # pragma: no cover - default is a no-op
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class ReplayBackend(ScoringBackend):
    """Serves losses from a saved vector.

    Not a mock in the pejorative sense: it replays real measurements, so the
    comparison, alignment, reporting and CLI paths are exercised against genuine
    numbers without touching a GPU.
    """

    def __init__(self, source: LossVector | Path | str, *, spec: EngineSpec | None = None):
        if isinstance(source, (str, Path)):
            source = LossVector.load(Path(source))
        self.vector = source
        self.spec = spec or EngineSpec()
        self._map = source.as_map()

    def score(
        self,
        refs: Sequence[str],
        *,
        model_label: str | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> LossVector:
        refs = list(refs)
        missing = [r for r in refs if r not in self._map]
        if missing:
            raise EvaluationError(
                f"replay backend has no loss for {len(missing)} ref(s), "
                f"e.g. {missing[:3]}"
            )
        if progress:
            for i, _ in enumerate(refs, 1):
                progress(i, len(refs))
        return LossVector(
            refs=tuple(refs),
            losses=tuple(self._map[r] for r in refs),
            model_label=model_label or self.vector.model_label,
            model_digest=self.vector.model_digest,
            sequence_set=self.vector.sequence_set,
            manifest_sha256=self.vector.manifest_sha256,
            engine=dict(self.vector.engine),
            notes=("replayed",),
        )


class TorchBackend(ScoringBackend):
    """The real thing: transcribed from ``teutonic/evaluator/engine.py``.

    Mirrors the engine exactly -- ``model.model(...)`` then a chunked
    ``lm_head``, labels shifted by one, ``reduction="none"`` cross entropy
    accumulated in fp32, concatenated, summed once, divided by ``seq_len - 1``.

    Unverified until :mod:`evaluate_losses.parity` tier 3 has been run against
    the engine's own ``compute_per_sequence_loss`` on real weights.
    """

    def __init__(
        self,
        model_path: str | Path,
        loader,
        *,
        spec: EngineSpec | None = None,
        model_digest: str = "",
        device_map: str = "auto",
    ):
        self.spec = spec or EngineSpec()
        self.spec.require()
        self.model_path = str(model_path)
        self.loader = loader  # a FineWebLoader, supplying token sequences
        self.model_digest = model_digest
        self.device_map = device_map
        self._torch = None
        self._model = None

    # -- lazy dependencies -------------------------------------------------

    def _ensure_torch(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise BackendUnavailableError(
                "TorchBackend needs torch and transformers installed"
            ) from exc
        self._torch = (torch, F)
        return self._torch

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        torch, _ = self._ensure_torch()
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise BackendUnavailableError(
                "TorchBackend needs transformers installed"
            ) from exc
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.spec.attn_implementation,
            device_map=self.device_map,
        )
        self._model.eval()
        return self._model

    # -- scoring -----------------------------------------------------------

    def score_tokens(self, tokens: Sequence[int]) -> float:
        """Loss for one sequence. The engine's arithmetic, position by position."""
        torch, F = self._ensure_torch()
        model = self._ensure_model()
        spec = self.spec

        if len(tokens) != spec.seq_len:
            raise EvaluationError(
                f"expected {spec.seq_len} tokens, got {len(tokens)}; the engine "
                "refuses to truncate or pad evaluation sequences"
            )

        with torch.no_grad():
            input_device = next(model.parameters()).device
            input_ids = torch.tensor(
                [list(int(t) for t in tokens)], dtype=torch.long, device=input_device
            )
            if hasattr(model, "reset_state"):
                model.reset_state()
            hidden = model.model(input_ids, use_cache=spec.use_cache).last_hidden_state

            head_dev = next(model.lm_head.parameters()).device
            if hidden.device != head_dev:
                hidden = hidden.to(head_dev)
            labels_full = (
                input_ids if input_ids.device == head_dev else input_ids.to(head_dev)
            )

            n_pos = labels_full.size(1) - 1
            per_token = []
            for start in range(0, n_pos, spec.lm_head_chunk):
                end = min(start + spec.lm_head_chunk, n_pos)
                logits = model.lm_head(hidden[:, start:end, :])
                labels = labels_full[:, start + 1 : end + 1]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction="none",
                )
                # .float() before accumulating: fp32 even though the model is bf16.
                per_token.append(loss.reshape(1, -1).float())
                del logits
            # Concatenate then sum once, so lm_head_chunk cannot change the score.
            total = torch.cat(per_token, dim=1).sum(dim=1)
            return float((total / n_pos).float().cpu().item())

    def score(
        self,
        refs: Sequence[str],
        *,
        model_label: str,
        progress: Callable[[int, int], None] | None = None,
    ) -> LossVector:
        from fineweb_loader.refs import SequenceRef

        refs = list(refs)
        started = time.monotonic()
        losses: list[float] = []
        for i, ref in enumerate(refs, 1):
            parsed = SequenceRef.parse(ref)
            # allow_holdout: evaluation legitimately reads held-out sequences.
            rows = self.loader.sequences([parsed])
            losses.append(self.score_tokens(list(rows[0])))
            if progress:
                progress(i, len(refs))

        return LossVector(
            refs=tuple(refs),
            losses=tuple(losses),
            model_label=model_label,
            model_digest=self.model_digest,
            manifest_sha256=getattr(self.loader.manifest, "digest", ""),
            engine=self.spec.to_dict(),
            wall_time_s=round(time.monotonic() - started, 3),
        )

    def close(self) -> None:
        self._model = None
        if self._torch is not None:
            torch, _ = self._torch
            if torch.cuda.is_available():  # pragma: no cover - GPU only
                torch.cuda.empty_cache()


def expected_positions(seq_len: int = 2048) -> int:
    """Convenience re-export so callers need not import engine directly."""
    return n_positions(seq_len)
