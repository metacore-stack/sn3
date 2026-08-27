"""Writing checkpoints that are actually submittable, and resuming from them.

Two rules govern everything here:

1. **The six locked files must be byte-identical to the king's.**
   ``save_pretrained`` rewrites ``config.json`` -- reordered keys, a bumped
   ``transformers_version`` -- and the bytes then differ even though every value
   matches. That is ``GenesisContractMismatch``, three of the four packaging
   deaths in the live history. They are restored after every save.

2. **Training state never goes in the model directory.**
   Optimizer state, scheduler, RNG and the routing bias are large and are not
   part of the submission. A stray file in the uploaded tree is an *undeclared*
   object, which is ``ArtifactIntegrityError``. State is written to a sibling
   directory instead.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import CheckpointError, ResumeError

# chain.toml [seed.contract_files]; restored verbatim after every save.
LOCKED_FILES = (
    "chat_template.jinja.txt",
    "config.json",
    "configuration_mimo_v2.py",
    "modeling_mimo_v2.py",
    "tokenizer.json",
    "tokenizer_config.json",
)

STATE_FILENAME = "training_state.pt"
BIAS_FILENAME = "routing_bias.json"
META_FILENAME = "checkpoint.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SaveResult:
    model_dir: Path
    state_dir: Path
    step: int
    restored: tuple[str, ...]
    missing_from_king: tuple[str, ...]
    shape_mismatch: tuple[str, ...] = ()

    @property
    def contract_complete(self) -> bool:
        return bool(self.restored) and not self.missing_from_king

    @property
    def submittable(self) -> bool:
        """Whether this directory could be uploaded as it stands."""
        return self.contract_complete and not self.shape_mismatch


# Config keys that must agree before the king's config.json may be copied over
# a checkpoint. Restoring it onto a differently shaped model would produce a
# directory whose config describes one architecture and whose weights are
# another -- which loads as nonsense and fails at the evaluator.
SHAPE_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "n_routed_experts",
    "num_experts_per_tok",
    "vocab_size",
)


def architecture_matches(model_config, king_dir: Path) -> list[str]:
    """Shape keys where a model disagrees with the king's config.json."""
    king_config_path = Path(king_dir) / "config.json"
    if not king_config_path.is_file():
        return []
    try:
        king_config = json.loads(king_config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    differences: list[str] = []
    for key in SHAPE_KEYS:
        theirs = king_config.get(key)
        ours = getattr(model_config, key, None)
        if theirs is not None and ours is not None and theirs != ours:
            differences.append(f"{key}: model={ours} king={theirs}")
    return differences


def restore_locked_files(
    model_dir: Path, king_dir: Path | None, *, names: Iterable[str] = LOCKED_FILES
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Copy the locked files from the king over whatever ``save_pretrained`` wrote.

    Returns ``(restored, missing_from_king)``.
    """
    if king_dir is None:
        return (), tuple(names)
    king_dir = Path(king_dir)
    model_dir = Path(model_dir)
    restored: list[str] = []
    missing: list[str] = []
    for name in names:
        source = king_dir / name
        if not source.is_file():
            missing.append(name)
            continue
        shutil.copyfile(source, model_dir / name)
        restored.append(name)
    return tuple(restored), tuple(missing)


def save_checkpoint(
    *,
    model,
    step: int,
    output_dir: Path,
    king_dir: Path | None,
    optimizer=None,
    scheduler=None,
    balancer=None,
    metrics: dict[str, Any] | None = None,
    keep_state: bool = True,
    context=None,
) -> SaveResult:
    """Write a submittable model directory plus a separate resume state.

    Under FSDP the parameters live sharded across ranks, so the full state dict
    has to be gathered collectively -- every rank must call this -- and only
    rank zero writes. Saving a shard directly would produce a checkpoint nobody
    can load.
    """
    import torch

    from .distributed import DistributedContext, gather_full_state_dict, unwrap

    context = context or DistributedContext()
    output_dir = Path(output_dir)
    model_dir = output_dir / f"checkpoint-{step:06d}"
    state_dir = output_dir / f"state-{step:06d}"

    inner = unwrap(model)
    was_training = inner.training
    inner.eval()
    try:
        # Collective: all ranks participate, rank zero receives the full dict.
        state_dict = gather_full_state_dict(model, context)
    finally:
        if was_training:
            inner.train()

    if not context.is_main:
        # Non-main ranks took part in the gather and are done. They return the
        # paths so callers can log uniformly, but write nothing.
        return SaveResult(
            model_dir=model_dir,
            state_dir=state_dir,
            step=step,
            restored=(),
            missing_from_king=(),
            shape_mismatch=(),
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    was_training = inner.training
    inner.eval()
    try:
        inner.save_pretrained(
            model_dir, safe_serialization=True, state_dict=state_dict
        )
    except Exception as exc:  # noqa: BLE001
        raise CheckpointError(f"save_pretrained failed at step {step}: {exc}") from exc
    finally:
        if was_training:
            inner.train()

    # Only restore the king's locked files onto a model of the king's shape.
    # Copying them onto a miniature yields a config describing a 110B model
    # beside miniature weights, which then cannot even be reloaded.
    shape_mismatch: tuple[str, ...] = ()
    if king_dir is not None:
        shape_mismatch = tuple(architecture_matches(inner.config, king_dir))
    if shape_mismatch:
        restored, missing = (), tuple(LOCKED_FILES)
    else:
        restored, missing = restore_locked_files(model_dir, king_dir)

    if keep_state:
        state_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"step": step}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        payload["rng"] = {
            "torch": torch.get_rng_state(),
        }
        torch.save(payload, state_dir / STATE_FILENAME)
        if balancer is not None:
            (state_dir / BIAS_FILENAME).write_text(
                json.dumps(balancer.bias_state()), encoding="utf-8"
            )

    (model_dir / META_FILENAME).unlink(missing_ok=True)
    meta = {
        "step": step,
        "restored_locked_files": list(restored),
        "missing_from_king": list(missing),
        "shape_mismatch": list(shape_mismatch),
        "metrics": metrics or {},
    }
    # Metadata lives with the state, never in the model directory: anything
    # extra there is an undeclared object at ingest.
    if keep_state:
        (state_dir / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return SaveResult(
        model_dir=model_dir,
        state_dir=state_dir,
        step=step,
        restored=restored,
        missing_from_king=missing,
        shape_mismatch=shape_mismatch,
    )


def load_checkpoint_state(
    state_dir: Path,
    *,
    optimizer=None,
    scheduler=None,
    balancer=None,
    strict: bool = True,
) -> int:
    """Restore optimizer, scheduler, RNG and routing bias. Returns the step."""
    import torch

    state_dir = Path(state_dir)
    path = state_dir / STATE_FILENAME
    if not path.is_file():
        raise ResumeError(f"{path} does not exist")
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if optimizer is not None:
        if "optimizer" not in payload and strict:
            raise ResumeError(f"{path} has no optimizer state")
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    rng = payload.get("rng") or {}
    if "torch" in rng:
        torch.set_rng_state(rng["torch"].to(dtype=torch.uint8, device="cpu"))

    bias_path = state_dir / BIAS_FILENAME
    if balancer is not None and bias_path.is_file():
        balancer.load_bias_state(json.loads(bias_path.read_text(encoding="utf-8")))

    return int(payload.get("step", 0))


def latest_checkpoint(output_dir: Path) -> tuple[Path, Path] | None:
    """Newest ``(model_dir, state_dir)`` pair, or None."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    for model_dir in reversed(checkpoints):
        step = model_dir.name.split("-", 1)[-1]
        state_dir = output_dir / f"state-{step}"
        if (state_dir / STATE_FILENAME).is_file():
            return model_dir, state_dir
    return None


def prune_checkpoints(output_dir: Path, keep: int = 3) -> list[Path]:
    """Delete all but the newest ``keep`` checkpoints. A full one is ~220 GB."""
    output_dir = Path(output_dir)
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    removed: list[Path] = []
    for model_dir in checkpoints[: max(0, len(checkpoints) - keep)]:
        step = model_dir.name.split("-", 1)[-1]
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.rmtree(output_dir / f"state-{step}", ignore_errors=True)
        removed.append(model_dir)
    return removed


def verify_locked_files(model_dir: Path, expected: dict[str, str]) -> list[str]:
    """Names whose bytes do not match the expected hashes."""
    model_dir = Path(model_dir)
    wrong: list[str] = []
    for name, digest in sorted(expected.items()):
        path = model_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            wrong.append(name)
    return wrong
