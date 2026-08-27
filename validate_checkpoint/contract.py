"""The submission contract, read from chain.toml and the validator's own source.

Nothing here is hard-coded that the repository already states. The six contract
hashes, the locked config keys and the permitted code files all come from the
clone, so a change upstream changes what this package enforces.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ContractError

# access/storage.py:25 -- the hard ceiling on an upload.
MAX_UPLOAD_BYTES = 250_000_000_000

# miner/upload_model.py:66 -- the CLI creates this; the miner must not supply it.
RESERVED_FILENAME = "manifest.json"

# archs/mimo/__init__.py -- the only Python files a snapshot may contain.
DEFAULT_ALLOWED_CODE_FILES = ("configuration_mimo_v2.py", "modeling_mimo_v2.py")
DEFAULT_MODEL_TYPE = "mimo_v2"

# evaluation/policy.py:13 -- checked for every generation, before extra_lock_keys.
FALLBACK_GENERIC_LOCK_KEYS = (
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "model_type",
    "tie_word_embeddings",
    "rope_theta",
    "max_position_embeddings",
    "max_seq_len",
)

SEARCH_ROOTS = (
    Path.home() / "Documents" / "teutonic",
    Path.cwd() / "teutonic",
    Path.cwd().parent / "teutonic",
)


def find_chain_toml(explicit: Path | str | None = None) -> Path:
    """Locate chain.toml, preferring an explicit path."""
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_dir():
            path = path / "chain.toml"
        if not path.is_file():
            raise ContractError(f"{path} does not exist")
        return path
    for root in SEARCH_ROOTS:
        candidate = root / "chain.toml"
        if candidate.is_file():
            return candidate
    raise ContractError(
        "could not find chain.toml; clone https://github.com/unarbos/teutonic "
        "or pass --chain"
    )


def generic_lock_keys() -> tuple[str, ...]:
    """The validator's own GENERIC_CONFIG_LOCK_KEYS, if its policy is reachable."""
    try:
        from evaluate_losses.policy import load_policy

        keys = getattr(load_policy(), "GENERIC_CONFIG_LOCK_KEYS", None)
        if keys:
            return tuple(keys)
    except Exception:
        pass
    return FALLBACK_GENERIC_LOCK_KEYS


@dataclass(frozen=True)
class Contract:
    """Everything a submission is judged against, structurally."""

    path: Path
    name: str
    seed_repo: str
    repo_pattern: str
    arch_module: str
    contract_files: dict[str, str]
    extra_lock_keys: tuple[str, ...]
    generic_lock_keys: tuple[str, ...]
    allowed_code_files: tuple[str, ...] = DEFAULT_ALLOWED_CODE_FILES
    model_type: str = DEFAULT_MODEL_TYPE
    eval_n: int | None = None
    delta_threshold: float | None = None
    dataset_label: str | None = None
    initial_weight_uids: tuple[int, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def locked_config_keys(self) -> tuple[str, ...]:
        """Every config key that must match the king, generic plus arch-specific."""
        return tuple(sorted(set(self.generic_lock_keys) | set(self.extra_lock_keys)))

    @property
    def n_locked_keys(self) -> int:
        return len(self.locked_config_keys)

    def name_matches(self, model_name: str) -> bool:
        """Whether a submission name satisfies ``repo_pattern``.

        The pattern is owner-qualified (``^[^/]+/…``) while the miner CLI's
        ``--name`` takes a bare name, so a bare name is retested against a
        synthetically owner-qualified form. Splitting the pattern on ``/``
        would be wrong -- the ``[^/]`` character class contains one.
        """
        if not self.repo_pattern:
            return True
        if re.match(self.repo_pattern, model_name):
            return True
        if "/" in model_name:
            return False
        return bool(re.match(self.repo_pattern, f"owner/{model_name}"))

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Contract":
        chain_path = find_chain_toml(path)
        try:
            data = tomllib.loads(chain_path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError) as exc:
            raise ContractError(f"{chain_path} is not readable TOML: {exc}") from exc

        chain = data.get("chain") or {}
        arch = data.get("arch") or {}
        seed = data.get("seed") or {}
        evaluation = data.get("evaluation") or {}
        contract_files = dict(seed.get("contract_files") or {})
        if not contract_files:
            raise ContractError(f"{chain_path} declares no [seed.contract_files]")

        return cls(
            path=chain_path,
            name=str(chain.get("name", "")),
            seed_repo=str(chain.get("seed_repo", "")),
            repo_pattern=str(chain.get("repo_pattern", "")),
            arch_module=str(arch.get("module", "")),
            contract_files=contract_files,
            extra_lock_keys=tuple(arch.get("extra_lock_keys") or ()),
            generic_lock_keys=generic_lock_keys(),
            eval_n=evaluation.get("n"),
            delta_threshold=evaluation.get("delta_threshold"),
            dataset_label=evaluation.get("dataset_label"),
            initial_weight_uids=tuple(seed.get("initial_weight_uids") or ()),
            raw=data,
        )
