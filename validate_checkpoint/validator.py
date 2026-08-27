"""Orchestration: run every rule against a directory and produce one report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path

from . import checks
from .contract import Contract
from .king import KingReference
from .report import Report


@dataclass(frozen=True)
class Options:
    """What to run. The expensive checks are opt-in."""

    hash_shards: bool = False
    finite: bool = False
    model_name: str | None = None
    ledger: Any = None

    @classmethod
    def thorough(cls, model_name: str | None = None) -> "Options":
        """Everything, including the two slow checks. Use before ``ready``."""
        return cls(hash_shards=True, finite=True, model_name=model_name)


def validate(
    model_dir: Path | str,
    *,
    contract: Contract,
    king: KingReference | None = None,
    options: Options | None = None,
) -> Report:
    """Apply every rule, in the order the real system would encounter them."""
    root = Path(model_dir).expanduser()
    options = options or Options()

    report = Report(model_dir=str(root))
    report.context = {
        "chain_toml": str(contract.path),
        "generation": contract.name,
        "locked_config_keys": contract.n_locked_keys,
        "contract_files": len(contract.contract_files),
        "king": (king.source if king else None),
        "king_digest": (king.digest if king else None),
        "hash_shards": options.hash_shards,
        "finite_scan": options.finite,
    }

    # Layer 1 -- what the uploader itself refuses.
    files = checks.check_tree(report, root)
    if not files:
        return report

    checks.check_size(report, files)
    checks.check_name(report, contract, options.model_name)

    # Layer 3 -- ingest, where a failure costs the hotkey.
    checks.check_contract_files(report, root, contract)
    checks.check_inventory(report, root, files, king)

    # Layer 4 -- what the evaluator refuses before touching a GPU.
    checks.check_allowed_code_files(report, root, contract)
    headers = checks.load_headers(report, root)
    if headers:
        checks.check_header_structure(report, headers)
        checks.check_mtp(report, headers)
        checks.check_dtypes(report, headers)
        checks.check_index(report, root, headers)
    checks.check_config_lock(report, root, contract, king)
    checks.check_weights_changed(report, root, king, hash_shards=options.hash_shards)
    digest = checks.check_reuse_limit(
        report, root, options.ledger, hash_shards=options.hash_shards
    )
    if digest:
        report.context["safetensors_digest"] = digest
    checks.check_finite(report, headers, enabled=options.finite)

    return report
