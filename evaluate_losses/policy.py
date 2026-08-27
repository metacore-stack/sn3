"""Locating the validator's own decision function.

The statistics are never reimplemented here. ``paired_bootstrap_verdict`` is
loaded out of the cloned Teutonic repository and called directly, so a change
upstream changes our answer too.

A plain ``import teutonic.evaluation.policy`` pulls the package ``__init__``,
which needs httpx and the rest of the validator's dependency tree. Loading the
single file by path avoids all of that -- the module itself imports only numpy
and the standard library.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from .errors import PolicyUnavailableError

ENV_REPO = "TEUTONIC_REPO"
RELATIVE_POLICY = Path("teutonic") / "evaluation" / "policy.py"

SEARCH_PATHS: tuple[Path, ...] = (
    Path.home() / "Documents" / "teutonic",
    Path.cwd() / "teutonic",
    Path.cwd().parent / "teutonic",
)


def candidate_paths() -> list[Path]:
    """Where policy.py might live, most explicit first."""
    found: list[Path] = []
    override = os.environ.get(ENV_REPO)
    if override:
        found.append(Path(override).expanduser())
    found.extend(SEARCH_PATHS)
    return found


@lru_cache(maxsize=1)
def load_policy(repo: str | None = None) -> ModuleType:
    """Load the validator's policy module, or explain precisely why we cannot."""
    roots = [Path(repo).expanduser()] if repo else candidate_paths()
    tried: list[str] = []
    for root in roots:
        path = root / RELATIVE_POLICY
        tried.append(str(path))
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_teutonic_policy", path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ImportError as exc:  # numpy missing, most likely
            raise PolicyUnavailableError(
                f"found {path} but could not import it: {exc}. "
                "The policy module needs numpy."
            ) from exc
        if not hasattr(module, "paired_bootstrap_verdict"):
            continue
        return module

    raise PolicyUnavailableError(
        "could not locate the validator's evaluation policy. Clone "
        "https://github.com/unarbos/teutonic and either place it at "
        f"~/Documents/teutonic or set {ENV_REPO}. Looked in:\n  "
        + "\n  ".join(tried)
    )


def paired_bootstrap_verdict(
    king_losses: Sequence[float],
    challenger_losses: Sequence[float],
    *,
    bootstrap_seed: int,
    n_bootstrap: int,
    alpha: float,
    delta_threshold: float,
    repo: str | None = None,
) -> dict[str, Any]:
    """Call the validator's function verbatim.

    Note ``accepted`` uses a strict ``>``: an LCB of exactly the threshold is a
    rejection.
    """
    module = load_policy(repo)
    return module.paired_bootstrap_verdict(
        list(king_losses),
        list(challenger_losses),
        bootstrap_seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        delta_threshold=delta_threshold,
    )


def policy_source(repo: str | None = None) -> str:
    """Path of the policy file actually in use, for provenance."""
    return getattr(load_policy(repo), "__file__", "<unknown>")


def available(repo: str | None = None) -> bool:
    try:
        load_policy(repo)
    except PolicyUnavailableError:
        return False
    return True
