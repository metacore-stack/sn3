"""Importing the locked MiMo architecture code without modifying it.

``modeling_mimo_v2.py`` opens with ``from .configuration_mimo_v2 import
MiMoV2Config`` -- a *relative* import, so loading the file directly by path
fails. Instead a synthetic package is registered whose ``__path__`` points at the
directory holding both files, and the two modules are imported as its submodules.

Nothing here writes to the checkpoint. The files are hash-pinned in chain.toml
and must reach the validator byte-identical.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ArchSourceError

MODEL_STORE_ROOT = "https://pub-0821d4e196224864af220294345fd141.r2.dev"
USER_AGENT = "mimo-adapter/1.0 (+read-only public model store)"

ARCH_FILES = ("configuration_mimo_v2.py", "modeling_mimo_v2.py")
CONFIG_FILE = "config.json"

SEARCH_ROOTS = (
    Path.home() / "Documents" / "sn3" / "state" / "arch",
    Path.cwd() / "arch",
)


@dataclass(frozen=True)
class ArchModules:
    """The imported architecture, plus where it came from."""

    configuration: types.ModuleType
    modeling: types.ModuleType
    directory: Path
    package: str

    @property
    def config_cls(self):
        return getattr(self.configuration, "MiMoV2Config")

    @property
    def causal_lm_cls(self):
        return getattr(self.modeling, "MiMoV2ForCausalLM")

    @property
    def gate_cls(self):
        return getattr(self.modeling, "MiMoV2MoEGate")

    @property
    def moe_cls(self):
        return getattr(self.modeling, "MiMoV2MoE")


def fetch_arch(
    digest: str,
    destination: Path | None = None,
    *,
    root: str = MODEL_STORE_ROOT,
    timeout: float = 60.0,
    include_config: bool = True,
) -> Path:
    """Download the architecture files for a king by digest.

    Three small files -- about 42 KB -- not the weights.
    """
    if not digest or len(digest) != 64:
        raise ArchSourceError(f"expected a 64-character digest, got {digest!r}")
    destination = Path(destination or (SEARCH_ROOTS[0] / digest)).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    base = f"{root.rstrip('/')}/models/sha256/{digest}"

    wanted = list(ARCH_FILES) + ([CONFIG_FILE] if include_config else [])
    for name in wanted:
        target = destination / name
        if target.is_file():
            continue
        request = urllib.request.Request(
            f"{base}/{name}", headers={"User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                target.write_bytes(response.read())
        except OSError as exc:
            raise ArchSourceError(f"could not fetch {base}/{name}: {exc}") from exc
    return destination


def find_arch_directory(explicit: Path | str | None = None) -> Path:
    """Locate a directory containing both architecture files.

    An explicit path is never a hint: if it does not hold both files this
    raises rather than searching elsewhere. Falling back would silently load a
    different king's architecture, and every parity result after that would
    describe the wrong model.
    """
    if explicit:
        directory = Path(explicit).expanduser()
        if all((directory / name).is_file() for name in ARCH_FILES):
            return directory
        missing = [name for name in ARCH_FILES if not (directory / name).is_file()]
        raise ArchSourceError(
            f"{directory} does not contain {', '.join(missing)}. "
            "Refusing to fall back to another directory — an explicit --arch "
            "must be the architecture you mean."
        )

    candidates: list[Path] = []
    for root in SEARCH_ROOTS:
        candidates.append(root)
        if root.is_dir():
            candidates.extend(sorted(p for p in root.iterdir() if p.is_dir()))

    for candidate in candidates:
        if all((candidate / name).is_file() for name in ARCH_FILES):
            return candidate
    raise ArchSourceError(
        "could not find configuration_mimo_v2.py and modeling_mimo_v2.py. "
        "Pass --arch <dir>, or --king-digest to download them."
    )


def load_arch(
    directory: Path | str | None = None, *, package: str = "_mimo_arch"
) -> ArchModules:
    """Import the architecture as a synthetic package.

    The package name is suffixed per directory so two different kings can be
    loaded in one process without colliding.
    """
    directory = find_arch_directory(directory)
    suffix = abs(hash(str(directory.resolve()))) % (10**8)
    package_name = f"{package}_{suffix}"

    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(directory)]  # type: ignore[attr-defined]
        pkg.__package__ = package_name
        sys.modules[package_name] = pkg

    try:
        configuration = importlib.import_module(f"{package_name}.configuration_mimo_v2")
        modeling = importlib.import_module(f"{package_name}.modeling_mimo_v2")
    except ImportError as exc:
        raise ArchSourceError(
            f"could not import the architecture from {directory}: {exc}. "
            "torch and transformers must be installed."
        ) from exc

    for attr in ("MiMoV2Config",):
        if not hasattr(configuration, attr):
            raise ArchSourceError(f"{directory} configuration module has no {attr}")
    for attr in ("MiMoV2ForCausalLM", "MiMoV2MoEGate", "MiMoV2MoE"):
        if not hasattr(modeling, attr):
            raise ArchSourceError(f"{directory} modeling module has no {attr}")

    return ArchModules(
        configuration=configuration,
        modeling=modeling,
        directory=directory,
        package=package_name,
    )


def read_reference_config(directory: Path | str | None = None) -> dict[str, Any]:
    """The king's ``config.json`` if it sits alongside the architecture files."""
    import json

    directory = find_arch_directory(directory)
    path = directory / CONFIG_FILE
    if not path.is_file():
        raise ArchSourceError(
            f"{path} not found; fetch it with --king-digest so the miniature can "
            "inherit the real routing settings"
        )
    return json.loads(path.read_text(encoding="utf-8"))
