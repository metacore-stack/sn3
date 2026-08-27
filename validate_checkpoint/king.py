"""Reference data for the current king.

A full king checkpoint is ~220 GB, but everything needed to validate a
challenger's *structure* is a few kilobytes: the published ``manifest.json``
(16 entries with path, size and sha256) plus ``config.json``. Both are fetched
by digest from the public model store, so the checks below run on a laptop.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import KingUnavailableError

MODEL_STORE_ROOT = "https://pub-0821d4e196224864af220294345fd141.r2.dev"
USER_AGENT = "validate-checkpoint/1.0 (+read-only public model store)"

# Small files worth fetching; the weight shards deliberately are not.
REFERENCE_FILES = ("manifest.json", "config.json", "model.safetensors.index.json")


@dataclass(frozen=True)
class KingFile:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class KingReference:
    """What a challenger is measured against."""

    digest: str
    files: dict[str, KingFile] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    index: dict[str, Any] = field(default_factory=dict)
    model_name: str = ""
    source: str = ""

    # -- accessors ---------------------------------------------------------

    @property
    def paths(self) -> list[str]:
        return sorted(self.files)

    @property
    def shard_names(self) -> list[str]:
        return sorted(p for p in self.files if p.endswith(".safetensors"))

    def sha256_for(self, path: str) -> str | None:
        entry = self.files.get(path)
        return entry.sha256 if entry else None

    def size_for(self, path: str) -> int | None:
        entry = self.files.get(path)
        return entry.size if entry else None

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files.values())

    @property
    def weight_map(self) -> dict[str, str]:
        return dict(self.index.get("weight_map") or {})

    # -- construction ------------------------------------------------------

    @classmethod
    def from_directory(cls, root: Path | str) -> "KingReference":
        """Build from a local king checkpoint, hashing nothing.

        Sizes come from the filesystem; hashes come from the published manifest
        if one is present alongside, since re-hashing 220 GB is not something to
        do implicitly.
        """
        root = Path(root).expanduser()
        if not root.is_dir():
            raise KingUnavailableError(f"{root} is not a directory")

        files: dict[str, KingFile] = {}
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in payload.get("files") or []:
                files[entry["path"]] = KingFile(
                    path=entry["path"],
                    sha256=str(entry.get("sha256", "")),
                    size=int(entry.get("size", 0)),
                )
        else:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(root).as_posix()
                    files[relative] = KingFile(relative, "", path.stat().st_size)

        return cls(
            digest="",
            files=files,
            config=_read_json(root / "config.json"),
            index=_read_json(root / "model.safetensors.index.json"),
            source=str(root),
        )

    @classmethod
    def from_digest(
        cls, digest: str, *, root: str = MODEL_STORE_ROOT, timeout: float = 60.0,
        cache_dir: Path | None = None,
    ) -> "KingReference":
        """Fetch the king's small reference files by digest.

        Downloads a few kilobytes, never a weight shard.
        """
        if not digest or len(digest) != 64:
            raise KingUnavailableError(f"expected a 64-character digest, got {digest!r}")
        base = f"{root.rstrip('/')}/models/sha256/{digest}"

        payloads: dict[str, Any] = {}
        for name in REFERENCE_FILES:
            cached = (cache_dir / digest / name) if cache_dir else None
            if cached and cached.is_file():
                payloads[name] = json.loads(cached.read_text(encoding="utf-8"))
                continue
            try:
                request = urllib.request.Request(
                    f"{base}/{name}", headers={"User-Agent": USER_AGENT}
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = response.read()
            except OSError as exc:
                if name == "manifest.json":
                    raise KingUnavailableError(
                        f"could not fetch {base}/{name}: {exc}"
                    ) from exc
                continue
            payloads[name] = json.loads(raw.decode("utf-8"))
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(raw)

        manifest = payloads.get("manifest.json") or {}
        files = {
            entry["path"]: KingFile(
                path=entry["path"],
                sha256=str(entry.get("sha256", "")),
                size=int(entry.get("size", 0)),
            )
            for entry in manifest.get("files") or []
        }
        if not files:
            raise KingUnavailableError(f"{base}/manifest.json listed no files")

        return cls(
            digest=digest,
            files=files,
            config=payloads.get("config.json") or {},
            index=payloads.get("model.safetensors.index.json") or {},
            model_name=str(manifest.get("model_name", "")),
            source=base,
        )

    @classmethod
    def resolve(
        cls,
        *,
        directory: Path | str | None = None,
        digest: str | None = None,
        cache_dir: Path | None = None,
    ) -> "KingReference":
        if directory:
            return cls.from_directory(directory)
        if digest:
            return cls.from_digest(digest, cache_dir=cache_dir)
        raise KingUnavailableError("supply either a king directory or a king digest")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}
