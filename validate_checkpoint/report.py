"""Check results and the overall verdict.

Every check records which validation *layer* it mirrors, because that determines
whether a failure is recoverable. Layers 1-2 run in the miner CLI before upload
and can be fixed and retried. Layers 3-4 run after ``ready``, which has already
consumed the hotkey permanently.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Iterable


class Layer(IntEnum):
    """Where the real system performs this check."""

    PREFLIGHT = 1  # miner/upload_model.py, before upload — recoverable
    MANIFEST = 2  # manifest construction and signing — recoverable
    INGEST = 3  # access/storage.py, after ready — FATAL
    EVALUATOR = 4  # evaluator/engine.py, before GPU load — FATAL
    LOCAL = 5  # our own sanity checks, not mirrored upstream

    @property
    def recoverable(self) -> bool:
        return self in (Layer.PREFLIGHT, Layer.MANIFEST, Layer.LOCAL)

    @property
    def label(self) -> str:
        return {
            Layer.PREFLIGHT: "preflight",
            Layer.MANIFEST: "manifest",
            Layer.INGEST: "ingest",
            Layer.EVALUATOR: "evaluator",
            Layer.LOCAL: "local",
        }[self]


class Status(IntEnum):
    PASS = 0
    SKIP = 1
    WARN = 2
    FAIL = 3

    @property
    def label(self) -> str:
        return {
            Status.PASS: "PASS",
            Status.SKIP: "SKIP",
            Status.WARN: "WARN",
            Status.FAIL: "FAIL",
        }[self]


@dataclass(frozen=True)
class Check:
    """One rule, and what happened when it was applied."""

    name: str
    layer: Layer
    status: Status
    detail: str = ""
    error_code: str = ""
    items: tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        return self.status is Status.FAIL

    @property
    def fatal(self) -> bool:
        """A failure that would be discovered only after ``ready``."""
        return self.failed and not self.layer.recoverable

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["layer"] = self.layer.label
        payload["status"] = self.status.label
        payload["items"] = list(self.items)
        return payload


@dataclass
class Report:
    """The full result for one checkpoint directory."""

    model_dir: str
    checks: list[Check] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    # -- assembly ----------------------------------------------------------

    def add(
        self,
        name: str,
        layer: Layer,
        status: Status,
        detail: str = "",
        *,
        error_code: str = "",
        items: Iterable[str] = (),
    ) -> Check:
        check = Check(
            name=name,
            layer=layer,
            status=status,
            detail=detail,
            error_code=error_code,
            items=tuple(items),
        )
        self.checks.append(check)
        return check

    def ok(self, name: str, layer: Layer, detail: str = "") -> Check:
        return self.add(name, layer, Status.PASS, detail)

    def fail(
        self, name: str, layer: Layer, detail: str, *, error_code: str = "",
        items: Iterable[str] = (),
    ) -> Check:
        return self.add(name, layer, Status.FAIL, detail, error_code=error_code, items=items)

    def skip(self, name: str, layer: Layer, detail: str) -> Check:
        return self.add(name, layer, Status.SKIP, detail)

    def warn(self, name: str, layer: Layer, detail: str, items: Iterable[str] = ()) -> Check:
        return self.add(name, layer, Status.WARN, detail, items=items)

    # -- verdict -----------------------------------------------------------

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.failed]

    @property
    def fatal_failures(self) -> list[Check]:
        return [c for c in self.checks if c.fatal]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status is Status.WARN]

    @property
    def skipped(self) -> list[Check]:
        return [c for c in self.checks if c.status is Status.SKIP]

    @property
    def would_reject(self) -> bool:
        return bool(self.failures)

    @property
    def determinate(self) -> bool:
        """True when nothing material was left unchecked."""
        return not self.skipped

    @property
    def error_codes(self) -> list[str]:
        seen = []
        for check in self.failures:
            if check.error_code and check.error_code not in seen:
                seen.append(check.error_code)
        return seen

    @property
    def verdict(self) -> str:
        if self.failures:
            return "WOULD BE REJECTED"
        if self.skipped:
            return "NO FAILURES (some checks skipped)"
        return "CLEAN"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_dir": self.model_dir,
            "verdict": self.verdict,
            "would_reject": self.would_reject,
            "determinate": self.determinate,
            "error_codes": self.error_codes,
            "counts": {
                "total": len(self.checks),
                "passed": sum(1 for c in self.checks if c.status is Status.PASS),
                "failed": len(self.failures),
                "warned": len(self.warnings),
                "skipped": len(self.skipped),
            },
            "context": self.context,
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
