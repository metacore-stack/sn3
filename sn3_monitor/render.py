"""Terminal rendering helpers. Colour is used only on a real TTY."""

from __future__ import annotations

import os
import sys
from typing import Sequence

from .drift import Severity

_RESET = "\033[0m"
_COLOURS = {
    Severity.OK: "\033[32m",
    Severity.WARN: "\033[33m",
    Severity.STALE: "\033[33m",
    Severity.ABORT: "\033[31m",
}


def supports_colour(stream=None) -> bool:
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(stream, "isatty") and stream.isatty()


def paint(text: str, severity: Severity) -> str:
    if not supports_colour():
        return text
    return f"{_COLOURS.get(severity, '')}{text}{_RESET}"


def heading(text: str) -> str:
    return f"\n{text}\n{'─' * len(text)}"


def kv(pairs: Sequence[tuple[str, object]], indent: str = "  ") -> str:
    if not pairs:
        return ""
    width = max(len(str(k)) for k, _ in pairs)
    lines = [
        f"{indent}{str(key).ljust(width)}  {'—' if value is None else value}"
        for key, value in pairs
    ]
    return "\n".join(lines)


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], indent: str = "  ") -> str:
    if not rows:
        return f"{indent}(none)"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))
    out = [
        indent + "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)),
        indent + "  ".join("─" * w for w in widths),
    ]
    for row in rows:
        out.append(
            indent + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))
        )
    return "\n".join(out)


def pct(value: float | None, places: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{places}f}%"


def num(value: float | None, places: int = 4) -> str:
    return "—" if value is None else f"{value:.{places}f}"
