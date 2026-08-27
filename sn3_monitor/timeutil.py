"""Timestamp parsing and duration helpers.

Teutonic emits ISO-8601 with a trailing ``Z``. Several fields are nullable, so
every parser here accepts ``None`` and returns ``None`` rather than raising.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$", re.IGNORECASE)

_UNIT_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}


def now() -> datetime:
    """Current time, always timezone-aware UTC."""
    return datetime.now(timezone.utc)


def parse_ts(value: str | None) -> datetime | None:
    """Parse a Teutonic timestamp. Returns None for null/blank/unparseable input."""
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_duration(value: str) -> timedelta:
    """Parse a duration such as ``30m``, ``24h`` or ``7d``."""
    match = _DURATION.match(value)
    if not match:
        raise ValueError(
            f"cannot parse duration {value!r}; expected forms like 30m, 24h, 7d"
        )
    amount, unit = match.groups()
    return timedelta(seconds=float(amount) * _UNIT_SECONDS[unit.lower()])


def age_of(value: str | None, *, reference: datetime | None = None) -> timedelta | None:
    """Age of a timestamp, or None if it could not be parsed."""
    parsed = parse_ts(value)
    if parsed is None:
        return None
    return (reference or now()) - parsed


def humanize(delta: timedelta | None) -> str:
    """Render a duration compactly: ``2d 3h``, ``58m``, ``12s``."""
    if delta is None:
        return "unknown"
    seconds = int(delta.total_seconds())
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{sign}{days}d {hours}h"
    if hours:
        return f"{sign}{hours}h {minutes}m"
    if minutes:
        return f"{sign}{minutes}m"
    return f"{sign}{seconds}s"


def iso(moment: datetime | None) -> str | None:
    """Render a datetime back to the ``...Z`` form Teutonic uses."""
    if moment is None:
        return None
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
