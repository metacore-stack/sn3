"""Retrieval of the public Teutonic documents, with mirror fallback.

Only the dashboard carries a heartbeat. The dataset manifest is a configuration
document that legitimately sits unchanged for days, so freshness is enforced on
the dashboard alone.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from .errors import FetchError, StaleDocumentError
from .timeutil import age_of, now, parse_ts

USER_AGENT = "sn3-monitor/1.0 (+read-only public dashboard poller)"

DASHBOARD_URLS: tuple[str, ...] = (
    "https://teutonic.ai/dashboard.json",
    "https://pub-fedac496355c4edc9aed57189e6e190f.r2.dev/dashboard.json",
)
DATASETS_URLS: tuple[str, ...] = (
    "https://teutonic.ai/datasets/manifest.json",
)
SHARD_MANIFEST_URL = (
    "https://pub-d923bc4e8fcb45f6b703bc750bcf8aa6.r2.dev/finewebedu/manifest.json"
)

DEFAULT_TIMEOUT = 20.0
DEFAULT_MAX_AGE = timedelta(minutes=30)


@dataclass(frozen=True)
class Document:
    """A fetched JSON document plus where and when it came from."""

    data: dict[str, Any]
    source: str
    fetched_at: datetime
    reported_at: datetime | None

    @property
    def age(self) -> timedelta | None:
        """How stale the document claims to be, by its own timestamp."""
        if self.reported_at is None:
            return None
        return self.fetched_at - self.reported_at


def _get(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    parsed = json.loads(payload.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise FetchError(f"{url} did not return a JSON object")
    return parsed


def fetch_json(
    urls: Sequence[str],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    timestamp_keys: Sequence[str] = (),
) -> Document:
    """Fetch the first URL that succeeds.

    Each URL is tried in order; the last failure is reported if all of them fail.
    """
    failures: list[str] = []
    for url in urls:
        try:
            data = _get(url, timeout)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            failures.append(f"{url}: {exc}")
            continue
        except json.JSONDecodeError as exc:
            failures.append(f"{url}: invalid JSON ({exc})")
            continue
        reported = None
        for key in timestamp_keys:
            reported = parse_ts(data.get(key))
            if reported is not None:
                break
        return Document(
            data=data, source=url, fetched_at=now(), reported_at=reported
        )
    raise FetchError("all sources failed:\n  " + "\n  ".join(failures))


def load_local(path: Path, *, timestamp_keys: Sequence[str] = ()) -> Document:
    """Load a previously downloaded document from disk."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FetchError(f"{path} does not exist") from exc
    except json.JSONDecodeError as exc:
        raise FetchError(f"{path} is not valid JSON: {exc}") from exc
    reported = None
    for key in timestamp_keys:
        reported = parse_ts(data.get(key))
        if reported is not None:
            break
    return Document(
        data=data, source=str(path), fetched_at=now(), reported_at=reported
    )


def fetch_dashboard(
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_age: timedelta | None = DEFAULT_MAX_AGE,
    local: Path | None = None,
) -> Document:
    """Fetch the dashboard and assert it is recent enough to act on."""
    keys = ("updated_at", "generated_at")
    document = (
        load_local(local, timestamp_keys=keys)
        if local is not None
        else fetch_json(DASHBOARD_URLS, timeout=timeout, timestamp_keys=keys)
    )
    if max_age is not None and document.age is not None and document.age > max_age:
        raise StaleDocumentError(
            f"dashboard from {document.source} reports "
            f"{document.reported_at} which is older than the {max_age} limit"
        )
    return document


def fetch_datasets(
    *, timeout: float = DEFAULT_TIMEOUT, local: Path | None = None
) -> Document:
    """Fetch the dataset manifest.

    No freshness assertion: ``generated_at`` here marks when the evaluation
    configuration was cut, not a heartbeat, and stays fixed for days at a time.
    """
    keys = ("generated_at",)
    if local is not None:
        return load_local(local, timestamp_keys=keys)
    return fetch_json(DATASETS_URLS, timeout=timeout, timestamp_keys=keys)


def dashboard_age(document: Document) -> str:
    """Human-readable staleness of a dashboard document."""
    from .timeutil import humanize

    return humanize(age_of(document.data.get("updated_at"), reference=document.fetched_at))
