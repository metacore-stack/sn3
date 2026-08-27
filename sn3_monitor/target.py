"""The pinned target: an immutable description of what you are training against.

Every experiment should reference a snapshot id. "Trained against the king" is
not reproducible; "trained against 20260826T2114Z-c345e657" is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .fetch import Document
from .timeutil import iso, now, parse_ts

SNAPSHOT_VERSION = 1


@dataclass(frozen=True)
class DataSource:
    """One evaluation corpus named by the dataset manifest."""

    name: str | None
    manifest_sha256: str | None
    tokenizer: str | None
    sequence_length: int | None
    total_shards: int | None
    total_tokens: int | None
    proportion: float | None

    @classmethod
    def from_manifest(cls, entry: dict[str, Any]) -> "DataSource":
        return cls(
            name=entry.get("name"),
            manifest_sha256=entry.get("manifest_sha256"),
            tokenizer=entry.get("tokenizer"),
            sequence_length=entry.get("sequence_length"),
            total_shards=entry.get("total_shards"),
            total_tokens=entry.get("total_tokens"),
            proportion=entry.get("proportion"),
        )


@dataclass(frozen=True)
class Target:
    """Everything that must hold constant for an experiment to stay valid."""

    snapshot_id: str
    snapshot_version: int
    pinned_at: str

    # Identity of the model you must beat.
    king_digest: str | None
    king_reign: int | None
    king_repo: str | None
    king_uid: int | None
    king_hotkey: str | None
    king_crowned_at: str | None
    king_loss: float | None

    # The competition contract.
    generation: str | None
    competition: str | None
    netuid: int | None
    seed_repo: str | None

    # Acceptance rule. Present in two documents; both are recorded so a
    # divergence between them is visible rather than silently resolved.
    delta_from_king: float | None
    delta_from_datasets: float | None
    eval_n: int | None

    # Evaluation data contract.
    dataset_version: str | None
    sources: tuple[DataSource, ...] = ()

    # Provenance.
    dashboard_source: str | None = None
    datasets_source: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def delta(self) -> float | None:
        """The threshold to plan against; prefers the dataset manifest."""
        if self.delta_from_datasets is not None:
            return self.delta_from_datasets
        return self.delta_from_king

    @property
    def delta_disagrees(self) -> bool:
        """True when the two published thresholds do not match."""
        if self.delta_from_king is None or self.delta_from_datasets is None:
            return False
        return self.delta_from_king != self.delta_from_datasets

    @property
    def short_digest(self) -> str:
        return (self.king_digest or "unknown")[:8]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = [asdict(source) for source in self.sources]
        payload["notes"] = list(self.notes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Target":
        data = dict(payload)
        data["sources"] = tuple(
            DataSource(**source) for source in data.get("sources", [])
        )
        data["notes"] = tuple(data.get("notes", []))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_live(cls, dashboard: Document, datasets: Document) -> "Target":
        """Build a target from a freshly fetched dashboard and dataset manifest."""
        board = dashboard.data
        king = board.get("king") or {}
        chain = board.get("chain") or {}
        manifest = datasets.data

        digest = king.get("king_digest") or king.get("model_digest")
        pinned = now()
        stamp = pinned.strftime("%Y%m%dT%H%M%SZ")
        snapshot_id = f"{stamp}-{(digest or 'unknown')[:8]}"

        sources = tuple(
            DataSource.from_manifest(entry)
            for entry in (manifest.get("sources") or [])
            if isinstance(entry, dict)
        )

        notes: list[str] = []
        delta_king = king.get("delta")
        delta_data = manifest.get("delta_threshold")
        if delta_king is not None and delta_data is not None and delta_king != delta_data:
            notes.append(
                f"delta disagreement: king.delta={delta_king} "
                f"datasets.delta_threshold={delta_data}"
            )
        if parse_ts(king.get("crowned_at")) is None:
            notes.append("king.crowned_at missing or unparseable")

        return cls(
            snapshot_id=snapshot_id,
            snapshot_version=SNAPSHOT_VERSION,
            pinned_at=iso(pinned) or "",
            king_digest=digest,
            king_reign=king.get("reign_number"),
            king_repo=king.get("model_repo"),
            king_uid=king.get("uid"),
            king_hotkey=king.get("hotkey"),
            king_crowned_at=king.get("crowned_at"),
            king_loss=king.get("avg_challenger_loss"),
            generation=chain.get("generation"),
            competition=chain.get("competition"),
            netuid=chain.get("netuid"),
            seed_repo=chain.get("seed_repo"),
            delta_from_king=delta_king,
            delta_from_datasets=delta_data,
            eval_n=manifest.get("eval_n"),
            dataset_version=manifest.get("config_version"),
            sources=sources,
            dashboard_source=dashboard.source,
            datasets_source=datasets.source,
            notes=tuple(notes),
        )
