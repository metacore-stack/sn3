"""Pure-stdlib safetensors header reading, plus a streaming non-finite scan.

The format is simple enough that no dependency is warranted:

    [8 bytes little-endian uint64 header length N]
    [N bytes UTF-8 JSON header]
    [tensor data]

The header maps every tensor name to ``{"dtype", "shape", "data_offsets"}``,
with offsets relative to the start of the data block. Reading it costs a few
kilobytes regardless of how large the shard is, which is what makes structural
validation of a 220 GB checkpoint a sub-second operation.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .errors import SafetensorsFormatError

HEADER_PREFIX_BYTES = 8
MAX_HEADER_BYTES = 200 * 1024 * 1024  # generous; guards against a corrupt length

# Bytes per element, for the dtypes a checkpoint of this kind can contain.
DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}


@dataclass(frozen=True)
class TensorInfo:
    """One tensor's header entry."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def n_bytes(self) -> int:
        return self.end - self.begin

    @property
    def n_elements(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    @property
    def expected_bytes(self) -> int | None:
        size = DTYPE_SIZES.get(self.dtype)
        return None if size is None else self.n_elements * size

    def is_mtp(self) -> bool:
        """Multi-token-prediction weights, which the evaluator rejects outright.

        Matched as a case-insensitive substring, exactly as the engine does.
        """
        return "mtp" in self.name.lower()


@dataclass(frozen=True)
class SafetensorsHeader:
    """A parsed shard header."""

    path: Path
    tensors: tuple[TensorInfo, ...]
    metadata: dict
    data_offset: int
    file_size: int

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.tensors]

    @property
    def dtypes(self) -> set[str]:
        return {t.dtype for t in self.tensors}

    def by_name(self) -> dict[str, TensorInfo]:
        return {t.name: t for t in self.tensors}

    def problems(self) -> list[str]:
        """Structural faults detectable from the header alone."""
        found: list[str] = []
        for tensor in self.tensors:
            if tensor.begin < 0 or tensor.end < tensor.begin:
                found.append(f"{tensor.name}: invalid data_offsets [{tensor.begin}, {tensor.end}]")
                continue
            absolute_end = self.data_offset + tensor.end
            if absolute_end > self.file_size:
                found.append(
                    f"{tensor.name}: data ends at byte {absolute_end} but the file is "
                    f"{self.file_size} bytes — shard is truncated"
                )
            expected = tensor.expected_bytes
            if expected is not None and expected != tensor.n_bytes:
                found.append(
                    f"{tensor.name}: shape {tensor.shape} of {tensor.dtype} needs "
                    f"{expected} bytes but the header reserves {tensor.n_bytes}"
                )
        return found


def read_header(path: Path) -> SafetensorsHeader:
    """Parse a shard header without touching tensor data."""
    path = Path(path)
    size = path.stat().st_size
    if size < HEADER_PREFIX_BYTES:
        raise SafetensorsFormatError(f"{path.name}: too small to be a safetensors file")

    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(HEADER_PREFIX_BYTES))
        if length <= 0 or length > MAX_HEADER_BYTES:
            raise SafetensorsFormatError(
                f"{path.name}: implausible header length {length}"
            )
        if HEADER_PREFIX_BYTES + length > size:
            raise SafetensorsFormatError(
                f"{path.name}: header claims {length} bytes but the file is {size}"
            )
        raw = handle.read(length)

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SafetensorsFormatError(f"{path.name}: unreadable header ({exc})") from exc
    if not isinstance(payload, dict):
        raise SafetensorsFormatError(f"{path.name}: header is not a JSON object")

    metadata = payload.pop("__metadata__", {}) or {}
    tensors: list[TensorInfo] = []
    for name, entry in payload.items():
        if not isinstance(entry, dict):
            raise SafetensorsFormatError(f"{path.name}: malformed entry for {name!r}")
        offsets = entry.get("data_offsets") or [0, 0]
        try:
            begin, end = int(offsets[0]), int(offsets[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise SafetensorsFormatError(
                f"{path.name}: bad data_offsets for {name!r}"
            ) from exc
        tensors.append(
            TensorInfo(
                name=name,
                dtype=str(entry.get("dtype", "")),
                shape=tuple(int(d) for d in (entry.get("shape") or ())),
                begin=begin,
                end=end,
            )
        )

    return SafetensorsHeader(
        path=path,
        tensors=tuple(sorted(tensors, key=lambda t: t.name)),
        metadata=dict(metadata),
        data_offset=HEADER_PREFIX_BYTES + length,
        file_size=size,
    )


def iter_headers(root: Path) -> Iterator[SafetensorsHeader]:
    """Every ``.safetensors`` header under ``root``, in sorted order."""
    for path in sorted(Path(root).rglob("*.safetensors")):
        yield read_header(path)


# -- non-finite scan --------------------------------------------------------

# bfloat16 is the top 16 bits of a float32: 1 sign, 8 exponent, 7 mantissa.
# Exponent all-ones means inf or nan, so masking with 0x7F80 is a complete test.
_BF16_EXPONENT_MASK = 0x7F80
_F16_EXPONENT_MASK = 0x7C00

_CHUNK = 4 * 1024 * 1024


def scan_nonfinite(
    header: SafetensorsHeader, *, limit: int = 8
) -> list[tuple[str, int]]:
    """Find tensors containing NaN or Inf.

    Returns ``(tensor_name, count)`` pairs, stopping after ``limit`` offending
    tensors. Streams the file rather than loading it, so a 30 GB shard costs
    disk bandwidth and no memory.

    Only BF16 and F16 are scanned; other dtypes are skipped rather than guessed
    at, and skipping is reported by the caller.
    """
    found: list[tuple[str, int]] = []
    with header.path.open("rb") as handle:
        for tensor in header.tensors:
            if len(found) >= limit:
                break
            if tensor.dtype == "BF16":
                mask = _BF16_EXPONENT_MASK
            elif tensor.dtype == "F16":
                mask = _F16_EXPONENT_MASK
            else:
                continue
            handle.seek(header.data_offset + tensor.begin)
            remaining = tensor.n_bytes
            hits = 0
            while remaining > 0:
                block = handle.read(min(_CHUNK, remaining))
                if not block:
                    break
                remaining -= len(block)
                usable = len(block) - (len(block) % 2)
                for value in struct.unpack(f"<{usable // 2}H", block[:usable]):
                    if value & mask == mask:
                        hits += 1
            if hits:
                found.append((tensor.name, hits))
    return found


def scannable_dtypes(header: SafetensorsHeader) -> tuple[set[str], set[str]]:
    """Split a header's dtypes into ``(scannable, skipped)``."""
    scannable = {d for d in header.dtypes if d in {"BF16", "F16"}}
    return scannable, header.dtypes - scannable
