"""Memory-mapped reading of the pre-tokenized ``.npy`` shards.

numpy is optional. When it is installed we use it; otherwise a small pure-stdlib
reader parses the ``.npy`` header and mmaps the token block directly. The shards
are plain 1-D little-endian ``uint32`` arrays, which is well within what
``memoryview.cast`` can handle, so nothing is lost by the fallback except speed.
"""

from __future__ import annotations

import array
import ast
import mmap
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .errors import NpyFormatError

try:  # pragma: no cover - trivial import guard
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

NUMPY_AVAILABLE = _np is not None

MAGIC = b"\x93NUMPY"
# The competition's shards are uint32; anything else means the contract moved.
ACCEPTED_DESCRS = {"<u4", "|u4", "=u4", "u4"}


@dataclass(frozen=True)
class NpyHeader:
    """Parsed ``.npy`` header."""

    descr: str
    fortran_order: bool
    shape: tuple[int, ...]
    data_offset: int
    major: int
    minor: int

    @property
    def itemsize(self) -> int:
        return 4  # only uint32 shards are accepted

    @property
    def count(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total


def read_header(path: Path) -> NpyHeader:
    """Parse a ``.npy`` header without reading the data block."""
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != MAGIC:
            raise NpyFormatError(f"{path} is not a .npy file (bad magic {magic!r})")
        major = handle.read(1)[0]
        minor = handle.read(1)[0]
        if major == 1:
            length = int.from_bytes(handle.read(2), "little")
        elif major in (2, 3):
            length = int.from_bytes(handle.read(4), "little")
        else:
            raise NpyFormatError(f"{path}: unsupported .npy version {major}.{minor}")
        raw = handle.read(length)
        offset = handle.tell()

    try:
        header = ast.literal_eval(raw.decode("latin1").strip())
    except (ValueError, SyntaxError) as exc:
        raise NpyFormatError(f"{path}: unreadable .npy header ({exc})") from exc
    if not isinstance(header, dict):
        raise NpyFormatError(f"{path}: .npy header is not a dict")

    descr = str(header.get("descr", ""))
    shape = tuple(int(d) for d in header.get("shape", ()))
    if descr not in ACCEPTED_DESCRS:
        raise NpyFormatError(
            f"{path}: expected uint32 shard data, header declares descr={descr!r}"
        )
    if header.get("fortran_order"):
        raise NpyFormatError(f"{path}: Fortran-ordered shards are not supported")
    if len(shape) != 1:
        raise NpyFormatError(f"{path}: expected a 1-D token array, got shape {shape}")
    if descr.startswith("<") and sys.byteorder != "little":
        raise NpyFormatError(
            f"{path}: little-endian shard on a big-endian host; install numpy"
        )

    return NpyHeader(
        descr=descr,
        fortran_order=False,
        shape=shape,
        data_offset=offset,
        major=major,
        minor=minor,
    )


class Shard:
    """A memory-mapped shard, addressed as fixed-length sequences.

    Use as a context manager, or call :meth:`close` when finished. Sequences come
    back as numpy arrays when numpy is present and as ``array.array("I")`` copies
    otherwise; both index identically and both stay valid after the shard closes.
    """

    def __init__(self, path: Path, seq_len: int = 2048, *, prefer_numpy: bool = True):
        self.path = Path(path)
        self.seq_len = int(seq_len)
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.header = read_header(self.path)
        self._use_numpy = bool(prefer_numpy and NUMPY_AVAILABLE)
        self._handle = None
        self._mmap = None
        self._view = None
        self._array = None
        self._open()

    # -- lifecycle ---------------------------------------------------------

    def _open(self) -> None:
        if self._use_numpy:
            self._array = _np.load(self.path, mmap_mode="r")
            if self._array.dtype != _np.uint32:
                raise NpyFormatError(
                    f"{self.path}: expected uint32, numpy reports {self._array.dtype}"
                )
            return
        self._handle = self.path.open("rb")
        self._mmap = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        block = memoryview(self._mmap)[self.header.data_offset :]
        usable = (len(block) // 4) * 4
        self._view = block[:usable].cast("I")

    def close(self) -> None:
        self._array = None
        if self._view is not None:
            try:
                self._view.release()
            except BufferError:
                # A caller is still holding a derived buffer. Drop our reference
                # and let refcounting close the mapping once they release it.
                pass
            self._view = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "Shard":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- shape -------------------------------------------------------------

    @property
    def n_tokens(self) -> int:
        if self._array is not None:
            return int(self._array.shape[0])
        return len(self._view) if self._view is not None else 0

    @property
    def n_sequences(self) -> int:
        """Whole sequences only; a trailing partial sequence is not addressable."""
        return self.n_tokens // self.seq_len

    @property
    def has_ragged_tail(self) -> bool:
        return self.n_tokens % self.seq_len != 0

    # -- access ------------------------------------------------------------

    def sequence(self, index: int):
        """One sequence of ``seq_len`` tokens."""
        if index < 0:
            index += self.n_sequences
        if not 0 <= index < self.n_sequences:
            raise IndexError(
                f"sequence {index} out of range for {self.n_sequences} sequences"
            )
        start = index * self.seq_len
        stop = start + self.seq_len
        if self._array is not None:
            return self._array[start:stop]
        # A detached copy, not a live view: at 2048 uint32 (8 KiB) the cost is
        # trivial, and it means a returned sequence cannot outlive - or block -
        # the closing of the underlying mmap.
        return array.array("I", self._view[start:stop])

    def sequences(self, indices: Sequence[int]):
        """Several sequences, stacked when numpy is available."""
        ordered = list(indices)
        if self._array is not None:
            if not ordered:
                return _np.empty((0, self.seq_len), dtype=_np.uint32)
            rows = [self.sequence(i) for i in ordered]
            return _np.stack(rows)
        return [self.sequence(i) for i in ordered]

    def tolist(self, index: int) -> list[int]:
        """A sequence as plain Python ints, for tests and inspection."""
        return list(self.sequence(index))

    def __len__(self) -> int:
        return self.n_sequences

    def __repr__(self) -> str:
        backend = "numpy" if self._array is not None else "stdlib-mmap"
        return (
            f"<Shard {self.path.name} sequences={self.n_sequences} "
            f"tokens={self.n_tokens} backend={backend}>"
        )
