"""Read and validate the header of a FAISS index file without importing faiss.

Layout (faiss-cpu 1.16.0, little endian): 4-byte fourcc, int32 d, int64 ntotal,
two int64 dummies, uint8 is_trained, int32 metric_type and, when metric_type > 1,
a float32 metric_arg.  Flat indexes continue with a uint64 element count followed
by ``count`` float32 values, so their exact file size is known.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import MalformedStoreError
from .limits import Limits

# fourcc -> LangChain-facing index name.  Anything else is refused before faiss reads it.
FOURCC_NAMES: dict[str, str] = {
    "IxF2": "IndexFlatL2",
    "IxFI": "IndexFlatIP",
    "IxFl": "IndexFlat",
    "IHNf": "IndexHNSWFlat",
    "IwFl": "IndexIVFFlat",
    "IxMp": "IndexIDMap",
    "IxM2": "IndexIDMap2",
    "IxPq": "IndexPQ",
}
_FLAT = {"IxF2", "IxFI", "IxFl"}
_ID_MAPPED = {"IxMp", "IxM2"}
_METRICS = {0: "inner_product", 1: "L2", 2: "L1", 3: "Linf", 4: "Lp", 20: "Canberra", 21: "BrayCurtis", 22: "JensenShannon", 23: "Jaccard"}
_BASE = struct.Struct("<4siqqqBi")  # fourcc, d, ntotal, dummy, dummy, is_trained, metric_type
_MAX_DIM = 65536


@dataclass(frozen=True)
class FaissHeader:
    """Facts read from the first bytes of ``index.faiss``."""

    fourcc: str
    index_type: str
    d: int
    ntotal: int
    is_trained: bool
    metric_type: int
    metric_arg: float | None
    file_size: int

    @property
    def metric_name(self) -> str:
        return _METRICS.get(self.metric_type, str(self.metric_type))

    @property
    def id_mapped(self) -> bool:
        return self.fourcc in _ID_MAPPED

    def to_dict(self) -> dict[str, Any]:
        return {
            "fourcc": self.fourcc,
            "index_type": self.index_type,
            "d": self.d,
            "ntotal": self.ntotal,
            "is_trained": self.is_trained,
            "metric_type": self.metric_type,
            "metric_name": self.metric_name,
            "metric_arg": self.metric_arg,
            "file_size": self.file_size,
        }


def read_faiss_header(path: str | Path, *, limits: Limits = Limits()) -> FaissHeader:
    """Validate and return the header of a FAISS index file.

    Raises :class:`MalformedStoreError` (``code`` is ``unsupported-index-type`` or
    ``bad-faiss-header``) before anything is handed to ``faiss.read_index``.
    """
    p = Path(path)
    try:
        file_size = p.stat().st_size
        with p.open("rb") as fh:
            head = fh.read(_BASE.size + 4 + 8)
    except OSError as exc:
        raise MalformedStoreError(f"cannot read {p}: {exc.strerror or exc}", code="missing-file",
                                  params={"path": str(p)}) from None
    if file_size > limits.max_faiss_bytes:
        raise MalformedStoreError(
            f"{p.name} is {file_size} bytes, above the limit of {limits.max_faiss_bytes}. "
            "Raise Limits.max_faiss_bytes if this size is expected.",
            code="limit-exceeded", params={"limit": "max_faiss_bytes"},
        )
    if len(head) < _BASE.size:
        raise MalformedStoreError(f"{p.name} is truncated ({file_size} bytes); not a FAISS index.", code="bad-faiss-header")
    raw_fourcc, d, ntotal, _d1, _d2, trained, metric = _BASE.unpack_from(head)
    fourcc = raw_fourcc.decode("latin1")
    if fourcc not in FOURCC_NAMES:
        raise MalformedStoreError(
            f"{p.name} has index type {fourcc!r}, which is not on the supported list "
            f"({', '.join(sorted(FOURCC_NAMES))}). It was not passed to faiss.",
            code="unsupported-index-type",
        )
    if not 1 <= d <= _MAX_DIM:
        raise MalformedStoreError(f"{p.name} claims dimension d={d} (allowed 1..{_MAX_DIM}).", code="bad-faiss-header")
    if not 0 <= ntotal <= limits.max_vectors:
        raise MalformedStoreError(f"{p.name} claims ntotal={ntotal} (allowed 0..{limits.max_vectors}).", code="bad-faiss-header")
    if trained not in (0, 1):
        raise MalformedStoreError(f"{p.name} has an invalid is_trained byte ({trained}).", code="bad-faiss-header")
    if metric not in _METRICS:
        raise MalformedStoreError(f"{p.name} has unknown metric_type {metric}.", code="bad-faiss-header")
    offset = _BASE.size
    metric_arg: float | None = None
    if metric > 1:
        if len(head) < offset + 4:
            raise MalformedStoreError(f"{p.name} is truncated inside the header.", code="bad-faiss-header")
        (metric_arg,) = struct.unpack_from("<f", head, offset)
        offset += 4
    if fourcc in _FLAT:
        if len(head) < offset + 8:
            raise MalformedStoreError(f"{p.name} is truncated inside the header.", code="bad-faiss-header")
        (count,) = struct.unpack_from("<Q", head, offset)
        if count != ntotal * d:
            raise MalformedStoreError(
                f"{p.name}: vector count {count} does not equal ntotal*d = {ntotal * d}.", code="bad-faiss-header"
            )
        expected = offset + 8 + count * 4
        if file_size != expected:
            raise MalformedStoreError(
                f"{p.name} is {file_size} bytes but a flat index with ntotal={ntotal}, d={d} must be {expected}.",
                code="bad-faiss-header",
            )
    elif file_size < offset + 8:
        raise MalformedStoreError(f"{p.name} is truncated ({file_size} bytes).", code="bad-faiss-header")
    return FaissHeader(
        fourcc=fourcc,
        index_type=FOURCC_NAMES[fourcc],
        d=d,
        ntotal=ntotal,
        is_trained=bool(trained),
        metric_type=metric,
        metric_arg=metric_arg,
        file_size=file_size,
    )
