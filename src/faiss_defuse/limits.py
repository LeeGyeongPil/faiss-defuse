"""Resource limits applied before any large read or allocation."""

from __future__ import annotations

from dataclasses import dataclass

_MiB = 1024 * 1024
_GiB = 1024 * _MiB


@dataclass(frozen=True)
class Limits:
    """Upper bounds that protect against oversized or hostile files.

    All values can be raised by the caller for genuinely huge stores.

    ``max_doc_nodes`` and ``max_total_nodes`` bound how many metadata values are written for one
    document and for the whole store: shared references in a pickle are expanded when converting,
    so a tiny pickle could otherwise produce an enormous docstore.  ``max_docstore_bytes`` bounds
    the JSON that is written as well as the file that is read.
    """

    max_pickle_bytes: int = 512 * _MiB
    max_opcodes: int = 50_000_000
    max_string_bytes: int = 64 * _MiB
    max_faiss_bytes: int = 16 * _GiB
    max_vectors: int = 200_000_000
    max_docstore_bytes: int = 2 * _GiB
    max_findings: int = 200
    max_depth: int = 100
    max_doc_nodes: int = 100_000
    max_total_nodes: int = 10_000_000
