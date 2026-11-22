"""faiss-defuse: inspect, convert and load LangChain FAISS stores without executing pickle.

The core (scan, header reading, convert, inspect) uses only the standard library.
``load_safe``, ``save_safe`` and the top-k part of ``verify`` need the ``[langchain]`` extra.
"""

from __future__ import annotations

from ._faiss_header import FaissHeader, read_faiss_header
from ._loader import load_safe, save_safe
from ._pickle_scan import Finding, ScanReport, scan, scan_bytes
from ._store import Manifest, StoreInfo, convert, inspect
from ._verify import VerifyReport, verify
from .errors import (
    DefuseError,
    MalformedStoreError,
    ManifestMismatchError,
    MissingExtraError,
    OptionConflictError,
    OutputExistsError,
    UnsafePickleError,
    UnsupportedContentError,
)
from .limits import Limits

__version__ = "0.1.1"

__all__ = [
    "DefuseError",
    "FaissHeader",
    "Finding",
    "Limits",
    "MalformedStoreError",
    "Manifest",
    "ManifestMismatchError",
    "MissingExtraError",
    "OptionConflictError",
    "OutputExistsError",
    "ScanReport",
    "StoreInfo",
    "UnsafePickleError",
    "UnsupportedContentError",
    "VerifyReport",
    "__version__",
    "convert",
    "inspect",
    "load_safe",
    "read_faiss_header",
    "save_safe",
    "scan",
    "scan_bytes",
    "verify",
]
