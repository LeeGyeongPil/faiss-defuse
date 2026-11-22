"""Inert decoder for LangChain FAISS ``index.pkl``.

A restricted ``find_class`` that returns real classes can still be abused when an
allowed class exposes dangerous attributes.  This decoder never imports LangChain
at all: every allowed ``(module, name)`` pair is mapped to an inert stub class or
to a small function in this file that only builds plain values.  The static scan
(`_pickle_scan`) always runs first; this is the second gate.
"""

from __future__ import annotations

import datetime as _dt
import io
import pickle
import struct
from dataclasses import dataclass, replace
from typing import Any

from ._allowlist import ALLOWED_GLOBALS, DOCSTORE_NAMES, DOCUMENT_NAMES, LEGACY_MODULE_PREFIXES
from ._docstate import DecodedDoc, normalize_doc, normalize_docstore
from ._pickle_scan import Finding, ScanReport, scan_bytes
from .errors import MalformedStoreError, UnsafePickleError, UnsupportedContentError
from .limits import Limits


class _UnsupportedValue(Exception):
    """Raised by a value shim for a shape it cannot represent."""


class Stub:
    """Inert stand-in for an allowed LangChain class.  It only remembers its state."""

    __slots__ = ("state", "kind")

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # constructor arguments are ignored
        self.state: Any = None
        self.kind = ""

    def __setstate__(self, state: Any) -> None:
        self.state = state


def _make_stub(kind: str) -> type:
    cls = type("Stub_" + kind, (Stub,), {"__slots__": ()})
    return cls


_STUB_DOCSTORE = _make_stub("docstore")
_STUB_DOCUMENT = _make_stub("document")


class _DType:
    """Minimal numpy dtype stand-in: remembers the type string and byte order."""

    __slots__ = ("name", "order")

    def __init__(self, name: Any = "", *_: Any) -> None:
        self.name = name if isinstance(name, str) else ""
        self.order = "|"

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, tuple) and len(state) > 1 and state[1] in ("<", ">", "|", "="):
            self.order = state[1]


_NUMPY_FORMATS = {
    "b1": "?", "i1": "b", "i2": "h", "i4": "i", "i8": "q",
    "u1": "B", "u2": "H", "u4": "I", "u8": "Q",
    "f2": "e", "f4": "f", "f8": "d",
}


def _np_scalar(dtype: Any, raw: Any) -> Any:
    if not isinstance(dtype, _DType) or not isinstance(raw, (bytes, bytearray)):
        raise _UnsupportedValue("numpy scalar with unexpected arguments")
    fmt = _NUMPY_FORMATS.get(dtype.name)
    if fmt is None:
        raise _UnsupportedValue(f"numpy scalar dtype {dtype.name!r}")
    order = ">" if dtype.order == ">" else "<"
    if len(raw) != struct.calcsize(fmt):
        raise _UnsupportedValue("numpy scalar with wrong byte length")
    return struct.unpack(order + fmt, bytes(raw))[0]


def _reconstructor(cls: Any, base: Any, state: Any) -> Any:
    if base is not object or not (isinstance(cls, type) and issubclass(cls, Stub)):
        raise _UnsupportedValue("copyreg._reconstructor for a non-stub class")
    return cls.__new__(cls)


def _codecs_encode(text: Any, encoding: Any = "utf-8", errors: Any = "strict") -> bytes:
    if encoding != "latin1" or not isinstance(text, str):
        raise _UnsupportedValue("_codecs.encode with an encoding other than latin1")
    return text.encode("latin1")


_RESOLVED: dict[tuple[str, str], Any] = {}
for _pair in DOCSTORE_NAMES:
    _RESOLVED[_pair] = _STUB_DOCSTORE
for _pair in DOCUMENT_NAMES:
    _RESOLVED[_pair] = _STUB_DOCUMENT
for _mod in ("builtins", "__builtin__"):
    _RESOLVED[(_mod, "set")] = set
    _RESOLVED[(_mod, "frozenset")] = frozenset
    _RESOLVED[(_mod, "object")] = object
_RESOLVED[("collections", "OrderedDict")] = __import__("collections").OrderedDict
_RESOLVED[("copyreg", "_reconstructor")] = _reconstructor
_RESOLVED[("_codecs", "encode")] = _codecs_encode
_RESOLVED[("datetime", "date")] = _dt.date
_RESOLVED[("datetime", "datetime")] = _dt.datetime
_RESOLVED[("datetime", "time")] = _dt.time
_RESOLVED[("datetime", "timedelta")] = _dt.timedelta
_RESOLVED[("datetime", "timezone")] = _dt.timezone
_RESOLVED[("numpy", "dtype")] = _DType
_RESOLVED[("numpy.core.multiarray", "scalar")] = _np_scalar
_RESOLVED[("numpy._core.multiarray", "scalar")] = _np_scalar
assert set(_RESOLVED) == set(ALLOWED_GLOBALS), "allow-list and decoder table must match"


class _DecodeBudget:
    """Bounds the work that copies data while ``index.pkl`` is decoded.

    The unpickler itself never copies: a memo reference (``BINGET``) only reuses an object.
    But some allowed constructors copy their argument - ``frozenset(L)``, ``set(L)``,
    ``OrderedDict(L)``, ``_codecs.encode(s)`` - and so does reading each document's state.
    A pickle can call ``frozenset(L)`` thousands of times on one memoised list ``L`` for
    a few bytes per call (a 100 kB file asked for 630 MB).

    ``copies``: a real pickler writes every argument it passes out in full, so everything
    such a constructor copies is already in the file once.  The total may therefore not exceed
    twice the file size (plus a little room).  This is not a tunable limit; going over it means
    the file was crafted.

    ``metadata``: documents may legitimately share one metadata dict (old LangChain versions
    did not copy it), and later steps (``load_safe``'s ``Document``, ``inspect``) touch every
    key of every document, so the summed key count is bounded by ``Limits.max_total_nodes``.
    """

    def __init__(self, pickle_size: int, limits: Limits) -> None:
        self.copy_cap = 2 * pickle_size + 65_536
        self.copied = 0
        self.max_keys = limits.max_total_nodes
        self.keys = 0

    def copy(self, n: int) -> None:
        self.copied += n
        if self.copied > self.copy_cap:
            raise MalformedStoreError(
                f"index.pkl reuses shared objects so often that decoding would copy more than {self.copy_cap} "
                "items (twice the file size). Real LangChain stores never do this; the file looks crafted "
                "to exhaust memory.",
                code="reference-amplification", params={"cap": str(self.copy_cap)},
            )

    def metadata(self, n: int) -> None:
        self.keys += n
        if self.keys > self.max_keys:
            raise MalformedStoreError(
                f"the documents' metadata holds more than {self.max_keys} keys in total (shared metadata "
                "counts once per document). Raise Limits.max_total_nodes if this is expected.",
                code="limit-exceeded", params={"limit": "max_total_nodes"},
            )


def _sized(arg: Any) -> int:
    if not hasattr(arg, "__len__"):
        raise _UnsupportedValue("constructor argument without a length")
    return len(arg)


def _counting(ctor: Any, budget: _DecodeBudget) -> Any:
    """Wrap a copying constructor so that it pays for its argument *before* copying it."""

    def build(*args: Any) -> Any:
        if args:
            budget.copy(_sized(args[0]))
        return ctor(*args)

    return build


def _plain_dict(*args: Any) -> dict[Any, Any]:
    # OrderedDict instances carry a __dict__, so a BUILD opcode could copy a shared state dict into
    # each of them.  A plain dict keeps the order, encodes the same and has no __dict__.
    return dict(*args)


_COPYING: dict[tuple[str, str], Any] = {
    ("builtins", "set"): set,
    ("__builtin__", "set"): set,
    ("builtins", "frozenset"): frozenset,
    ("__builtin__", "frozenset"): frozenset,
    ("collections", "OrderedDict"): _plain_dict,
    ("_codecs", "encode"): _codecs_encode,
}


class _InertUnpickler(pickle.Unpickler):
    """Unpickler whose ``find_class`` only knows the exact allow-listed pairs."""

    def __init__(self, file: Any, budget: _DecodeBudget) -> None:
        super().__init__(file)
        self.budget = budget

    def find_class(self, module: str, name: str) -> Any:  # noqa: D401
        pair = (module, name)
        if pair in _COPYING:  # constructors that copy their argument pay for it first
            return _counting(_COPYING[pair], self.budget)
        try:
            return _RESOLVED[pair]
        except KeyError:
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed") from None


@dataclass
class DecodedStore:
    """What ``index.pkl`` contained, as plain Python data."""

    docs: dict[str, DecodedDoc]  # docstore id -> document
    index_to_docstore_id: dict[int, str]
    imports: list[str]
    legacy: bool  # True if any import comes from the old ``langchain.*`` layout
    mixed: bool  # True if old and new layouts are mixed


def decode_checked(data: bytes, *, limits: Limits = Limits(), source: str = "<bytes>") -> tuple[ScanReport, DecodedStore]:
    """Scan, then decode.  Raises ``UnsafePickleError`` / ``UnsupportedContentError`` / ``MalformedStoreError``."""
    report = scan_bytes(data, limits=limits, source=source)
    if report.verdict == "unsafe":
        raise UnsafePickleError(report)
    if report.verdict == "unsupported":
        raise UnsupportedContentError(report)
    budget = _DecodeBudget(len(data), limits)
    try:
        top = _InertUnpickler(io.BytesIO(data), budget).load()
    except _UnsupportedValue as exc:
        raise UnsupportedContentError(_with_finding(report, str(exc)), str(exc)) from None
    except MalformedStoreError:  # the decode budget refused a copy
        raise
    except Exception as exc:  # damaged but statically harmless data
        raise MalformedStoreError(f"could not decode index.pkl: {type(exc).__name__}: {exc}", code="malformed-pickle") from None
    if not (isinstance(top, tuple) and len(top) == 2 and isinstance(top[0], _STUB_DOCSTORE) and isinstance(top[1], dict)):
        raise MalformedStoreError(
            "index.pkl is not a (InMemoryDocstore, index_to_docstore_id) pair. "
            "If it was written by serialize_to_bytes(), that format is not supported yet.",
            code="unexpected-structure",
        )
    docstore, idmap = top
    for key, value in idmap.items():
        if not (type(key) is int and isinstance(value, str)):
            raise MalformedStoreError("index_to_docstore_id must map int -> str.", code="unexpected-structure")
    raw_docs = normalize_docstore(docstore, _STUB_DOCUMENT, charge=budget.copy)
    docs: dict[str, DecodedDoc] = {}
    for doc_id, stub in raw_docs.items():
        doc = normalize_doc(doc_id, stub, _STUB_DOCUMENT, charge=budget.copy)
        budget.metadata(len(doc.metadata))
        docs[doc_id] = doc
    legacy_hits = [i for i in report.imports if i.startswith(LEGACY_MODULE_PREFIXES)]
    current_hits = [i for i in report.imports if i.startswith(("langchain_core.", "langchain_community."))]
    return report, DecodedStore(
        docs=docs,
        index_to_docstore_id=dict(idmap),
        imports=report.imports,
        legacy=bool(legacy_hits),
        mixed=bool(legacy_hits and current_hits),
    )


def _with_finding(report: ScanReport, detail: str) -> ScanReport:
    finding = Finding("unsupported-value-type", "unsupported", -1, "", detail=detail)
    return replace(report, verdict="unsupported", findings=[*report.findings, finding])
