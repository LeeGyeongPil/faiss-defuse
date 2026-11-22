"""Type-tagged JSON encoding for document metadata.

Plain JSON loses tuples, sets, dates and non-string keys.  Values that JSON
cannot hold exactly are wrapped in a one-key object whose key starts with ``$``.
Dictionaries whose keys are not plain strings (or start with ``$``) are written
as ``{"$map": [[key, value], ...]}`` so that no real key can be mistaken for a tag.
See docs/format.md.
"""

from __future__ import annotations

import base64
import datetime as dt
import math
import sys
from typing import Any

from .errors import MalformedStoreError


class UnsupportedType(Exception):
    """A value of a type that has no tag.  ``type_name`` names it."""

    def __init__(self, type_name: str) -> None:
        self.type_name = type_name
        super().__init__(type_name)


def _numpy_scalar(value: Any) -> Any:
    """Convert a live numpy scalar (``save_safe`` path) into a Python number."""
    if type(value).__module__ == "numpy" and getattr(value, "shape", None) == () and hasattr(value, "item"):
        return value.item()
    return value


class Budget:
    """Running cost of everything written so far.

    A pickle may reference one object many times (memo references).  Encoding expands every
    reference, so a few hundred bytes can describe gigabytes of JSON.  The budget is charged for
    each value *and each dictionary key* as it is expanded - its node count and an estimate of its
    JSON size - and raises ``limit-exceeded`` before the memory is spent.
    Call :meth:`start_document` before each document to reset the per-document node count.
    """

    def __init__(self, *, max_doc_nodes: int = 100_000, max_total_nodes: int = 10_000_000,
                 max_bytes: int = 2 * 1024**3) -> None:
        self.max_doc_nodes = max_doc_nodes
        self.max_total_nodes = max_total_nodes
        self.max_bytes = max_bytes
        self.doc_nodes = 0
        self.total_nodes = 0
        self.total_bytes = 0

    @classmethod
    def from_limits(cls, limits: Any) -> "Budget":
        return cls(max_doc_nodes=limits.max_doc_nodes, max_total_nodes=limits.max_total_nodes,
                   max_bytes=limits.max_docstore_bytes)

    def start_document(self) -> None:
        self.doc_nodes = 0

    def node(self) -> None:
        self.doc_nodes += 1
        self.total_nodes += 1
        if self.doc_nodes > self.max_doc_nodes:
            raise MalformedStoreError(
                f"one document's metadata expands to more than {self.max_doc_nodes} values "
                "(shared references are expanded). Raise Limits.max_doc_nodes if this is expected.",
                code="limit-exceeded", params={"limit": "max_doc_nodes"},
            )
        if self.total_nodes > self.max_total_nodes:
            raise MalformedStoreError(
                f"metadata of the store expands to more than {self.max_total_nodes} values "
                "(shared references are expanded). Raise Limits.max_total_nodes if this is expected.",
                code="limit-exceeded", params={"limit": "max_total_nodes"},
            )

    def size(self, nbytes: int) -> None:
        self.total_bytes += nbytes
        if self.total_bytes > self.max_bytes:
            raise MalformedStoreError(
                f"the converted docstore would be larger than {self.max_bytes} bytes. "
                "Raise Limits.max_docstore_bytes if this is expected.",
                code="limit-exceeded", params={"limit": "max_docstore_bytes"},
            )


_LOG10_2 = 0.30103


def _int_too_long(value: int) -> bool:
    """True if ``str(value)`` would hit Python's integer-to-text digit limit (4300 digits by default).

    ``json.dumps`` would raise a bare ``ValueError`` for such a number, so it is refused up front.
    The bit length decides cheaply; only numbers right at the limit are compared exactly.
    """
    limit = sys.get_int_max_str_digits()
    if limit <= 0:  # the limit was switched off for this interpreter
        return False
    bits = value.bit_length()
    if bits * _LOG10_2 + 1 < limit:
        return False
    if bits * _LOG10_2 > limit + 1:
        return True
    return abs(value) >= 10**limit


def _json_size(value: Any) -> int:
    """Rough number of JSON characters ``value`` itself takes, without its children."""
    if isinstance(value, str):
        return len(value) + 2  # two quotes
    if type(value) in (bytes, bytearray):
        return len(value) * 4 // 3 + 16  # base64 + {"$bytes":...}
    if type(value) is int:
        return int(value.bit_length() * _LOG10_2) + 2  # digits + sign
    return 16  # null/true/false, a float, brackets or a {"$tag": ...} wrapper


def encode(value: Any, *, max_depth: int = 100, budget: Budget | None = None, _depth: int = 0) -> Any:
    """Return a JSON-ready structure for ``value`` (raises :class:`UnsupportedType`).

    With a ``budget`` the expanded size is bounded (see :class:`Budget`): every value and every
    key is charged *before* it is expanded, so a shared object that is referenced many times is
    refused before its copies exist.
    """
    if _depth > max_depth:
        raise MalformedStoreError(f"metadata nested deeper than {max_depth} levels.", code="limit-exceeded",
                                  params={"limit": "max_depth"})
    value = _numpy_scalar(value)
    nxt = _depth + 1
    if budget is not None:
        budget.node()
        budget.size(_json_size(value) + 1)  # + 1 for the separating comma

    def rec(item: Any) -> Any:
        return encode(item, max_depth=max_depth, budget=budget, _depth=nxt)

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and type(value) is int:
        if _int_too_long(value):  # json.dumps cannot write it; refuse it with a clear type name
            raise UnsupportedType(f"builtins.int(digits>{sys.get_int_max_str_digits()})")
        return value
    if isinstance(value, float) and type(value) is float:
        if math.isnan(value):
            return {"$float": "nan"}
        if math.isinf(value):
            return {"$float": "inf" if value > 0 else "-inf"}
        return value
    if type(value) is list:
        return [rec(v) for v in value]
    if type(value) is tuple:
        return {"$tuple": [rec(v) for v in value]}
    if type(value) in (set, frozenset):
        items = sorted((rec(v) for v in value), key=_sort_key)
        return {"$set" if type(value) is set else "$frozenset": items}
    if type(value) in (bytes, bytearray):
        return {"$bytes": base64.b64encode(bytes(value)).decode("ascii")}
    if type(value) is dt.datetime:
        return {"$datetime": value.isoformat()}
    if type(value) is dt.date:
        return {"$date": value.isoformat()}
    if type(value) is dt.time:
        return {"$time": value.isoformat()}
    if type(value) is dt.timedelta:
        return {"$timedelta": [value.days, value.seconds, value.microseconds]}
    if isinstance(value, dict):
        if all(type(k) is str and not k.startswith("$") for k in value):
            out: dict[str, Any] = {}
            for k, v in value.items():
                if budget is not None:
                    budget.size(len(k) + 3)  # the key is written for every reference: "key":
                out[k] = rec(v)
            return out
        return {"$map": [[rec(k), rec(v)] for k, v in value.items()]}  # tagged keys go through rec()
    t = type(value)
    raise UnsupportedType(f"{t.__module__}.{t.__qualname__}")


def _sort_key(encoded: Any) -> str:
    import json

    return json.dumps(encoded, sort_keys=True)


def decode(value: Any, *, max_depth: int = 100, _depth: int = 0) -> Any:
    """Inverse of :func:`encode`."""
    if _depth > max_depth:
        raise MalformedStoreError(f"metadata nested deeper than {max_depth} levels.", code="limit-exceeded",
                                  params={"limit": "max_depth"})
    nxt = _depth + 1

    def rec(item: Any) -> Any:
        return decode(item, max_depth=max_depth, _depth=nxt)

    if isinstance(value, list):
        return [rec(v) for v in value]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        (key, inner), = value.items()
        if key.startswith("$"):
            try:
                return _decode_tag(key, inner, rec)
            except MalformedStoreError:
                raise
            except Exception as exc:
                raise MalformedStoreError(f"invalid {key} value: {exc}", code="bad-tag") from None
    for key in value:
        if key.startswith("$"):
            raise MalformedStoreError(f"unexpected key {key!r} in an object with other keys.", code="bad-tag")
    return {k: rec(v) for k, v in value.items()}


def _decode_tag(key: str, inner: Any, rec: Any) -> Any:
    if key == "$float":
        return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[inner]
    if key == "$tuple":
        return tuple(rec(v) for v in inner)
    if key == "$set":
        return {_hashable(rec(v)) for v in inner}
    if key == "$frozenset":
        return frozenset(_hashable(rec(v)) for v in inner)
    if key == "$bytes":
        return base64.b64decode(inner.encode("ascii"), validate=True)
    if key == "$datetime":
        return dt.datetime.fromisoformat(inner)
    if key == "$date":
        return dt.date.fromisoformat(inner)
    if key == "$time":
        return dt.time.fromisoformat(inner)
    if key == "$timedelta":
        days, seconds, micro = inner
        return dt.timedelta(days=days, seconds=seconds, microseconds=micro)
    if key == "$map":
        return {_hashable(rec(k)): rec(v) for k, v in inner}
    raise MalformedStoreError(f"unknown tag {key!r}.", code="bad-tag")


def _hashable(value: Any) -> Any:
    hash(value)  # raises TypeError for lists/dicts
    return value
