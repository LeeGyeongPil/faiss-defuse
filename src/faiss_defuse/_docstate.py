"""Normalise the state of stubbed LangChain objects (Pydantic v1, v2 or plain dict)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import MalformedStoreError

_KNOWN_KEYS = {"page_content", "metadata", "id", "type"}


@dataclass
class DecodedDoc:
    """One document as plain data."""

    page_content: str
    metadata: dict[Any, Any]
    doc_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _no_charge(n: int) -> None:
    return None


def _state_dict(stub: Any, what: str, charge: Callable[[int], None] = _no_charge) -> dict[Any, Any]:
    """Copy of the stub's attribute dict.  ``charge(n)`` is called before ``n`` entries are copied
    (one shared state dict may sit behind many documents; see ``_stub_decode._DecodeBudget``)."""
    state = getattr(stub, "state", None)
    # Pydantic v2 writes {"__dict__": ..., "__pydantic_fields_set__": ...}; v1 writes
    # {"__dict__": ..., "__fields_set__": ...}; very old objects were plain dicts.
    if isinstance(state, dict):
        if "__dict__" in state and isinstance(state["__dict__"], dict):
            charge(len(state["__dict__"]))
            inner = dict(state["__dict__"])
            extra = state.get("__pydantic_extra__")
            if isinstance(extra, dict):
                charge(len(extra))
                for key, value in extra.items():
                    inner.setdefault(key, value)
            return inner
        charge(len(state))
        return dict(state)
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], dict):
        charge(len(state[0]))
        return dict(state[0])
    raise MalformedStoreError(
        f"unexpected {what} state of type {type(state).__name__}.", code="unexpected-structure"
    )


def normalize_docstore(docstore: Any, document_cls: type, charge: Callable[[int], None] = _no_charge) -> dict[str, Any]:
    """Return ``{docstore_id: document_stub}`` from an ``InMemoryDocstore`` stub."""
    state = _state_dict(docstore, "docstore", charge)
    inner = state.get("_dict")
    if not isinstance(inner, dict):
        raise MalformedStoreError(
            f"docstore state has no '_dict' mapping (keys: {sorted(map(str, state))}).",
            code="unexpected-structure",
        )
    for key, value in inner.items():
        if not isinstance(key, str) or not isinstance(value, document_cls):
            raise MalformedStoreError(
                "docstore entries must be str -> Document.", code="unexpected-structure"
            )
    return inner


def normalize_doc(doc_id: str, stub: Any, document_cls: type, charge: Callable[[int], None] = _no_charge) -> DecodedDoc:
    state = _state_dict(stub, f"document {doc_id!r}", charge)
    content = state.get("page_content")
    if not isinstance(content, str):
        raise MalformedStoreError(
            f"document {doc_id!r} has no text page_content (keys: {sorted(map(str, state))}).",
            code="unexpected-structure",
        )
    metadata = state.get("metadata", {})
    if not isinstance(metadata, dict):
        raise MalformedStoreError(f"document {doc_id!r} metadata is not a mapping.", code="unexpected-structure")
    inner_id = state.get("id")
    if inner_id is not None and not isinstance(inner_id, str):
        raise MalformedStoreError(f"document {doc_id!r} has a non-string id.", code="unexpected-structure")
    extra = {str(k): v for k, v in state.items() if k not in _KNOWN_KEYS and not str(k).startswith("__")}
    return DecodedDoc(page_content=content, metadata=metadata, doc_id=inner_id, extra=extra)
