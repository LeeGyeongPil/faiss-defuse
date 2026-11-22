"""``load_safe`` and ``save_safe``: the flag-free replacements for ``load_local`` / ``save_local``."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable

from . import _adapter
from ._faiss_header import FaissHeader
from ._store import (
    Manifest,
    _norm_strategy,
    records_for,
    is_converted,
    load_original,
    read_converted,
    require_mapping,
    resolve_options,
    write_store,
)
from .errors import MalformedStoreError, OptionConflictError
from .limits import Limits


def _sanity_check_index(index: Any, header: FaissHeader) -> None:
    if int(index.ntotal) != header.ntotal or int(index.d) != header.d:
        raise MalformedStoreError(
            f"faiss reports d={index.d}, ntotal={index.ntotal} but the header said d={header.d}, ntotal={header.ntotal}.",
            code="bad-faiss-header",
        )


def load_safe(
    path: str | Path,
    embeddings: Any,
    *,
    index_name: str = "index",
    distance_strategy: Any = None,
    normalize_l2: bool | None = None,
    relevance_score_fn: Callable[[float], float] | None = None,
    allow_option_override: bool = False,
    limits: Limits = Limits(),
) -> Any:
    """Load a store as a LangChain ``FAISS`` object without ``allow_dangerous_deserialization``.

    ``path`` may be a pickle-free store (``manifest.json``: hashes are verified first) or an
    original ``save_local`` directory (statically scanned, decoded with inert stubs, and
    ``index.faiss`` header-checked before ``faiss.read_index``).  The pickle is never executed.

    Options recorded in a pickle-free store are applied.  Passing different values raises
    :class:`OptionConflictError` unless ``allow_option_override`` is true.
    """
    _adapter.require_langchain("load_safe")
    p = Path(path)
    if is_converted(p):
        conv = read_converted(p, limits=limits)
        options = dict(conv.manifest.faiss_options)
        if distance_strategy is not None:
            wanted = _norm_strategy(distance_strategy)
            if wanted != options["distance_strategy"]:
                _conflict("distance_strategy", options["distance_strategy"], wanted, allow_option_override)
                options["distance_strategy"] = wanted
        if normalize_l2 is not None and bool(normalize_l2) != options["normalize_L2"]:
            _conflict("normalize_l2", options["normalize_L2"], bool(normalize_l2), allow_option_override)
            options["normalize_L2"] = bool(normalize_l2)
        docs, idmap, header, faiss_path = conv.docs, conv.index_to_docstore_id, conv.header, conv.faiss_path
    else:
        original = load_original(p, index_name=index_name, limits=limits)
        require_mapping(original.header, original.store.index_to_docstore_id, set(original.store.docs))
        options, warns = resolve_options(original.header, distance_strategy, normalize_l2)
        for w in warns:
            if w["code"] != "distance-strategy-inferred":
                warnings.warn(f"faiss-defuse: {w['detail']}", UserWarning, stacklevel=2)
        docs, idmap, header, faiss_path = original.store.docs, original.store.index_to_docstore_id, original.header, original.faiss_path
    index = _adapter.read_index(faiss_path)
    _sanity_check_index(index, header)
    return _adapter.build_faiss(embeddings, index, docs, idmap, options, relevance_score_fn)


def _conflict(name: str, recorded: Any, wanted: Any, allow: bool) -> None:
    if not allow:
        raise OptionConflictError(
            f"{name}={wanted!r} conflicts with the value recorded in the store ({recorded!r}). "
            "Drop the argument to use the recorded value, or pass allow_option_override=True."
        )
    warnings.warn(f"faiss-defuse: overriding recorded {name} {recorded!r} with {wanted!r}", UserWarning, stacklevel=3)


def save_safe(vectorstore: Any, dst: str | Path, *, overwrite: bool = False, limits: Limits = Limits()) -> Manifest:
    """Save a live LangChain ``FAISS`` store without pickle (index.faiss + docstore.jsonl + manifest.json).

    ``distance_strategy`` and ``normalize_L2`` are read from the object and recorded.
    """
    _adapter.require_langchain("save_safe")
    index, docs, idmap, options = _adapter.live_parts(vectorstore)
    missing = [v for v in idmap.values() if v not in docs]
    if missing:
        raise MalformedStoreError(f"{len(missing)} mapped ids are missing from the docstore.", code="mapping-mismatch")
    records = records_for(((pos, idmap[pos], docs[idmap[pos]]) for pos in sorted(idmap)), limits)
    return write_store(
        Path(dst),
        write_index=lambda target: _adapter.write_index(index, target),
        records=records,
        options=options,
        warns=[],
        origin={"kind": "live", "legacy": False},
        overwrite=overwrite,
        check=lambda header: _check_live(header, idmap, set(docs)),
        limits=limits,
    )


def _check_live(header: FaissHeader, idmap: dict[int, str], ids: set[str]) -> None:
    require_mapping(header, idmap, ids)


__all__ = ["load_safe", "save_safe"]
