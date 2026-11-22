"""The only module that touches faiss, numpy and LangChain.

Everything LangChain-specific lives here so that a successor package for the
(archived) langchain-community FAISS wrapper needs a change in one place.  The
constructor call follows langchain-community 0.4.x.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Any, Callable

from ._docstate import DecodedDoc
from .errors import MissingExtraError


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def extras_available() -> bool:
    return all(_has(m) for m in ("faiss", "numpy", "langchain_core", "langchain_community"))


def require_langchain(what: str) -> None:
    if not extras_available():
        raise MissingExtraError(what)


def faiss_available() -> bool:
    return _has("faiss") and _has("numpy")


def _import_faiss() -> Any:
    import faiss  # noqa: PLC0415

    return faiss


def read_index(path: str | Path) -> Any:
    """``faiss.read_index`` - callers must have validated the file first."""
    return _import_faiss().read_index(str(path))


def write_index(index: Any, path: str | Path) -> None:
    _import_faiss().write_index(index, str(path))


def _langchain() -> tuple[Any, Any, Any, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # langchain-community announces its sunset on import
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.utils import DistanceStrategy
        from langchain_core.documents import Document
    return FAISS, InMemoryDocstore, DistanceStrategy, Document


def build_faiss(
    embeddings: Any,
    index: Any,
    docs: dict[str, DecodedDoc],
    idmap: dict[int, str],
    options: dict[str, Any],
    relevance_score_fn: Callable[[float], float] | None,
) -> Any:
    FAISS, InMemoryDocstore, DistanceStrategy, Document = _langchain()
    docstore = InMemoryDocstore(
        {key: Document(page_content=d.page_content, metadata=d.metadata, id=d.doc_id) for key, d in docs.items()}
    )
    kwargs: dict[str, Any] = {"normalize_L2": bool(options["normalize_L2"])}
    if options.get("distance_strategy"):
        kwargs["distance_strategy"] = DistanceStrategy[options["distance_strategy"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "Normalizing L2 is not applicable ..." is about the caller's choice
        return FAISS(embeddings, index, docstore, dict(idmap), relevance_score_fn=relevance_score_fn, **kwargs)


def live_parts(vectorstore: Any) -> tuple[Any, dict[str, DecodedDoc], dict[int, str], dict[str, Any]]:
    """Read index, documents, id map and options from a live LangChain ``FAISS`` object."""
    _, InMemoryDocstore, _, _ = _langchain()
    docstore = getattr(vectorstore, "docstore", None)
    if not isinstance(docstore, InMemoryDocstore):
        raise TypeError(
            "save_safe supports FAISS stores backed by InMemoryDocstore only "
            f"(got {type(docstore).__name__}). Use FAISS.from_documents()/from_texts()."
        )
    docs: dict[str, DecodedDoc] = {}
    for key, doc in docstore._dict.items():
        docs[key] = DecodedDoc(doc.page_content, dict(doc.metadata), getattr(doc, "id", None))
    strategy = getattr(vectorstore.distance_strategy, "name", str(vectorstore.distance_strategy))
    options = {"distance_strategy": strategy, "normalize_L2": bool(vectorstore._normalize_L2), "source": "recorded"}
    return vectorstore.index, docs, dict(vectorstore.index_to_docstore_id), options


def load_local_trusted(path: str | Path, dim: int, index_name: str) -> tuple[dict[str, DecodedDoc], dict[int, str]]:
    """Open the ORIGINAL pickle with LangChain's loader.  Only used by ``verify --trust-original``.

    This executes the pickle; the caller opts in explicitly because they built the store themselves.
    """
    FAISS, _, _, _ = _langchain()
    from langchain_core.embeddings import DeterministicFakeEmbedding

    vs = FAISS.load_local(
        str(path), DeterministicFakeEmbedding(size=dim), index_name=index_name, allow_dangerous_deserialization=True
    )
    docs = {k: DecodedDoc(v.page_content, dict(v.metadata), getattr(v, "id", None)) for k, v in vs.docstore._dict.items()}
    return docs, dict(vs.index_to_docstore_id)
