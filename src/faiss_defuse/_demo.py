"""Offline practice material: builds sample stores and walks through scan -> convert -> verify -> load_safe.

Everything here is real behaviour on real files; only the embeddings are the
fake, deterministic kind (``DeterministicFakeEmbedding``), so search quality is
meaningless.  The "malicious" sample runs a harmless ``echo`` that would create a
sentinel file if - and only if - someone unpickled it.
"""

from __future__ import annotations

import os
import pickle
import shlex
import shutil
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from . import _adapter
from ._pickle_scan import ScanReport, scan
from ._store import convert, load_original
from ._verify import VerifyReport, verify

SENTINEL_NAME = "PWNED"
_SENTENCES = [
    "FAISS stores vectors in a binary file.",
    "LangChain keeps document text in a pickle next to it.",
    "A pickle is a list of instructions, not plain data.",
    "Unpickling can import modules and call functions.",
    "A scan can read the instructions without running them.",
    "Hashes in a manifest catch changes after the build.",
    "Metadata can hold dates, tuples and sets.",
    "Cosine and L2 distance rank results differently.",
    "Chunking decides what a retriever can find.",
    "Embeddings turn text into points in space.",
    "Deterministic fake embeddings make demos repeatable.",
    "Every store should be verified before it ships.",
]


def make_clean_store(directory: Path, *, count: int = 12) -> None:
    """Write a real ``FAISS.save_local`` store (fake embeddings) into ``directory``."""
    _adapter.require_langchain("demo")
    import datetime as dt

    from langchain_core.documents import Document
    from langchain_core.embeddings import DeterministicFakeEmbedding

    FAISS, _, _, _ = _adapter._langchain()
    docs = [
        Document(
            page_content=_SENTENCES[i % len(_SENTENCES)] + f" (#{i})",
            metadata={"source": f"note-{i}.md", "page": i, "date": dt.date(2025, 1, 1 + i % 28), "tags": ("demo", f"t{i % 3}")},
            id=f"doc-{i}",
        )
        for i in range(count)
    ]
    store = FAISS.from_documents(docs, DeterministicFakeEmbedding(size=32))
    directory.mkdir(parents=True, exist_ok=True)
    store.save_local(str(directory))


@contextmanager
def _legacy_modules() -> Iterator[tuple[type, type]]:
    """Temporarily register fake pre-split LangChain modules so a legacy-shaped pickle can be written."""
    names = ["langchain", "langchain.docstore", "langchain.docstore.in_memory", "langchain.schema", "langchain.schema.document"]
    saved = {n: sys.modules.get(n) for n in names}
    for n in names:
        sys.modules[n] = types.ModuleType(n)

    class InMemoryDocstore:
        def __init__(self, data: dict[str, Any]) -> None:
            self._dict = data

    class Document:
        def __init__(self, content: str, metadata: dict[str, Any]) -> None:
            self.__dict__.update(page_content=content, metadata=metadata, lookup_str="", lookup_index=0, type="Document")

        def __getstate__(self) -> dict[str, Any]:
            return {"__dict__": dict(self.__dict__), "__fields_set__": {"page_content", "metadata"}}

    InMemoryDocstore.__qualname__ = "InMemoryDocstore"
    Document.__qualname__ = "Document"
    InMemoryDocstore.__module__ = "langchain.docstore.in_memory"
    Document.__module__ = "langchain.schema.document"
    sys.modules["langchain.docstore.in_memory"].InMemoryDocstore = InMemoryDocstore  # type: ignore[attr-defined]
    sys.modules["langchain.schema.document"].Document = Document  # type: ignore[attr-defined]
    try:
        yield InMemoryDocstore, Document
    finally:
        for n, mod in saved.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod


def make_legacy_store(directory: Path, clean: Path) -> None:
    """Re-write ``clean`` in the old ``langchain.*`` module layout (Pydantic v1 style state)."""
    original = load_original(clean)
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original.faiss_path, directory / "index.faiss")
    with _legacy_modules() as (Docstore, Document):
        docstore = Docstore({k: Document(d.page_content, d.metadata) for k, d in original.store.docs.items()})
        (directory / "index.pkl").write_bytes(pickle.dumps((docstore, original.store.index_to_docstore_id), protocol=4))


class _Payload:
    """Pickles to ``os.system(<echo ...>)`` - harmless, but leaves a file behind if executed."""

    def __init__(self, command: str) -> None:
        self.command = command

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return (os.system, (self.command,))


def make_malicious_store(directory: Path, clean: Path, sentinel: Path) -> None:
    """A store whose ``index.pkl`` would write ``sentinel`` if it were ever unpickled."""
    original = load_original(clean)
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original.faiss_path, directory / "index.faiss")
    command = f"echo faiss-defuse-demo-payload > {shlex.quote(str(sentinel))}"
    (directory / "index.pkl").write_bytes(pickle.dumps((_Payload(command), original.store.index_to_docstore_id), protocol=4))


@dataclass
class DemoResult:
    out: Path
    clean: ScanReport
    malicious: ScanReport
    legacy: ScanReport
    verify: VerifyReport
    sentinel: Path

    @property
    def sentinel_exists(self) -> bool:
        return self.sentinel.exists()


def run_demo(out: Path) -> DemoResult:
    """Build the samples under ``out`` and run the whole flow.  Never unpickles anything."""
    out.mkdir(parents=True, exist_ok=True)
    sentinel = out / SENTINEL_NAME
    sentinel.unlink(missing_ok=True)
    clean, legacy, malicious, converted = out / "clean", out / "legacy", out / "malicious", out / "converted"
    make_clean_store(clean)
    make_legacy_store(legacy, clean)
    make_malicious_store(malicious, clean, sentinel)
    reports = {"clean": scan(clean), "malicious": scan(malicious), "legacy": scan(legacy)}
    convert(clean, converted, overwrite=True)
    return DemoResult(out, reports["clean"], reports["malicious"], reports["legacy"], verify(converted, original=clean), sentinel)
