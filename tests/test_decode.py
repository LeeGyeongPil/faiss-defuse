from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from faiss_defuse._demo import _legacy_modules
from faiss_defuse._stub_decode import decode_checked
from faiss_defuse.errors import MalformedStoreError, UnsafePickleError

from conftest import make_docs


def test_decode_v2_and_legacy(clean_store: Path) -> None:
    _, current = decode_checked((clean_store / "index.pkl").read_bytes())
    expected = make_docs()
    assert len(current.docs) == len(expected)
    by_id = {d.id: d for d in expected}
    for doc_id, decoded in current.docs.items():
        original = by_id[decoded.doc_id]
        assert decoded.page_content == original.page_content
        assert decoded.metadata == original.metadata
    assert sorted(current.index_to_docstore_id) == list(range(20))
    assert current.legacy is False

    # Pydantic v1 style state under the pre-split module paths, with an extra legacy field.
    with _legacy_modules() as (Docstore, Document):
        legacy_blob = pickle.dumps((Docstore({"u1": Document("hello", {"source": "a.pdf"})}), {0: "u1"}), protocol=4)
    report, legacy = decode_checked(legacy_blob)
    assert report.imports == ["langchain.docstore.in_memory.InMemoryDocstore", "langchain.schema.document.Document"]
    doc = legacy.docs["u1"]
    assert (doc.page_content, doc.metadata, doc.doc_id) == ("hello", {"source": "a.pdf"}, None)
    assert doc.extra == {"lookup_str": "", "lookup_index": 0}
    assert legacy.legacy is True and legacy.mixed is False


def test_decode_refuses_unsafe_before_unpickling(sentinel: Path) -> None:
    from helpers import variant_reduce_os_system

    data, _ = variant_reduce_os_system(sentinel)
    with pytest.raises(UnsafePickleError) as exc:
        decode_checked(data)
    assert exc.value.report.findings[0].code == "disallowed-global"
    assert not sentinel.exists()


def test_wrong_top_level_structure_is_reported() -> None:
    with pytest.raises(MalformedStoreError) as exc:
        decode_checked(pickle.dumps(({"a": 1}, {}), protocol=4))
    assert exc.value.code == "unexpected-structure"


def test_numpy_scalars_decode_without_numpy_import(tmp_path: Path) -> None:
    """numpy scalars in metadata are read by struct, so the decoder works on stores from numpy-heavy loaders."""
    import numpy as np
    from langchain_core.documents import Document
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_community.vectorstores import FAISS

    doc = Document(page_content="x", metadata={"f": np.float32(1.5), "i": np.int64(7), "b": np.bool_(True)}, id="a")
    FAISS.from_documents([doc], DeterministicFakeEmbedding(size=8)).save_local(str(tmp_path))
    report, store = decode_checked((tmp_path / "index.pkl").read_bytes())
    meta = next(iter(store.docs.values())).metadata
    assert meta == {"f": 1.5, "i": 7, "b": True}
    assert all(type(v) in (float, int, bool) for v in meta.values())
    assert any("multiarray.scalar" in i for i in report.imports)
