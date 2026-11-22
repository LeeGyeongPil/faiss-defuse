from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import pytest
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from conftest import make_docs
from faiss_defuse import (
    ManifestMismatchError,
    MalformedStoreError,
    OptionConflictError,
    convert,
    inspect,
    load_safe,
    save_safe,
)
from helpers import flip_byte


def _top(vs: FAISS, query: str = "doc 3 about topic 0", k: int = 5) -> list[tuple[str, float]]:
    return [(d.id, round(float(s), 5)) for d, s in vs.similarity_search_with_score(query, k=k)]


def test_load_safe_converted_and_original(clean_store: Path, tmp_path: Path, emb) -> None:
    reference = FAISS.load_local(str(clean_store), emb, allow_dangerous_deserialization=True)
    convert(clean_store, tmp_path / "conv")
    for path in (tmp_path / "conv", clean_store):
        store = load_safe(path, emb)  # no allow_dangerous_deserialization anywhere
        assert isinstance(store, FAISS)
        assert _top(store) == _top(reference)
        assert len(store.docstore._dict) == 20
    # a custom index_name works for original stores
    named = tmp_path / "named"
    reference.save_local(str(named), index_name="vec")
    assert _top(load_safe(named, emb, index_name="vec")) == _top(reference)


def test_tamper_detected(clean_store: Path, tmp_path: Path, emb) -> None:
    convert(clean_store, tmp_path / "conv")
    for name in ("docstore.jsonl", "index.faiss"):
        copy = tmp_path / f"copy-{name}"
        shutil.copytree(tmp_path / "conv", copy)
        flip_byte(copy / name, 12)
        with pytest.raises(ManifestMismatchError) as exc:
            load_safe(copy, emb)
        assert exc.value.file == name
    # appended bytes change the size, which is reported too
    grown = tmp_path / "grown"
    shutil.copytree(tmp_path / "conv", grown)
    with (grown / "docstore.jsonl").open("ab") as fh:
        fh.write(b"\n")
    with pytest.raises(ManifestMismatchError):
        load_safe(grown, emb)


def test_manifest_cannot_name_other_files(clean_store: Path, tmp_path: Path, emb) -> None:
    import json

    convert(clean_store, tmp_path / "conv")
    m = json.loads((tmp_path / "conv" / "manifest.json").read_text())
    m["files"]["../../etc/passwd"] = {"sha256": "0" * 64, "size": 1}
    (tmp_path / "conv" / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(MalformedStoreError) as exc:
        load_safe(tmp_path / "conv", emb)
    assert exc.value.code == "bad-manifest"


def test_mapping_mismatch(work_store: Path, emb, tmp_path: Path) -> None:
    vs = FAISS.load_local(str(work_store), emb, allow_dangerous_deserialization=True)
    del vs.index_to_docstore_id[3]
    vs.save_local(str(work_store))
    with pytest.raises(MalformedStoreError) as exc:
        load_safe(work_store, emb)
    assert exc.value.code == "mapping-mismatch"
    with pytest.raises(MalformedStoreError) as exc2:
        convert(work_store, tmp_path / "out")
    assert exc2.value.code == "mapping-mismatch"
    assert not inspect(work_store).ok


def test_options_preserved_and_conflict(tmp_path: Path, emb) -> None:
    original = FAISS.from_documents(make_docs(), emb, normalize_L2=True, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    manifest = save_safe(original, tmp_path / "safe")
    assert manifest.faiss_options == {"distance_strategy": "MAX_INNER_PRODUCT", "normalize_L2": True, "source": "recorded"}
    assert manifest.origin["kind"] == "live"
    assert sorted(p.name for p in (tmp_path / "safe").iterdir()) == ["docstore.jsonl", "index.faiss", "manifest.json"]

    restored = load_safe(tmp_path / "safe", emb)
    assert restored._normalize_L2 is True
    assert restored.distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT
    assert _top(restored) == _top(original)

    with pytest.raises(OptionConflictError, match="normalize_l2"):
        load_safe(tmp_path / "safe", emb, normalize_l2=False)
    with pytest.raises(OptionConflictError, match="distance_strategy"):
        load_safe(tmp_path / "safe", emb, distance_strategy="EUCLIDEAN_DISTANCE")
    load_safe(tmp_path / "safe", emb, normalize_l2=True, distance_strategy="max_inner_product")  # same values: fine
    with pytest.warns(UserWarning, match="overriding"):
        overridden = load_safe(tmp_path / "safe", emb, normalize_l2=False, allow_option_override=True)
    assert overridden._normalize_L2 is False


def test_original_store_load_warns_about_unknown_normalize(clean_store: Path, emb) -> None:
    with pytest.warns(UserWarning, match="normalize_L2"):
        load_safe(clean_store, emb)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        load_safe(clean_store, emb, normalize_l2=False, distance_strategy="EUCLIDEAN_DISTANCE")


def test_save_safe_refuses_existing_and_foreign_docstore(tmp_path: Path, emb) -> None:
    from faiss_defuse import OutputExistsError

    vs = FAISS.from_documents(make_docs(3), emb)
    save_safe(vs, tmp_path / "s")
    with pytest.raises(OutputExistsError):
        save_safe(vs, tmp_path / "s")
    save_safe(vs, tmp_path / "s", overwrite=True)
    vs.docstore = object()  # type: ignore[assignment]
    with pytest.raises(TypeError, match="InMemoryDocstore"):
        save_safe(vs, tmp_path / "t")


def test_load_safe_without_extras(monkeypatch: pytest.MonkeyPatch, clean_store: Path, emb) -> None:
    from faiss_defuse import MissingExtraError, _adapter

    monkeypatch.setattr(_adapter, "extras_available", lambda: False)
    with pytest.raises(MissingExtraError, match=r"faiss-defuse\[langchain\]"):
        load_safe(clean_store, emb)
