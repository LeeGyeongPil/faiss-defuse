from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import pytest
from helpers import fake_modules
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from conftest import make_docs
from faiss_defuse import UnsupportedContentError, convert, inspect, scan
from faiss_defuse._demo import _legacy_modules
from faiss_defuse.cli import main
from faiss_defuse.errors import OutputExistsError, UnsafePickleError


def test_manifest_content(clean_store: Path, tmp_path: Path) -> None:
    manifest = convert(clean_store, tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["docstore.jsonl", "index.faiss", "manifest.json"]
    raw = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert raw["format"] == "faiss-defuse/1"
    assert raw["doc_count"] == 20 == manifest.doc_count
    assert raw["index"]["fourcc"] == "IxF2" and raw["index"]["d"] == 32 and raw["index"]["ntotal"] == 20
    assert set(raw["files"]) == {"index.faiss", "docstore.jsonl"}
    assert all(len(v["sha256"]) == 64 for v in raw["files"].values())
    assert (tmp_path / "out" / "index.faiss").read_bytes() == (clean_store / "index.faiss").read_bytes()
    first = json.loads((tmp_path / "out" / "docstore.jsonl").read_text().splitlines()[0])
    assert first["pos"] == 0 and first["page_content"] == "doc 0 about topic 0" and first["doc_id"] == "id-0"
    assert first["metadata"]["tags"] == ["a", "b"]


def test_convert_refuses_to_overwrite_without_flag(clean_store: Path, tmp_path: Path) -> None:
    convert(clean_store, tmp_path / "out")
    with pytest.raises(OutputExistsError):
        convert(clean_store, tmp_path / "out")
    convert(clean_store, tmp_path / "out", overwrite=True)
    with pytest.raises(OutputExistsError):
        convert(clean_store, clean_store, overwrite=True)


def test_convert_refuses_unsafe_store(clean_store: Path, tmp_path: Path, sentinel: Path) -> None:
    from helpers import variant_reduce_os_system

    bad = tmp_path / "bad"
    shutil.copytree(clean_store, bad)
    (bad / "index.pkl").write_bytes(variant_reduce_os_system(sentinel)[0])
    with pytest.raises(UnsafePickleError):
        convert(bad, tmp_path / "out")
    assert not (tmp_path / "out").exists() and not sentinel.exists()


def _timestamp_blob() -> bytes:
    def _unpickle_timestamp(*args: object) -> object:  # never called: the scan stops first
        raise AssertionError

    _unpickle_timestamp.__module__ = "pandas._libs.tslibs.timestamps"
    _unpickle_timestamp.__qualname__ = "_unpickle_timestamp"

    class Timestamp:
        def __reduce__(self):  # noqa: ANN204
            return (_unpickle_timestamp, (1, None))

    with fake_modules(**{"pandas": {}, "pandas._libs": {}, "pandas._libs.tslibs": {},
                         "pandas._libs.tslibs.timestamps": {"_unpickle_timestamp": _unpickle_timestamp}}):
        with _legacy_modules() as (Docstore, Doc):
            blob = pickle.dumps((Docstore({"a": Doc("x", {"ts": Timestamp()})}), {0: "a"}), protocol=4)
    return blob


def test_unsupported_value_type(clean_store: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = tmp_path / "ts"
    store.mkdir()
    shutil.copyfile(clean_store / "index.faiss", store / "index.faiss")
    (store / "index.pkl").write_bytes(_timestamp_blob())
    report = scan(store)
    assert report.verdict == "unsupported"
    assert report.findings[0].code == "unsupported-value-type"
    assert report.findings[0].name == "_unpickle_timestamp"
    with pytest.raises(UnsupportedContentError) as exc:
        convert(store, tmp_path / "out")
    assert "_unpickle_timestamp" in str(exc.value) and "not a safety verdict" in str(exc.value)
    assert main(["--lang", "en", "convert", str(store), str(tmp_path / "out2")]) == 1
    assert "UNSUPPORTED" in capsys.readouterr().out
    assert main(["--lang", "en", "scan", str(store)]) == 1


def test_unsupported_nested_type_is_found_at_encode_time(tmp_path: Path, emb) -> None:
    """A type that is statically harmless but has no tag (a numpy dtype we cannot read) is reported, not guessed."""
    import numpy as np

    FAISS.from_documents([Document(page_content="x", metadata={"arr": np.array([1, 2])}, id="a")], emb).save_local(str(tmp_path / "s"))
    with pytest.raises(UnsupportedContentError):
        convert(tmp_path / "s", tmp_path / "o")


def test_inferred_options_warn(tmp_path: Path, emb) -> None:
    vs = FAISS.from_documents(make_docs(), emb, normalize_L2=True, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    vs.save_local(str(tmp_path / "src"))
    manifest = convert(tmp_path / "src", tmp_path / "dst")
    assert manifest.faiss_options == {"distance_strategy": "MAX_INNER_PRODUCT", "normalize_L2": False, "source": "inferred"}
    assert "normalize-l2-unknown" in {w["code"] for w in manifest.warnings}
    explicit = convert(tmp_path / "src", tmp_path / "dst2", distance_strategy="max_inner_product", normalize_l2=True)
    assert explicit.faiss_options == {"distance_strategy": "MAX_INNER_PRODUCT", "normalize_L2": True, "source": "user"}
    assert not any(w["code"] == "normalize-l2-unknown" for w in explicit.warnings)
    with pytest.raises(ValueError, match="distance_strategy"):
        convert(tmp_path / "src", tmp_path / "dst3", distance_strategy="nope")


def test_legacy_store_converts_and_keeps_extra_fields(clean_store: Path, tmp_path: Path) -> None:
    from faiss_defuse._demo import make_legacy_store

    legacy = tmp_path / "legacy"
    make_legacy_store(legacy, clean_store)
    info = inspect(legacy)
    assert info.module_generation == "legacy" and info.ok
    manifest = convert(legacy, tmp_path / "out")
    assert manifest.origin["legacy"] is True
    rec = json.loads((tmp_path / "out" / "docstore.jsonl").read_text().splitlines()[0])
    assert rec["extra"] == {"lookup_str": "", "lookup_index": 0}


def test_inspect_reports_mapping_problem(work_store: Path, emb, tmp_path: Path) -> None:
    vs = FAISS.load_local(str(work_store), emb, allow_dangerous_deserialization=True)
    del vs.index_to_docstore_id[3]
    vs.save_local(str(work_store))
    info = inspect(work_store)
    assert not info.ok
    assert any("19 entries" in p for p in info.mapping_problems)
    assert any("positions have no document id" in p for p in info.mapping_problems)
