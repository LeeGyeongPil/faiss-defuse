"""Shared references in a pickle must not blow up the converted docstore (resource bounds)."""

from __future__ import annotations

import struct
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import DeterministicFakeEmbedding

from faiss_defuse import Limits, convert, inspect, scan
from faiss_defuse._jsontags import Budget, encode
from faiss_defuse._pickle_scan import scan_bytes
from faiss_defuse._store import read_jsonl, write_jsonl
from faiss_defuse._stub_decode import decode_checked
from faiss_defuse.cli import main
from faiss_defuse.errors import MalformedStoreError, UnsupportedContentError


def _save(path: Path, emb: DeterministicFakeEmbedding, metadata: dict[str, Any]) -> Path:
    FAISS.from_texts(["a", "bb"], emb, metadatas=[metadata, {"m": 1}]).save_local(str(path))
    return path


def test_shared_list_dag_is_refused_quickly(tmp_path: Path, emb: DeterministicFakeEmbedding) -> None:
    node: list[Any] = ["x"]
    for _ in range(30):  # 2**30 expanded values from a pickle of a few hundred bytes
        node = [node, node]
    src = _save(tmp_path / "dag", emb, {"m": node})
    assert (src / "index.pkl").stat().st_size < 2000
    assert scan(src).safe  # the pickle itself is harmless; the amplification only appears when converting

    started = time.monotonic()
    with pytest.raises(MalformedStoreError) as info:
        convert(src, tmp_path / "out")
    assert info.value.code == "limit-exceeded"
    assert "Limits.max_doc_nodes" in info.value.detail
    assert time.monotonic() - started < 10
    assert not (tmp_path / "out" / "docstore.jsonl").exists()


def test_cli_reports_limit_with_exit_code_3(tmp_path: Path, emb: DeterministicFakeEmbedding,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    node: list[Any] = ["x"]
    for _ in range(30):
        node = [node, node]
    src = _save(tmp_path / "dag", emb, {"m": node})
    assert main(["--lang", "en", "convert", str(src), str(tmp_path / "out")]) == 3
    assert "max_doc_nodes" in capsys.readouterr().err


def test_repeated_large_bytes_value_hits_the_byte_budget(tmp_path: Path, emb: DeterministicFakeEmbedding) -> None:
    blob = b"\x00" * 200_000
    src = _save(tmp_path / "blob", emb, {"m": [blob] * 50})  # one object, fifty references
    assert scan(src).safe
    with pytest.raises(MalformedStoreError) as info:
        convert(src, tmp_path / "out", limits=Limits(max_docstore_bytes=1_000_000))
    assert info.value.code == "limit-exceeded"
    assert "max_docstore_bytes" in info.value.detail


def test_total_node_limit_counts_across_documents(tmp_path: Path, emb: DeterministicFakeEmbedding) -> None:
    src = _save(tmp_path / "many", emb, {"m": list(range(50))})
    with pytest.raises(MalformedStoreError) as info:
        convert(src, tmp_path / "out", limits=Limits(max_total_nodes=20))
    assert "max_total_nodes" in info.value.detail


def test_normal_store_converts_within_default_limits(tmp_path: Path, emb: DeterministicFakeEmbedding) -> None:
    src = _save(tmp_path / "ok", emb, {"m": list(range(500)), "t": ("a", "b")})
    assert convert(src, tmp_path / "out").doc_count == 2


def test_budget_unit() -> None:
    budget = Budget(max_doc_nodes=3)
    with pytest.raises(MalformedStoreError):
        encode([1, 2, 3, 4], budget=budget)
    budget = Budget(max_doc_nodes=3)
    assert encode([1, 2], budget=budget) == [1, 2]
    budget.start_document()  # the per-document count starts again
    assert encode([1, 2], budget=budget) == [1, 2]


# --------------------------------------------------------------------------- round 3: keys, ints, decode-time copies
def _measure(fn: Any) -> tuple[BaseException | None, float, int]:
    """Run ``fn``; return (exception, seconds, peak traced bytes)."""
    tracemalloc.start()
    started = time.monotonic()
    error: BaseException | None = None
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 - the caller checks the type
        error = exc
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return error, time.monotonic() - started, peak


def _nested_shared_keys() -> dict[str, Any]:
    level: dict[str, Any] = {"k" * 1000: 0}
    for _ in range(12):  # 4,096 copies of a 1,000-character key, but only ~8,000 nodes
        level = {"x": level, "y": level}
    return level


AMPLIFIED = {
    # one dict with a long key, referenced 40,000 times (the round-2 QA reproduction)
    "shared-key": {"m": [{"k" * 10_000: 1}] * 40_000},
    # one long string value referenced 40,000 times
    "shared-value": {"m": ["v" * 10_000] * 40_000},
    # a dict whose keys are not plain strings goes through the {"$map": ...} branch
    "shared-map-key": {"m": [{("k" * 10_000,): 1}] * 40_000},
    # shared dicts inside shared dicts
    "nested-shared-key": {"m": _nested_shared_keys()},
}


@pytest.mark.parametrize("shape", sorted(AMPLIFIED))
def test_amplified_metadata_is_refused_while_encoding(shape: str, tmp_path: Path, emb: DeterministicFakeEmbedding) -> None:
    src = _save(tmp_path / shape, emb, AMPLIFIED[shape])
    assert (src / "index.pkl").stat().st_size < 200_000  # a small file ...
    assert scan(src).safe  # ... that executes nothing
    dst = tmp_path / "out"
    error, seconds, peak = _measure(lambda: convert(src, dst, limits=Limits(max_docstore_bytes=1_000_000)))
    assert isinstance(error, MalformedStoreError) and error.code == "limit-exceeded", error
    assert error.params["limit"] == "max_docstore_bytes"
    assert seconds < 5 and peak < 50_000_000, (seconds, peak)  # refused before the copies exist
    assert not dst.exists()  # refused while encoding: nothing was written, not even the directory


def test_shared_megabyte_key_is_refused_under_default_limits(tmp_path: Path, emb: DeterministicFakeEmbedding) -> None:
    src = _save(tmp_path / "mb", emb, {"m": [{"k" * 1_000_000: 1}] * 3_000})  # ~3 GB of JSON from ~1 MB
    error, seconds, peak = _measure(lambda: convert(src, tmp_path / "out"))
    assert isinstance(error, MalformedStoreError) and error.params.get("limit") == "max_docstore_bytes", error
    assert seconds < 10 and peak < 200_000_000, (seconds, peak)


def test_shared_long_integer_is_charged() -> None:
    # The standard pickler never memoises ints, but a hand-made file can (BINPUT/BINGET work for any object).
    error, seconds, peak = _measure(lambda: encode([10**3999] * 40_000, budget=Budget(max_bytes=1_000_000)))
    assert isinstance(error, MalformedStoreError) and error.params == {"limit": "max_docstore_bytes"}, error
    assert seconds < 5 and peak < 50_000_000, (seconds, peak)


def test_key_bytes_are_charged_in_the_budget() -> None:
    budget = Budget(max_bytes=10_000)
    with pytest.raises(MalformedStoreError):
        encode({"k" * 20_000: 1}, budget=budget)  # the value is tiny; the key alone is over the budget
    budget = Budget(max_bytes=10_000)
    assert encode({"k" * 100: 1}, budget=budget) == {"k" * 100: 1}


def test_too_long_integer_is_unsupported_not_a_crash(tmp_path: Path, emb: DeterministicFakeEmbedding,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    src = _save(tmp_path / "int", emb, {"n": 10**5000})  # json.dumps cannot write 5,001 digits
    assert scan(src).safe
    with pytest.raises(UnsupportedContentError) as info:
        convert(src, tmp_path / "out")
    assert "builtins.int(digits>4300)" in str(info.value)
    assert info.value.report.size == (src / "index.pkl").stat().st_size  # the real scan summary is kept
    assert not (tmp_path / "out").exists()
    assert main(["--lang", "ko", "convert", str(src), str(tmp_path / "cli")]) == 1  # exit 1, not a usage error
    err = capsys.readouterr().err
    assert "지원하지 않는 값 타입" in err and "builtins.int(digits>4300)" in err
    assert not (tmp_path / "cli").exists()
    assert encode(10**4299) == 10**4299  # exactly at the limit is still fine


def _frozenset_calls(n_items: int, n_calls: int) -> bytes:
    """``frozenset(L)`` called ``n_calls`` times on one memoised list ``L`` (protocol 2)."""
    out = bytearray(b"\x80\x02cbuiltins\nfrozenset\nq\x00](")
    out += b"".join(b"J" + struct.pack("<i", i) for i in range(n_items))
    out += b"e\x85q\x01("  # APPENDS, TUPLE1, BINPUT 1, MARK
    out += b"h\x00h\x01R" * n_calls  # BINGET 0, BINGET 1, REDUCE
    return bytes(out + b"l.")


def _codecs_calls(n_chars: int, n_calls: int) -> bytes:
    """``_codecs.encode(s, 'latin1')`` called ``n_calls`` times on one memoised string."""
    raw = ("a" * n_chars).encode()
    out = bytearray(b"\x80\x02c_codecs\nencode\nq\x00")
    out += b"X" + struct.pack("<I", len(raw)) + raw + b"X\x06\x00\x00\x00latin1\x86q\x01("
    out += b"h\x00h\x01R" * n_calls
    return bytes(out + b"l.")


@pytest.mark.parametrize("blob", [_frozenset_calls(20_000, 300), _codecs_calls(1_000_000, 300)], ids=["frozenset", "codecs"])
def test_copying_constructors_cannot_amplify_while_decoding(blob: bytes) -> None:
    assert scan_bytes(blob).safe  # statically harmless: only allow-listed globals
    error, seconds, peak = _measure(lambda: decode_checked(blob))
    assert isinstance(error, MalformedStoreError) and error.code == "reference-amplification", error
    assert seconds < 5 and peak < 60_000_000, (seconds, peak)  # was 630 MB / 300 MB before the decode budget


def test_ordered_dict_cannot_take_a_shared_state(tmp_path: Path) -> None:
    """BUILD on an OrderedDict would copy a state dict into its __dict__; the decoder returns plain dicts."""
    blob = b"\x80\x02ccollections\nOrderedDict\nq\x00}q\x01(K\x01K\x01ub)Rh\x01b."  # OrderedDict() then BUILD
    with pytest.raises(MalformedStoreError) as info:
        decode_checked(blob)
    assert info.value.code == "malformed-pickle"


def test_shared_metadata_across_documents_is_bounded(tmp_path: Path, emb: DeterministicFakeEmbedding) -> None:
    shared = {f"key{i}": i for i in range(1_000)}
    vs = FAISS.from_texts([f"t{i}" for i in range(200)], emb)
    for doc in vs.docstore._dict.values():  # type: ignore[attr-defined]
        doc.__dict__["metadata"] = shared  # old LangChain versions shared the dict like this
    vs.save_local(str(tmp_path / "shared"))
    assert inspect(tmp_path / "shared").doc_count == 200  # 200,000 keys in total: fine by default
    with pytest.raises(MalformedStoreError) as info:
        inspect(tmp_path / "shared", limits=Limits(max_total_nodes=100_000))
    assert info.value.params == {"limit": "max_total_nodes"}


def test_cli_limit_option(tmp_path: Path, emb: DeterministicFakeEmbedding, capsys: pytest.CaptureFixture[str]) -> None:
    src = _save(tmp_path / "many", emb, {"m": list(range(50))})
    assert main(["--lang", "en", "convert", str(src), str(tmp_path / "a"), "--limit", "max_total_nodes=20"]) == 3
    err = capsys.readouterr().err
    assert "max_total_nodes" in err and "--limit max_total_nodes=<number>" in err  # tells the CLI user how
    assert main(["--lang", "ko", "convert", str(src), str(tmp_path / "b"), "--limit", "max_total_nodes=20"]) == 3
    assert "명령줄에서 허용하려면 --limit max_total_nodes=<숫자>" in capsys.readouterr().err
    assert main(["convert", str(src), str(tmp_path / "c"), "--limit", "max_total_nodes=1_000"]) == 0
    for bad in ("max_total_nodes", "nope=5", "max_total_nodes=0", "max_total_nodes=x"):
        with pytest.raises(SystemExit) as stop:
            main(["--lang", "ko", "scan", str(src), "--limit", bad])
        assert stop.value.code == 2
        assert "--limit 값" in capsys.readouterr().err


def test_writer_counts_utf8_bytes_and_streams(tmp_path: Path) -> None:
    record = {"pos": 0, "id": "a", "doc_id": None, "page_content": "가" * 1_000, "metadata": {}}
    path = tmp_path / "docstore.jsonl"
    with pytest.raises(MalformedStoreError):  # 1,000 characters but 3,000 bytes
        write_jsonl(path, [record], Limits(max_docstore_bytes=2_000))
    assert path.stat().st_size == 0  # stopped before the bytes reached the file
    write_jsonl(path, [record], Limits(max_docstore_bytes=4_000))
    assert read_jsonl(path, Limits(max_docstore_bytes=4_000))[0]["page_content"] == "가" * 1_000


def test_writer_keeps_lone_surrogates(tmp_path: Path) -> None:
    record = {"pos": 0, "id": "a", "doc_id": None, "page_content": "x\ud800y", "metadata": {"k\udfff": "\ud83d"}}
    path = tmp_path / "docstore.jsonl"
    write_jsonl(path, [record])
    assert read_jsonl(path, Limits()) == [record]


def test_failed_write_removes_the_directory_it_created(tmp_path: Path, emb: DeterministicFakeEmbedding) -> None:
    FAISS.from_texts(["가" * 1_000], emb).save_local(str(tmp_path / "src"))
    dst = tmp_path / "new" / "out"
    with pytest.raises(MalformedStoreError):  # the estimate (characters) passes, the written bytes do not
        convert(tmp_path / "src", dst, limits=Limits(max_docstore_bytes=2_000))
    assert not dst.exists()
    keep = tmp_path / "keep"
    keep.mkdir()
    (keep / "notes.txt").write_text("mine")
    with pytest.raises(MalformedStoreError):
        convert(tmp_path / "src", keep, limits=Limits(max_docstore_bytes=2_000))
    assert sorted(p.name for p in keep.iterdir()) == ["notes.txt"]  # a directory we did not create stays
