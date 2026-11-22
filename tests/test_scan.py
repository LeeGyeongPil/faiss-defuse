from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from helpers import (
    opcode_positions,
    variant_dotted_name,
    variant_extension_code,
    variant_memo_stack_global,
    variant_persistent_id,
    variant_reduce_os_system,
    variant_trailing_data,
    variant_unresolved_global,
)

from faiss_defuse import Limits, scan, scan_bytes
from faiss_defuse._demo import _Payload, _legacy_modules
from faiss_defuse.errors import MalformedStoreError

import pickle


def test_clean_store_is_safe(clean_store: Path) -> None:
    report = scan(clean_store)
    assert report.verdict == "safe"
    assert report.findings == []
    assert report.imports == [
        "langchain_community.docstore.in_memory.InMemoryDocstore",
        "langchain_core.documents.base.Document",
    ]
    assert report.protocol == 4
    assert report.opcode_counts["STACK_GLOBAL"] >= 2


def test_scan_accepts_pickle_file_path(clean_store: Path) -> None:
    assert scan(clean_store / "index.pkl").verdict == "safe"


def _check(data: bytes, expected: dict) -> None:
    report = scan_bytes(data)
    assert report.verdict == "unsafe"
    hits = [f for f in report.findings if f.code == expected["code"]]
    assert hits, f"no {expected['code']} in {[f.code for f in report.findings]}"
    finding = hits[0]
    assert finding.offset == expected["offset"]
    assert finding.opcode == expected["opcode"]
    positions = opcode_positions(data)
    if expected["code"] == "trailing-data":  # the offset is the first byte after STOP
        assert positions[finding.offset - 1] == "STOP"
    else:
        assert positions[finding.offset] == expected["opcode"]
    if expected["target"]:
        assert (finding.module, finding.name) == expected["target"]


def _build(name: str, sentinel: Path, clean_pickle: bytes) -> tuple[bytes, dict]:
    return {
        "reduce": lambda: variant_reduce_os_system(sentinel),
        "memo": lambda: variant_memo_stack_global(sentinel),
        "dotted": variant_dotted_name,
        "ext": variant_extension_code,
        "persid": variant_persistent_id,
        "trailing": lambda: variant_trailing_data(clean_pickle, sentinel),
        "unresolved": variant_unresolved_global,
    }[name]()


@pytest.mark.parametrize("name", ["reduce", "memo", "dotted", "ext", "persid", "trailing", "unresolved"])
def test_malicious_variants_rejected(name: str, clean_store: Path, sentinel: Path) -> None:
    data, expected = _build(name, sentinel, (clean_store / "index.pkl").read_bytes())
    _check(data, expected)
    assert not sentinel.exists()


def test_document_swap_is_rejected(sentinel: Path) -> None:
    """A structurally normal store in which one document carries a payload in its metadata."""
    with _legacy_modules() as (Docstore, Document):
        store = Docstore({"a": Document("fine", {}), "b": Document("evil", {"k": _Payload("echo x > " + str(sentinel))})})
        data = pickle.dumps((store, {0: "a", 1: "b"}), protocol=4)
    report = scan_bytes(data)
    assert report.verdict == "unsafe"
    assert [f.code for f in report.findings] == ["disallowed-global"]
    assert (report.findings[0].module, report.findings[0].name) == ("posix", "system")
    assert not sentinel.exists()


def test_control_unpickling_really_executes(tmp_path: Path, sentinel: Path) -> None:
    """Control: the same bytes DO create the sentinel when someone unpickles them (in a child process)."""
    data, _ = variant_reduce_os_system(sentinel)
    blob = tmp_path / "evil.pkl"
    blob.write_bytes(data)
    subprocess.run([sys.executable, "-I", "-c", "import pickle,sys; pickle.loads(open(sys.argv[1],'rb').read())", str(blob)],
                   check=False, capture_output=True)
    assert sentinel.exists(), "the control payload did not fire; the sentinel check would prove nothing"
    sentinel.unlink()
    scan_bytes(data)
    assert not sentinel.exists()


def test_known_unsupported_type_is_not_unsafe() -> None:
    from helpers import Built, s

    b = Built()
    b.add(s("pandas._libs.tslibs.timestamps")).add(s("_unpickle_timestamp")).add(b"\x93", "g").add(b"." )
    report = scan_bytes(b.bytes())
    assert report.verdict == "unsupported"
    assert report.findings[0].code == "unsupported-value-type"


def test_limits_are_enforced(clean_store: Path) -> None:
    data = (clean_store / "index.pkl").read_bytes()
    assert scan_bytes(data, limits=Limits(max_pickle_bytes=10)).findings[0].code == "limit-exceeded"
    assert scan_bytes(data, limits=Limits(max_opcodes=5)).findings[0].code == "limit-exceeded"
    assert scan_bytes(data, limits=Limits(max_string_bytes=3)).findings[0].code == "limit-exceeded"


def test_huge_memo_index_is_refused() -> None:
    # LONG_BINPUT with index 4,000,000,000 would make the C unpickler allocate gigabytes.
    data = b"\x80\x04N" + b"r" + (4_000_000_000).to_bytes(4, "little") + b"."
    report = scan_bytes(data)
    assert report.verdict == "unsafe" and report.findings[0].code == "limit-exceeded"


@pytest.mark.parametrize("data", [b"", b"\x80\x04", b"not a pickle", b"\xff\xff"])
def test_damaged_input_is_malformed_not_a_crash(data: bytes) -> None:
    report = scan_bytes(data)
    assert report.verdict == "unsafe"
    assert report.findings[0].code == "malformed-pickle"


def test_plain_data_pickles_are_safe() -> None:
    assert scan_bytes(pickle.dumps({"a": [1, 2.5, "x", None, (1, 2)]}, protocol=0)).verdict == "safe"
    assert scan_bytes(pickle.dumps({"a": {1, 2}}, protocol=2)).imports == ["__builtin__.set"]


def test_missing_and_converted_paths(tmp_path: Path) -> None:
    with pytest.raises(MalformedStoreError) as exc:
        scan(tmp_path / "nope")
    assert exc.value.code == "missing-file"
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(MalformedStoreError) as exc2:
        scan(tmp_path)
    assert exc2.value.code == "no-pickle"
