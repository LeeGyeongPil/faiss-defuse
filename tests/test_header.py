from __future__ import annotations

import shutil
import struct
from pathlib import Path

import faiss
import numpy as np
import pytest

from faiss_defuse import _adapter, load_safe, read_faiss_header
from faiss_defuse.errors import MalformedStoreError

D = 16


def _indexes() -> dict[str, tuple[str, faiss.Index]]:
    x = np.random.default_rng(0).random((300, D), dtype="float32")
    cases = {
        "IxF2": faiss.IndexFlatL2(D),
        "IxFI": faiss.IndexFlatIP(D),
        "IxFl": faiss.IndexFlat(D, faiss.METRIC_L1),
        "IHNf": faiss.IndexHNSWFlat(D, 8),
        "IwFl": faiss.index_factory(D, "IVF4,Flat"),
        "IxMp": faiss.IndexIDMap(faiss.IndexFlatL2(D)),
        "IxPq": faiss.index_factory(D, "PQ4x4"),
    }
    out = {}
    for fourcc, idx in cases.items():
        if not idx.is_trained:
            idx.train(x)
        if fourcc == "IxMp":
            idx.add_with_ids(x, np.arange(300, dtype="int64") + 1000)
        else:
            idx.add(x)
        out[fourcc] = (fourcc, idx)
    return out


@pytest.mark.parametrize("fourcc", ["IxF2", "IxFI", "IxFl", "IHNf", "IwFl", "IxMp", "IxPq"])
def test_header_matches_faiss(fourcc: str, tmp_path: Path) -> None:
    _, idx = _indexes()[fourcc]
    path = tmp_path / "x.faiss"
    faiss.write_index(idx, str(path))
    header = read_faiss_header(path)
    assert header.fourcc == fourcc
    assert (header.d, header.ntotal, header.metric_type) == (idx.d, idx.ntotal, idx.metric_type)
    assert header.is_trained is True
    assert header.file_size == path.stat().st_size
    assert header.id_mapped == (fourcc == "IxMp")


def _tamper(src: Path, dst: Path, how: str) -> None:
    raw = bytearray(src.read_bytes())
    if how == "fourcc":
        raw[:4] = b"ZZZZ"
    elif how == "d0":
        raw[4:8] = struct.pack("<i", 0)
    elif how == "ntotal":
        raw[8:16] = struct.pack("<q", struct.unpack("<q", raw[8:16])[0] + 1)
    elif how == "ntotal_huge":
        raw[8:16] = struct.pack("<q", 2**62)
    elif how == "truncated":
        raw = raw[:20]
    elif how == "trimmed":
        raw = raw[:-4]
    dst.write_bytes(bytes(raw))


@pytest.mark.parametrize("how", ["fourcc", "d0", "ntotal", "ntotal_huge", "truncated", "trimmed"])
def test_tampered_header_rejected_before_read_index(how: str, work_store: Path, emb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(_adapter, "read_index", lambda p: calls.append(str(p)))
    _tamper(work_store / "index.faiss", work_store / "index.faiss", how)
    with pytest.raises(MalformedStoreError) as exc:
        load_safe(work_store, emb)
    assert exc.value.code in {"unsupported-index-type", "bad-faiss-header"}
    if how == "fourcc":
        assert exc.value.code == "unsupported-index-type"
    assert calls == [], "faiss.read_index must not be reached for a tampered header"


def test_missing_and_oversized_files(tmp_path: Path) -> None:
    from faiss_defuse import Limits

    with pytest.raises(MalformedStoreError) as exc:
        read_faiss_header(tmp_path / "nope.faiss")
    assert exc.value.code == "missing-file"
    idx = faiss.IndexFlatL2(D)
    idx.add(np.zeros((3, D), dtype="float32"))
    faiss.write_index(idx, str(tmp_path / "a.faiss"))
    with pytest.raises(MalformedStoreError) as exc2:
        read_faiss_header(tmp_path / "a.faiss", limits=Limits(max_faiss_bytes=10))
    assert exc2.value.code == "limit-exceeded"
    with pytest.raises(MalformedStoreError):
        read_faiss_header(tmp_path / "a.faiss", limits=Limits(max_vectors=2))
    shutil.rmtree(tmp_path, ignore_errors=True)
