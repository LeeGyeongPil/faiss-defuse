from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest
from helpers import variant_reduce_os_system

from faiss_defuse import _adapter
from faiss_defuse.cli import main


def test_scan_clean_exit0(clean_store: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--lang", "en", "scan", str(clean_store)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("SAFE  index.pkl  imports: InMemoryDocstore, Document")
    assert main(["scan", str(clean_store), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"] == "safe" and data["findings"] == []
    assert data["imports"] == [
        "langchain_community.docstore.in_memory.InMemoryDocstore",
        "langchain_core.documents.base.Document",
    ]


def test_scan_unsafe_exit1(clean_store: Path, tmp_path: Path, sentinel: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad"
    shutil.copytree(clean_store, bad)
    data, expected = variant_reduce_os_system(sentinel)
    (bad / "index.pkl").write_bytes(data)
    assert main(["--lang", "en", "scan", str(bad)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("UNSAFE")
    assert f"offset 0x{expected['offset']:x} STACK_GLOBAL os.system (disallowed-global)" in out
    assert not sentinel.exists()


def test_inspect_convert_verify_flow(clean_store: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--lang", "en", "inspect", str(clean_store)]) == 0
    out = capsys.readouterr().out
    assert "IndexFlatL2 (IxF2)  d=32  ntotal=20  metric=L2" in out
    assert "documents     20" in out and "mapping       OK" in out
    assert "src 20" in out and "page 20" in out

    dst = tmp_path / "safe"
    assert main(["--lang", "en", "convert", str(clean_store), str(dst), "--distance-strategy", "euclidean_distance"]) == 0
    out = capsys.readouterr().out
    assert "Converted 20 documents" in out and "normalize-l2-unknown" in out
    assert main(["convert", str(clean_store), str(dst)]) == 2  # already there
    assert "destination not empty" in capsys.readouterr().err
    assert main(["convert", str(clean_store), str(dst), "--overwrite", "--normalize-l2", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["faiss_options"]["normalize_L2"] is True
    assert main(["--lang", "en", "inspect", str(dst)]) == 0
    assert "pickle-free store" in capsys.readouterr().out


def test_exit_codes_for_damaged_and_missing(tmp_path: Path, clean_store: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["scan", str(tmp_path / "missing")]) == 3
    assert capsys.readouterr().err.startswith("error")
    broken = tmp_path / "broken"
    shutil.copytree(clean_store, broken)
    (broken / "index.faiss").write_bytes(b"short")
    assert main(["inspect", str(broken)]) == 1  # the pickle is fine, the index header is not
    assert main(["convert", str(broken), str(tmp_path / "x")]) == 3
    assert main(["scan", str(tmp_path / "x")]) == 3
    with pytest.raises(SystemExit) as exc:
        main(["scan"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc2:
        main(["scan", "x", "--lang", "fr"])
    assert exc2.value.code == 2


def test_missing_extras_exit4(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(_adapter, "extras_available", lambda: False)
    assert main(["--lang", "en", "demo", "--out", str(tmp_path / "d")]) == 4
    assert 'pip install "faiss-defuse[langchain]"' in capsys.readouterr().err


def test_demo_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_dir = tmp_path / "demo"
    started = time.monotonic()
    code = main(["--lang", "en", "demo", "--out", str(out_dir)])
    elapsed = time.monotonic() - started
    out = capsys.readouterr().out
    assert code == 0, out
    assert elapsed < 60
    lines = out.splitlines()
    for expected in ("clean: SAFE", "malicious: UNSAFE", "legacy: SAFE", "converted: VERIFIED"):
        assert expected in lines
    assert "Practice mode" in out and "fake" in out
    assert not (out_dir / "PWNED").exists()
    assert "sentinel file: not created" in out
    assert sorted(p.name for p in out_dir.iterdir()) == ["clean", "converted", "legacy", "malicious"]
    # the practice stores stay usable afterwards
    assert main(["scan", str(out_dir / "malicious")]) == 1
    assert main(["scan", str(out_dir / "legacy")]) == 0
    assert not (out_dir / "PWNED").exists()
    # and a second run over the same directory works
    assert main(["--lang", "ko", "demo", "--out", str(out_dir)]) == 0
