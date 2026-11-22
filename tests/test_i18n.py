from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from faiss_defuse.cli import main
from faiss_defuse.i18n import messages, resolve_lang

ENV_NAMES = ("FAISS_DEFUSE_LANG", "LC_ALL", "LC_MESSAGES", "LANG")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_lang_selection_and_keys(clean_store: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # --- precedence -----------------------------------------------------------------
    assert resolve_lang(None, {}) == "en"
    assert resolve_lang(None, {"LANG": "ko_KR.UTF-8"}) == "ko"
    assert resolve_lang(None, {"LANG": "en_US.UTF-8", "LC_ALL": "ko_KR.UTF-8"}) == "ko"
    assert resolve_lang(None, {"LC_ALL": "C", "LANG": "ko_KR.UTF-8"}) == "en"  # first non-empty POSIX variable wins
    assert resolve_lang(None, {"FAISS_DEFUSE_LANG": "ko", "LANG": "en_US.UTF-8"}) == "ko"
    assert resolve_lang(None, {"FAISS_DEFUSE_LANG": "en", "LANG": "ko_KR.UTF-8"}) == "en"
    assert resolve_lang("en", {"FAISS_DEFUSE_LANG": "ko", "LANG": "ko_KR.UTF-8"}) == "en"
    assert resolve_lang("ko", {"FAISS_DEFUSE_LANG": "en"}) == "ko"
    assert resolve_lang(None, {"FAISS_DEFUSE_LANG": "fr", "LANG": "ko_KR.UTF-8"}) == "ko"  # unknown value is ignored

    # --- the two message tables have identical keys and placeholders ---------------------
    en, ko = messages("en"), messages("ko")
    assert set(en) == set(ko)
    placeholder = re.compile(r"\{(\w+)\}")
    for key in en:
        assert set(placeholder.findall(en[key])) == set(placeholder.findall(ko[key])), key
    assert any(ord(ch) > 0x3000 for ch in ko["help.description"])  # really Korean

    # --- the screen really changes --------------------------------------------------
    assert main(["--lang", "en", "scan", str(clean_store)]) == 0
    english = capsys.readouterr().out
    assert main(["--lang", "ko", "scan", str(clean_store)]) == 0
    korean = capsys.readouterr().out
    assert "protocol 4" in english and "Nothing was executed." in english
    assert "프로토콜 4" in korean and "아무것도 실행되지 않았습니다." in korean
    assert english.split()[0] == korean.split()[0] == "SAFE"  # verdict tokens stay machine-friendly

    monkeypatch.setenv("LANG", "ko_KR.UTF-8")
    main(["scan", str(clean_store)])
    assert "프로토콜" in capsys.readouterr().out
    monkeypatch.setenv("FAISS_DEFUSE_LANG", "en")
    main(["scan", str(clean_store)])
    assert "protocol 4" in capsys.readouterr().out
    main(["scan", str(clean_store), "--lang", "ko"])  # --lang also works after the sub-command
    assert "프로토콜" in capsys.readouterr().out

    # --- help text is localised -------------------------------------------------------
    for lang, snippet in (("en", "Statically scan"), ("ko", "정적으로 스캔")):
        with pytest.raises(SystemExit) as exit_info:
            main(["--lang", lang, "scan", "--help"])
        assert exit_info.value.code == 0
        assert snippet in capsys.readouterr().out

    # --- --json keys do not depend on the language ------------------------------------------
    def keys(value: object) -> object:
        if isinstance(value, dict):
            return {k: keys(v) for k, v in value.items()}
        if isinstance(value, list):
            return [keys(v) for v in value]
        return None

    main(["--lang", "en", "scan", str(clean_store), "--json"])
    en_json = json.loads(capsys.readouterr().out)
    main(["--lang", "ko", "scan", str(clean_store), "--json"])
    ko_json = json.loads(capsys.readouterr().out)
    assert keys(en_json) == keys(ko_json)
    assert en_json == ko_json  # a clean store has no findings, so nothing is localised


def test_finding_message_is_localised_but_code_is_not(tmp_path: Path, sentinel: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from helpers import variant_reduce_os_system

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "index.pkl").write_bytes(variant_reduce_os_system(sentinel)[0])
    main(["--lang", "en", "scan", str(bad), "--json"])
    en = json.loads(capsys.readouterr().out)["findings"][0]
    main(["--lang", "ko", "scan", str(bad), "--json"])
    ko = json.loads(capsys.readouterr().out)["findings"][0]
    assert en["code"] == ko["code"] == "disallowed-global"
    assert en["message"] != ko["message"]
    assert {k for k in en} == {k for k in ko}


def _english_sentences(text: str) -> list[str]:
    """Lines (minus identifiers in backticks, flags and paths) that still read like English prose."""
    found = []
    for line in text.splitlines():
        words = re.findall(r"(?<![\w/.\-`\[])[A-Za-z]{4,}(?![\w/.\-`(\]])", re.sub(r"`[^`]*`|\S*[/\\.=_]\S*", " ", line))
        words = [w for w in words if w not in {"SAFE", "UNSAFE", "VERIFIED", "MISMATCH", "UNSUPPORTED", "top", "pip", "install"}]
        if len(words) >= 3:
            found.append(line)
    return found


def test_korean_output_has_no_english_messages(clean_store: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # convert warnings (normalize_L2 unknown + strategy inferred)
    assert main(["--lang", "ko", "convert", str(clean_store), str(tmp_path / "c")]) == 0
    out = capsys.readouterr().out
    assert "--normalize-l2" in out and "경고 [normalize-l2-unknown]" in out
    assert not _english_sentences(out), out

    # verify without the optional extras path: top-k skipped reason must be Korean and appear once
    assert main(["--lang", "ko", "verify", str(tmp_path / "c"), "--against", str(clean_store)]) in (0, 1)
    out = capsys.readouterr().out
    assert not _english_sentences(out), out

    # errors: missing file, no pickle, exists, limit
    assert main(["--lang", "ko", "scan", str(tmp_path / "missing")]) == 3
    err = capsys.readouterr().err
    assert "오류" in err and "missing" in err and not _english_sentences(err), err
    assert main(["--lang", "ko", "scan", str(tmp_path / "c")]) == 3
    err = capsys.readouterr().err
    assert not _english_sentences(err), err
    assert main(["--lang", "ko", "convert", str(clean_store), str(tmp_path / "c")]) == 2
    err = capsys.readouterr().err
    assert "--overwrite" in err and not _english_sentences(err), err
    assert main(["--lang", "ko", "verify", str(tmp_path / "c"), "--trust-original"]) == 2
    assert not _english_sentences(capsys.readouterr().err)


def test_english_verify_reason_is_not_repeated(clean_store: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from faiss_defuse import _adapter

    assert main(["--lang", "en", "convert", str(clean_store), str(tmp_path / "c")]) == 0
    capsys.readouterr()
    main(["--lang", "en", "verify", str(tmp_path / "c"), "--against", str(clean_store)])
    out = capsys.readouterr().out
    assert "top-k check skipped: top-k" not in out
    assert out.count("faiss/numpy") <= 1 or _adapter.faiss_available()


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_verify_without_extras_prints_reason_once(clean_store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str], lang: str) -> None:
    from faiss_defuse import _adapter

    monkeypatch.setattr(_adapter, "faiss_available", lambda: False)
    assert main(["--lang", lang, "convert", str(clean_store), str(tmp_path / "c")]) == 0
    capsys.readouterr()
    assert main(["--lang", lang, "verify", str(tmp_path / "c"), "--against", str(clean_store)]) in (0, 1)
    out = capsys.readouterr().out
    assert out.count("faiss/numpy") == 1, out
    assert "top-k check skipped" not in out
    if lang == "ko":
        assert not _english_sentences(out), out
