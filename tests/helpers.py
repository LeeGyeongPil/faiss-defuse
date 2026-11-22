"""Shared helpers: hand-assembled pickles, tamper utilities, sentinel payloads."""

from __future__ import annotations

import json
import pickletools
import shlex
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from faiss_defuse._store import DOCSTORE_FILE, FAISS_FILE, MANIFEST_FILE, sha256_file

PROTO4 = b"\x80\x04"
STOP = b"."


def s(text: str) -> bytes:
    """SHORT_BINUNICODE."""
    raw = text.encode()
    assert len(raw) < 256
    return b"\x8c" + bytes([len(raw)]) + raw


def payload_command(sentinel: Path) -> str:
    return f"echo faiss-defuse-test-payload > {shlex.quote(str(sentinel))}"


class Built:
    """A pickle under construction that remembers where interesting opcodes start."""

    def __init__(self) -> None:
        self.data = bytearray(PROTO4)
        self.marks: dict[str, int] = {}

    def add(self, chunk: bytes, mark: str | None = None) -> "Built":
        if mark:
            self.marks[mark] = len(self.data)
        self.data += chunk
        return self

    def bytes(self) -> bytes:
        return bytes(self.data)


def variant_reduce_os_system(sentinel: Path) -> tuple[bytes, dict[str, Any]]:
    b = Built()
    b.add(s("os")).add(s("system")).add(b"\x93", "g")
    b.add(s(payload_command(sentinel))).add(b"\x85R").add(b"}\x86" + STOP)
    return b.bytes(), {"code": "disallowed-global", "offset": b.marks["g"], "opcode": "STACK_GLOBAL", "target": ("os", "system")}


def variant_memo_stack_global(sentinel: Path) -> tuple[bytes, dict[str, Any]]:
    b = Built()
    b.add(s("builtins") + b"\x94")  # memo 0
    b.add(s("eval") + b"\x94")  # memo 1
    b.add(b"0" + b"0")  # POP, POP: the strings now only live in the memo
    b.add(b"h\x00h\x01").add(b"\x93", "g")  # BINGET 0, BINGET 1, STACK_GLOBAL
    b.add(s(f"__import__('os').system({payload_command(sentinel)!r})")).add(b"\x85R").add(b"}\x86" + STOP)
    return b.bytes(), {"code": "disallowed-global", "offset": b.marks["g"], "opcode": "STACK_GLOBAL", "target": ("builtins", "eval")}


def variant_dotted_name() -> tuple[bytes, dict[str, Any]]:
    b = Built()
    b.add(s("langchain_core.documents.base")).add(s("Document.__init__.__globals__")).add(b"\x93", "g")
    b.add(b"}\x86" + STOP)
    return b.bytes(), {"code": "dotted-name", "offset": b.marks["g"], "opcode": "STACK_GLOBAL",
                       "target": ("langchain_core.documents.base", "Document.__init__.__globals__")}


def variant_extension_code() -> tuple[bytes, dict[str, Any]]:
    b = Built()
    b.add(b"\x82\x01", "g").add(b"}\x86" + STOP)
    return b.bytes(), {"code": "extension-code", "offset": b.marks["g"], "opcode": "EXT1", "target": None}


def variant_persistent_id() -> tuple[bytes, dict[str, Any]]:
    b = Built()
    b.add(b"N").add(b"Q", "g").add(b"}\x86" + STOP)
    return b.bytes(), {"code": "persistent-id", "offset": b.marks["g"], "opcode": "BINPERSID", "target": None}


def variant_trailing_data(clean: bytes, sentinel: Path) -> tuple[bytes, dict[str, Any]]:
    tail = b"cos\nsystem\n(S" + repr(payload_command(sentinel)).encode() + b"\ntR."
    return clean + tail, {"code": "trailing-data", "offset": len(clean), "opcode": "STOP", "target": None}


def variant_unresolved_global() -> tuple[bytes, dict[str, Any]]:
    b = Built()
    b.add(b"NN").add(b"\x93", "g").add(b"}\x86" + STOP)
    return b.bytes(), {"code": "unresolved-global", "offset": b.marks["g"], "opcode": "STACK_GLOBAL", "target": None}


def opcode_positions(data: bytes) -> dict[int, str]:
    out: dict[int, str] = {}
    for op, _arg, pos in pickletools.genops(data):
        out[pos] = op.name
    return out


@contextmanager
def fake_modules(**modules: dict[str, Any]) -> Iterator[None]:
    """Register throw-away modules (name -> attribute dict) so classes in them can be pickled."""
    saved = {n: sys.modules.get(n) for n in modules}
    try:
        for name, attrs in modules.items():
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[name] = mod
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def flip_byte(path: Path, index: int = 10) -> None:
    raw = bytearray(path.read_bytes())
    raw[index] ^= 0x01
    path.write_bytes(bytes(raw))


def recompute_manifest(directory: Path) -> None:
    """Pretend to be an attacker who edits files and then fixes the hashes."""
    mpath = directory / MANIFEST_FILE
    manifest = json.loads(mpath.read_text())
    for name in (FAISS_FILE, DOCSTORE_FILE):
        manifest["files"][name] = {"sha256": sha256_file(directory / name), "size": (directory / name).stat().st_size}
    mpath.write_text(json.dumps(manifest))
