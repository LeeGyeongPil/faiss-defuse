"""Static pickle scanner.

Reads the opcode stream with :func:`pickletools.genops` and never calls the
unpickler, so nothing in the file is executed.  A small symbolic stack follows
strings through ``MEMOIZE`` / ``BINGET`` so that ``STACK_GLOBAL`` arguments can
be resolved even when they were memoised earlier in the stream.
"""

from __future__ import annotations

import pickletools
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._allowlist import ALLOWED_GLOBALS, KNOWN_UNSUPPORTED
from .errors import MalformedStoreError
from .limits import Limits

_OTHER = object()  # a stack value we do not track
_MARK = pickletools.markobject
_SLICE = pickletools.stackslice

_STRING_OPS = {
    "SHORT_BINUNICODE",
    "BINUNICODE",
    "BINUNICODE8",
    "UNICODE",
    "STRING",
    "BINSTRING",
    "SHORT_BINSTRING",
}
_PUT_OPS = {"PUT", "BINPUT", "LONG_BINPUT"}
_GET_OPS = {"GET", "BINGET", "LONG_BINGET"}
_LENGTH_OPS = {"BINUNICODE8", "BINBYTES8", "BYTEARRAY8", "LONG4", "BINUNICODE", "BINBYTES"}
_POSITION_RE = re.compile(r"position (\d+)")


@dataclass(frozen=True)
class Finding:
    """One piece of evidence.  ``offset`` is the byte offset of the opcode (-1 if unknown)."""

    code: str
    severity: str  # "unsafe" | "unsupported"
    offset: int
    opcode: str
    module: str = ""
    name: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "offset": self.offset,
            "opcode": self.opcode,
            "module": self.module,
            "name": self.name,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ScanReport:
    """Result of a static scan."""

    verdict: str  # "safe" | "unsafe" | "unsupported"
    findings: list[Finding] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)  # "module.name", sorted, unique
    opcode_counts: dict[str, int] = field(default_factory=dict)
    protocol: int = 0
    size: int = 0
    source: str = ""

    @property
    def safe(self) -> bool:
        return self.verdict == "safe"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "source": self.source,
            "protocol": self.protocol,
            "size": self.size,
            "imports": list(self.imports),
            "opcode_counts": dict(self.opcode_counts),
            "findings": [f.to_dict() for f in self.findings],
        }


class _Stop(Exception):
    """Internal: stop scanning (limit reached or structure broken)."""


def _split_global(arg: Any) -> tuple[str, str]:
    text = str(arg)
    module, _, name = text.partition(" ")
    return module, name


def scan_bytes(data: bytes, *, limits: Limits = Limits(), source: str = "<bytes>") -> ScanReport:
    """Scan pickle ``data`` without executing it."""
    findings: list[Finding] = []
    imports: set[str] = set()
    counts: Counter[str] = Counter()
    protocol = 0
    size = len(data)

    def add(code: str, severity: str, offset: int, opcode: str, module: str = "", name: str = "", detail: str = "") -> None:
        findings.append(Finding(code, severity, offset, opcode, module, name, detail))
        if len(findings) >= limits.max_findings:
            raise _Stop

    def check_global(module: str, name: str, offset: int, opcode: str) -> None:
        imports.add(f"{module}.{name}")
        if "." in name:
            add("dotted-name", "unsafe", offset, opcode, module, name,
                "attribute path in a global name (protocol 4 getattr walk)")
        elif (module, name) in ALLOWED_GLOBALS:
            return
        elif (module, name) in KNOWN_UNSUPPORTED:
            add("unsupported-value-type", "unsupported", offset, opcode, module, name,
                "known harmless type that this version cannot convert")
        else:
            add("disallowed-global", "unsafe", offset, opcode, module, name,
                "import outside the allow-list")

    if size > limits.max_pickle_bytes:
        findings.append(Finding("limit-exceeded", "unsafe", 0, "", detail=f"pickle is {size} bytes, limit {limits.max_pickle_bytes}"))
        return _report(findings, imports, counts, protocol, size, source)

    stack: list[Any] = []
    markstack: list[int] = []
    memo: dict[int, Any] = {}
    stop_pos = -1
    last_pos = 0
    n_ops = 0

    try:
        for op, arg, pos in pickletools.genops(data):
            last_pos = pos
            name = op.name
            counts[name] += 1
            n_ops += 1
            if n_ops > limits.max_opcodes:
                add("limit-exceeded", "unsafe", pos, name, detail=f"more than {limits.max_opcodes} opcodes")
                raise _Stop
            if name in _LENGTH_OPS or isinstance(arg, (str, bytes, bytearray)):
                length = (arg.bit_length() + 7) // 8 if isinstance(arg, int) else len(arg)
                if length > limits.max_string_bytes:
                    add("limit-exceeded", "unsafe", pos, name, detail=f"value of {length} bytes, limit {limits.max_string_bytes}")
                    raise _Stop

            if name in _PUT_OPS or name in _GET_OPS:
                # The C unpickler grows its memo array to the largest index it sees.
                if not 0 <= int(arg) <= size:
                    add("limit-exceeded", "unsafe", pos, name, detail=f"memo index {arg} is beyond the pickle size")
                    raise _Stop

            # ---- security checks -------------------------------------------------
            if name == "PROTO":
                protocol = int(arg)
            elif name in ("GLOBAL", "INST"):
                module, gname = _split_global(arg)
                check_global(module, gname, pos, name)
            elif name == "STACK_GLOBAL":
                if len(stack) >= 2 and isinstance(stack[-1], str) and isinstance(stack[-2], str):
                    check_global(stack[-2], stack[-1], pos, name)
                else:
                    add("unresolved-global", "unsafe", pos, name, detail="STACK_GLOBAL arguments are not plain strings")
            elif name in ("PERSID", "BINPERSID"):
                add("persistent-id", "unsafe", pos, name, detail="persistent ids call back into application code")
            elif name in ("EXT1", "EXT2", "EXT4"):
                add("extension-code", "unsafe", pos, name, detail="copyreg extension codes resolve to arbitrary globals")

            # ---- symbolic stack ---------------------------------------------------
            _step(stack, markstack, memo, op, arg)

            if name == "STOP":
                stop_pos = pos
                break
    except _Stop:
        return _report(findings, imports, counts, protocol, size, source)
    except Exception as exc:  # genops/stack errors on damaged input
        match = _POSITION_RE.search(str(exc))
        offset = int(match.group(1)) if match else last_pos
        findings.append(Finding("malformed-pickle", "unsafe", offset, "", detail=f"{type(exc).__name__}: {exc}"))
        return _report(findings, imports, counts, protocol, size, source)

    if stop_pos < 0:
        findings.append(Finding("malformed-pickle", "unsafe", last_pos, "", detail="no STOP opcode"))
    elif stop_pos + 1 < size:
        findings.append(
            Finding("trailing-data", "unsafe", stop_pos + 1, "STOP",
                    detail=f"{size - stop_pos - 1} bytes after STOP (possible polyglot)")
        )
    return _report(findings, imports, counts, protocol, size, source)


def _step(stack: list[Any], markstack: list[int], memo: dict[int, Any], op: Any, arg: Any) -> None:
    """Advance the symbolic stack by one opcode (mirrors ``pickletools.dis``)."""
    name = op.name
    if name == "MARK":
        markstack.append(len(stack))
        stack.append(_MARK)
        return
    if name == "POP":
        if stack and stack[-1] is _MARK:
            markstack.pop()
        if not stack:
            raise ValueError("stack underflow at POP")
        stack.pop()
        return
    if name == "POP_MARK":
        idx = markstack.pop()
        del stack[idx:]
        return
    if name == "DUP":
        stack.append(stack[-1])
        return
    if name in _PUT_OPS:
        memo[int(arg)] = stack[-1]
        return
    if name == "MEMOIZE":
        memo[len(memo)] = stack[-1]
        return
    if name in _GET_OPS:
        stack.append(memo[int(arg)])
        return
    if name in _STRING_OPS:
        stack.append(arg if isinstance(arg, str) else _OTHER)
        return
    if name == "STOP":
        return

    before = op.stack_before
    after = op.stack_after
    if _SLICE in before:
        idx = markstack.pop()
        if idx > len(stack):
            raise ValueError("mark beyond stack")
        del stack[idx:]
        need = len(before) - 2
    else:
        need = len(before)
    if len(stack) < need:
        raise ValueError(f"stack underflow at {name}")
    if need:
        del stack[len(stack) - need:]
    stack.extend(_OTHER for _ in after)


def _report(
    findings: list[Finding],
    imports: set[str],
    counts: Counter[str],
    protocol: int,
    size: int,
    source: str,
) -> ScanReport:
    if any(f.severity == "unsafe" for f in findings):
        verdict = "unsafe"
    elif findings:
        verdict = "unsupported"
    else:
        verdict = "safe"
    return ScanReport(
        verdict=verdict,
        findings=list(findings),
        imports=sorted(imports),
        opcode_counts=dict(counts),
        protocol=protocol,
        size=size,
        source=source,
    )


def resolve_pickle_path(path: str | Path, index_name: str = "index") -> Path:
    """Return the ``.pkl`` file for a store directory or an explicit pickle path."""
    p = Path(path)
    if p.is_dir():
        pkl = p / f"{index_name}.pkl"
        if not pkl.is_file():
            if (p / "manifest.json").is_file():
                raise MalformedStoreError(
                    f"{p} is already a pickle-free store (it has manifest.json); there is no pickle to scan. "
                    "Use `faiss-defuse inspect` or `verify` instead.",
                    code="no-pickle", params={"path": str(p)},
                )
            raise MalformedStoreError(
                f"{pkl} not found. Pass the directory written by FAISS.save_local() "
                "(use --index-name if you saved with a custom index_name).",
                code="missing-file", params={"path": str(pkl)},
            )
        return pkl
    if not p.is_file():
        raise MalformedStoreError(f"{p} does not exist.", code="missing-file", params={"path": str(p)})
    return p


def scan(path: str | Path, *, limits: Limits = Limits(), index_name: str = "index") -> ScanReport:
    """Scan a LangChain FAISS store directory or a single ``index.pkl``.  Never executes the pickle."""
    pkl = resolve_pickle_path(path, index_name)
    if pkl.stat().st_size > limits.max_pickle_bytes:
        return _report(
            [Finding("limit-exceeded", "unsafe", 0, "", detail=f"pickle is {pkl.stat().st_size} bytes, limit {limits.max_pickle_bytes}")],
            set(), Counter(), 0, pkl.stat().st_size, str(pkl),
        )
    return scan_bytes(pkl.read_bytes(), limits=limits, source=str(pkl))
