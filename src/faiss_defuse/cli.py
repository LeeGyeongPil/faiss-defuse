"""Command line interface: ``faiss-defuse scan|inspect|convert|verify|demo``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from . import __version__
from . import _store, _verify
from ._allowlist import STRUCT
from ._pickle_scan import ScanReport, scan
from .errors import (
    DefuseError,
    MalformedStoreError,
    ManifestMismatchError,
    MissingExtraError,
    OptionConflictError,
    OutputExistsError,
    UnsafePickleError,
    UnsupportedContentError,
)
from .i18n import messages, resolve_lang, t
from .limits import Limits

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_DAMAGED, EXIT_EXTRA = 0, 1, 2, 3, 4
_STRATEGIES = list(_store.VALID_STRATEGIES)


def _prescan_lang(argv: Sequence[str]) -> str | None:
    """Find ``--lang`` before argparse runs so the help text itself can be localised."""
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return None


_LIMIT_NAMES = [f.name for f in dataclasses.fields(Limits)]


def _limit_parser(lang: str) -> Callable[[str], tuple[str, int]]:
    """``--limit NAME=NUMBER`` -> ``(name, number)``; argparse turns the error into exit code 2."""

    def parse(text: str) -> tuple[str, int]:
        name, sep, value = text.partition("=")
        name = name.strip().replace("-", "_")
        try:
            number = int(value.strip())  # int() also accepts 1_000_000
        except ValueError:
            number = 0
        if not sep or name not in _LIMIT_NAMES or number < 1:
            raise argparse.ArgumentTypeError(t(lang, "error.limit_arg", value=text, names=", ".join(_LIMIT_NAMES)))
        return name, number

    return parse


def _limits(args: argparse.Namespace) -> Limits:
    """Default limits with the ``--limit`` overrides applied (a later value for the same name wins)."""
    return Limits(**dict(getattr(args, "limit", None) or []))


def build_parser(lang: str) -> argparse.ArgumentParser:
    def h(key: str) -> str:
        return t(lang, key)

    limits = argparse.ArgumentParser(add_help=False)  # shared by every command that reads a store
    limits.add_argument("--limit", action="append", metavar="NAME=NUMBER", type=_limit_parser(lang),
                        help=t(lang, "help.limit", names=", ".join(_LIMIT_NAMES)))

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lang", choices=["en", "ko"], default=argparse.SUPPRESS, help=h("help.lang"))
    parser = argparse.ArgumentParser(
        prog="faiss-defuse", description=h("help.description"), epilog=h("help.epilog"), parents=[common]
    )
    parser.add_argument("--version", action="version", version=f"faiss-defuse {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="{scan,inspect,convert,verify,demo}")

    def add(name: str, help_key: str, with_limits: bool = True) -> argparse.ArgumentParser:
        parents = [common, limits] if with_limits else [common]
        return sub.add_parser(name, help=h(help_key), description=h(help_key), parents=parents, epilog=h("help.epilog"))

    p = add("scan", "help.scan")
    p.add_argument("path", help=h("help.path"))
    p.add_argument("--json", action="store_true", help=h("help.json"))
    p.add_argument("--index-name", default="index", help=h("help.index_name"))

    p = add("inspect", "help.inspect")
    p.add_argument("path", help=h("help.path"))
    p.add_argument("--json", action="store_true", help=h("help.json"))
    p.add_argument("--index-name", default="index", help=h("help.index_name"))

    p = add("convert", "help.convert")
    p.add_argument("src", help=h("help.src"))
    p.add_argument("dst", help=h("help.dst"))
    p.add_argument("--overwrite", action="store_true", help=h("help.overwrite"))
    p.add_argument("--distance-strategy", type=str.upper, choices=_STRATEGIES, help=h("help.distance_strategy"))
    p.add_argument("--normalize-l2", action="store_true", default=None, help=h("help.normalize_l2"))
    p.add_argument("--json", action="store_true", help=h("help.json"))
    p.add_argument("--index-name", default="index", help=h("help.index_name"))

    p = add("verify", "help.verify")
    p.add_argument("converted", help=h("help.converted"))
    p.add_argument("--against", metavar="ORIGINAL", help=h("help.against"))
    p.add_argument("--trust-original", action="store_true", help=h("help.trust_original"))
    p.add_argument("--k", type=int, default=5, help=h("help.k"))
    p.add_argument("--probes", type=int, default=20, help=h("help.probes"))
    p.add_argument("--json", action="store_true", help=h("help.json"))
    p.add_argument("--index-name", default="index", help=h("help.index_name"))

    p = add("demo", "help.demo", with_limits=False)
    p.add_argument("--out", default="faiss-defuse-demo", help=h("help.out"))
    return parser


# ----------------------------------------------------------------------------- formatting
def _short_imports(report: ScanReport) -> str:
    names = [i.rsplit(".", 1)[1] if tuple(i.rsplit(".", 1)) in STRUCT else i for i in report.imports]
    return ", ".join(names) or "-"


def _fmt(lang: str, key: str, fallback: str, **params: Any) -> str:
    """Localised text for ``key``; English output and unknown keys use the library's own ``fallback`` text."""
    if lang == "en":
        return fallback
    try:
        return t(lang, key, **params)
    except (KeyError, IndexError):
        return fallback


def _malformed_text(lang: str, exc: MalformedStoreError) -> str:
    if lang == "en":
        text = exc.detail
    else:
        key = f"error.detail.{exc.code}"
        if key not in messages(lang):
            key = "error.detail.malformed-store"
        text = _fmt(lang, key, exc.detail, **exc.params)
    if exc.code == "limit-exceeded" and exc.params.get("limit") in _LIMIT_NAMES:
        text += " " + t(lang, "error.limit_hint", limit=exc.params["limit"])  # how a CLI user raises it
    return text


def _finding_message(lang: str, code: str) -> str:
    try:
        return t(lang, f"finding.{code}")
    except KeyError:
        return code


def _print_scan(report: ScanReport, lang: str) -> None:
    print(t(lang, "scan.summary", verdict=report.verdict.upper(), name=Path(report.source).name or report.source,
            imports=_short_imports(report), protocol=report.protocol,
            opcodes=f"{sum(report.opcode_counts.values()):,}", size=f"{report.size:,}"))
    for f in report.findings:
        target = f"{f.module}.{f.name}" if f.module or f.name else "-"
        print(t(lang, "scan.finding", offset=f"0x{f.offset:x}" if f.offset >= 0 else "?", opcode=f.opcode or "-",
                target=target, code=f.code, message=_finding_message(lang, f.code)))
    print(t(lang, "scan.executed_nothing"))


def _scan_json(report: ScanReport, lang: str) -> dict[str, Any]:
    data = report.to_dict()
    for entry in data["findings"]:
        entry["message"] = _finding_message(lang, entry["code"])
    return data


def _dump(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False))


# ----------------------------------------------------------------------------- commands
def _cmd_scan(args: argparse.Namespace, lang: str) -> int:
    report = scan(args.path, index_name=args.index_name, limits=_limits(args))
    if args.json:
        _dump(_scan_json(report, lang))
    else:
        _print_scan(report, lang)
    return EXIT_OK if report.safe else EXIT_FAIL


def _cmd_inspect(args: argparse.Namespace, lang: str) -> int:
    info = _store.inspect(args.path, index_name=args.index_name, limits=_limits(args))
    if args.json:
        data = info.to_dict()
        if info.scan:
            data["scan"] = _scan_json(info.scan, lang)
        _dump(data)
        return EXIT_OK if info.ok else EXIT_FAIL
    if info.kind == "converted":
        print(t(lang, "inspect.converted"))
    if info.header:
        h = info.header
        print(t(lang, "inspect.index", index_type=h.index_type, fourcc=h.fourcc, d=h.d, ntotal=h.ntotal, metric=h.metric_name))
    if info.header is None and info.header_error is not None:
        print(t(lang, "inspect.index_missing", detail=_malformed_text(lang, info.header_error)))
    if info.scan:
        print(t(lang, "inspect.pickle", verdict=info.scan.verdict.upper(), protocol=info.scan.protocol, imports=_short_imports(info.scan)))
        if not info.scan.safe:
            _print_scan(info.scan, lang)
    if info.doc_count is not None:
        print(t(lang, "inspect.docs", count=f"{info.doc_count:,}", orphans=info.orphan_docs))
        if info.mapping_problems:
            for code, params, text in info.mapping_items:
                print(t(lang, "inspect.mapping_bad", problem=_fmt(lang, f"mapping.{code}", text, **params)))
        elif info.header:
            print(t(lang, "inspect.mapping_ok"))
        keys = ", ".join(f"{k} {v}" for k, v in list(info.metadata_keys.items())[:12]) or "-"
        print(t(lang, "inspect.keys", keys=keys))
    if info.module_generation:
        print(t(lang, "inspect.generation", generation=info.module_generation))
    if info.faiss_options:
        o = info.faiss_options
        print(t(lang, "inspect.options", strategy=o["distance_strategy"], normalize=o["normalize_L2"], source=o.get("source", "?")))
    return EXIT_OK if info.ok else EXIT_FAIL


def _cmd_convert(args: argparse.Namespace, lang: str) -> int:
    manifest = _store.convert(
        args.src, args.dst, distance_strategy=args.distance_strategy, normalize_l2=args.normalize_l2,
        overwrite=args.overwrite, index_name=args.index_name, limits=_limits(args),
    )
    if args.json:
        _dump(manifest.to_dict())
        return EXIT_OK
    o = manifest.faiss_options
    print(t(lang, "convert.done", count=manifest.doc_count, dst=args.dst))
    print(t(lang, "convert.options", strategy=o["distance_strategy"], normalize=o["normalize_L2"], source=o["source"]))
    for w in manifest.warnings:
        detail = _fmt(lang, f"warn.{w['code']}", w["detail"], **w.get("params", {}))
        print(t(lang, "convert.warning", code=w["code"], detail=detail))
    return EXIT_OK


def _warning_text(lang: str, code: str, text: str) -> str:
    """The reason part of a verify warning, in the chosen language."""
    prefix = "top-k check skipped: "
    plain = text[len(prefix):] if text.startswith(prefix) else text
    key = f"verify.reason.{code}"
    return t(lang, key) if key in messages(lang) else plain


def _cmd_verify(args: argparse.Namespace, lang: str) -> int:
    if args.trust_original and not args.against:
        raise OptionConflictError(t(lang, "error.trust_needs_against"))
    if args.k < 1 or args.probes < 1:
        raise ValueError(t(lang, "error.k_probes"))
    report = _verify.verify(
        args.converted, original=args.against, trust_original=args.trust_original,
        k=args.k, probes=args.probes, index_name=args.index_name, limits=_limits(args),
    )
    if args.json:
        _dump(report.to_dict())
        return EXIT_OK if report.verified else EXIT_FAIL
    notes = list(zip(report.warning_codes, report.warnings))
    if report.verified:
        if not report.compared_with_original:
            print(t(lang, "verify.ok_integrity"))
        elif report.topk_equal:
            print(t(lang, "verify.ok_full", docs=report.docs_checked, k=report.k, probes=report.probes_run))
        else:
            reason = "-"
            if notes:  # the first warning is the reason; do not print it a second time as a note
                code, text = notes.pop(0)
                reason = _warning_text(lang, code, text)
            print(t(lang, "verify.ok_no_topk", docs=report.docs_checked, reason=reason))
        if args.trust_original:
            print(t(lang, "verify.trusted_note"))
    else:
        if report.mismatches:
            shown = ", ".join(report.mismatches[:10]) + (" ..." if len(report.mismatches) > 10 else "")
            print(t(lang, "verify.fail", count=len(report.mismatches), ids=shown))
        for c in report.checks:
            if not c["ok"] and c["name"] != "documents":
                print(t(lang, "verify.fail_check", name=c["name"]))
    for code, text in notes:
        print(t(lang, "verify.warning", detail=_warning_text(lang, code, text)))
    return EXIT_OK if report.verified else EXIT_FAIL


def _cmd_demo(args: argparse.Namespace, lang: str) -> int:
    from ._demo import run_demo

    print(t(lang, "demo.intro"))
    result = run_demo(Path(args.out))
    print(f"clean: {result.clean.verdict.upper()}")
    print(f"malicious: {result.malicious.verdict.upper()}")
    print(f"legacy: {result.legacy.verdict.upper()}")
    print(f"converted: {'VERIFIED' if result.verify.verified else 'MISMATCH'}")
    for f in result.malicious.findings[:1]:
        print(t(lang, "scan.finding", offset=f"0x{f.offset:x}", opcode=f.opcode, target=f"{f.module}.{f.name}",
                code=f.code, message=_finding_message(lang, f.code)))
    print(t(lang, "demo.sentinel_bad" if result.sentinel_exists else "demo.sentinel_ok"))
    print(t(lang, "demo.out", out=result.out))
    ok = (
        result.clean.safe and result.legacy.safe and result.malicious.verdict == "unsafe"
        and result.verify.verified and not result.sentinel_exists
    )
    return EXIT_OK if ok else EXIT_FAIL


_COMMANDS = {"scan": _cmd_scan, "inspect": _cmd_inspect, "convert": _cmd_convert, "verify": _cmd_verify, "demo": _cmd_demo}


def _error(lang: str, key: str, as_json: bool, detail: str, exc: BaseException, **extra: Any) -> None:
    message = t(lang, key, detail=detail, **extra)
    sys.stdout.flush()  # keep the order of stdout and stderr when both go to a terminal
    if as_json:
        _dump({"error": {"type": type(exc).__name__, "message": detail}})
    else:
        print(f"{t(lang, 'error.prefix')}: {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns the process exit code."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    lang = resolve_lang(_prescan_lang(args_list))
    parser = build_parser(lang)
    args = parser.parse_args(args_list)
    lang = resolve_lang(getattr(args, "lang", None))
    as_json = bool(getattr(args, "json", False))
    try:
        return _COMMANDS[args.command](args, lang)
    except UnsafePickleError as exc:
        if as_json:
            _dump(_scan_json(exc.report, lang))
        else:
            _print_scan(exc.report, lang)
        return EXIT_FAIL
    except UnsupportedContentError as exc:
        if as_json:
            _dump(_scan_json(exc.report, lang))
        else:
            _print_scan(exc.report, lang)
            sys.stdout.flush()
            print(_fmt(lang, "error.unsupported", str(exc), names=exc.names), file=sys.stderr)
        return EXIT_FAIL
    except ManifestMismatchError as exc:
        text = _fmt(lang, "error.manifest_mismatch", str(exc), file=exc.file, expected=exc.expected, actual=exc.actual)
        print(json.dumps({"error": {"type": "ManifestMismatchError", "message": str(exc)}}) if as_json
              else t(lang, "verify.fail_error", detail=text))
        return EXIT_FAIL
    except OptionConflictError as exc:
        _error(lang, "error.conflict", as_json, str(exc), exc)
        return EXIT_USAGE
    except OutputExistsError as exc:
        text = _fmt(lang, f"error.exists.{exc.code}", str(exc), path=exc.path)
        _error(lang, "error.exists", as_json, text, exc)
        return EXIT_USAGE
    except MissingExtraError as exc:
        text = _fmt(lang, "error.missing_extra_detail", str(exc), what=exc.what)
        _error(lang, "error.missing_extra", as_json, text, exc)
        return EXIT_EXTRA
    except MalformedStoreError as exc:
        _error(lang, "error.malformed", as_json, _malformed_text(lang, exc), exc, code=exc.code)
        return EXIT_DAMAGED
    except OSError as exc:
        _error(lang, "error.io", as_json, f"{exc.filename or ''} {exc.strerror or exc}".strip(), exc)
        return EXIT_DAMAGED
    except DefuseError as exc:
        _error(lang, "error.other", as_json, str(exc), exc)
        return EXIT_FAIL
    except ValueError as exc:  # invalid arguments caught by the library (for example k < 1)
        _error(lang, "error.other", as_json, str(exc), exc)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
