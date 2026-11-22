"""Reading, inspecting and writing stores (the pickle-free format and the original pickle form)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from . import _jsontags
from ._docstate import DecodedDoc
from ._faiss_header import FaissHeader, read_faiss_header
from ._pickle_scan import ScanReport, resolve_pickle_path, scan_bytes
from ._stub_decode import DecodedStore, decode_checked
from .errors import (
    MalformedStoreError,
    ManifestMismatchError,
    OutputExistsError,
    UnsafePickleError,
    UnsupportedContentError,
)
from .limits import Limits

FORMAT = "faiss-defuse/1"
FAISS_FILE = "index.faiss"
DOCSTORE_FILE = "docstore.jsonl"
MANIFEST_FILE = "manifest.json"
VALID_STRATEGIES = ("EUCLIDEAN_DISTANCE", "MAX_INNER_PRODUCT", "DOT_PRODUCT", "JACCARD", "COSINE")
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024


def _version() -> str:
    from . import __version__

    return __version__


# --------------------------------------------------------------------------- manifest
@dataclass
class Manifest:
    """``manifest.json`` of a pickle-free store."""

    files: dict[str, dict[str, Any]]
    index: dict[str, Any]
    doc_count: int
    faiss_options: dict[str, Any]
    origin: dict[str, Any]
    warnings: list[dict[str, Any]] = field(default_factory=list)
    format: str = FORMAT
    created_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "created_by": self.created_by or f"faiss-defuse {_version()}",
            "files": self.files,
            "index": self.index,
            "doc_count": self.doc_count,
            "faiss_options": self.faiss_options,
            "origin": self.origin,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "Manifest":
        if not isinstance(raw, dict):
            raise MalformedStoreError("manifest.json must hold a JSON object.", code="bad-manifest")
        if raw.get("format") != FORMAT:
            raise MalformedStoreError(
                f"manifest format {raw.get('format')!r} is not {FORMAT!r}. "
                "Upgrade faiss-defuse or rebuild the store.",
                code="unsupported-format-version",
            )
        try:
            files = raw["files"]
            index = raw["index"]
            opts = raw["faiss_options"]
            if set(files) != {FAISS_FILE, DOCSTORE_FILE}:
                raise MalformedStoreError(
                    f"manifest lists unexpected files {sorted(files)}; expected {FAISS_FILE} and {DOCSTORE_FILE}.",
                    code="bad-manifest",
                )
            for meta in files.values():
                if not (isinstance(meta["sha256"], str) and type(meta["size"]) is int):
                    raise KeyError("sha256/size")
            if not isinstance(opts["normalize_L2"], bool):
                raise KeyError("normalize_L2")
            strategy = opts["distance_strategy"]
            if strategy is not None and strategy not in VALID_STRATEGIES:
                raise KeyError("distance_strategy")
            for key in ("fourcc", "d", "ntotal", "metric_type"):
                index[key]
            if type(raw["doc_count"]) is not int:
                raise KeyError("doc_count")
        except MalformedStoreError:
            raise
        except (KeyError, TypeError) as exc:
            raise MalformedStoreError(f"manifest.json is missing or has invalid field {exc}.", code="bad-manifest") from None
        return cls(
            files=files,
            index=index,
            doc_count=raw["doc_count"],
            faiss_options=opts,
            origin=raw.get("origin", {}),
            warnings=list(raw.get("warnings", [])),
            created_by=str(raw.get("created_by", "")),
        )


# --------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_strategy(value: Any) -> str:
    name = getattr(value, "name", None) or str(value)
    name = name.upper()
    if name not in VALID_STRATEGIES:
        raise ValueError(f"distance_strategy {value!r} is not one of {', '.join(VALID_STRATEGIES)}.")
    return name


def resolve_options(
    header: FaissHeader,
    distance_strategy: Any = None,
    normalize_l2: bool | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Options for a store that was written without them (original pickle form)."""
    warns: list[dict[str, Any]] = []
    from_user = distance_strategy is not None and normalize_l2 is not None
    if distance_strategy is not None:
        strategy: str | None = _norm_strategy(distance_strategy)
    elif header.metric_type == 1:
        strategy = "EUCLIDEAN_DISTANCE"
    elif header.metric_type == 0:
        strategy = "MAX_INNER_PRODUCT"
    else:
        strategy = None
        warns.append({"code": "distance-strategy-unknown",
                      "detail": f"index metric {header.metric_name} has no LangChain DistanceStrategy; "
                                "pass distance_strategy explicitly (CLI: --distance-strategy)",
                      "params": {"metric": header.metric_name}})
    if normalize_l2 is None:
        normalize = False
        warns.append({"code": "normalize-l2-unknown",
                      "detail": "normalize_L2 is not stored in the original pickle and cannot be inferred; assumed False. "
                                "Pass normalize_l2=True (CLI: --normalize-l2) if the store was built with it.",
                      "params": {}})
    else:
        normalize = bool(normalize_l2)
    if distance_strategy is None and strategy is not None:
        warns.append({"code": "distance-strategy-inferred",
                      "detail": f"distance_strategy {strategy} inferred from the index metric ({header.metric_name})",
                      "params": {"strategy": strategy, "metric": header.metric_name}})
    return {"distance_strategy": strategy, "normalize_L2": normalize, "source": "user" if from_user else "inferred"}, warns


MappingProblem = tuple[str, dict[str, Any], str]  # (code, parameters, English text)


def check_mapping_items(header: FaissHeader, idmap: dict[int, str], doc_ids: set[str]) -> tuple[list[MappingProblem], int]:
    """Return (problems with code and parameters, orphan_document_count)."""
    problems: list[MappingProblem] = []
    if len(idmap) != header.ntotal:
        problems.append(("count", {"entries": len(idmap), "ntotal": header.ntotal},
                         f"index_to_docstore_id has {len(idmap)} entries but the index holds {header.ntotal} vectors"))
    if not header.id_mapped:
        missing = sorted(set(range(header.ntotal)) - set(idmap))
        extra = sorted(set(idmap) - set(range(header.ntotal)))
        if missing:
            problems.append(("no-id", {"count": len(missing), "first": str(missing[:5])},
                             f"{len(missing)} vector positions have no document id (first: {missing[:5]})"))
        if extra:
            problems.append(("outside", {"count": len(extra), "first": str(extra[:5])},
                             f"{len(extra)} mapped positions are outside the index (first: {extra[:5]})"))
    absent = sorted({v for v in idmap.values() if v not in doc_ids})
    if absent:
        problems.append(("absent", {"count": len(absent), "first": str(absent[:3])},
                         f"{len(absent)} mapped document ids are missing from the docstore (first: {absent[:3]})"))
    orphans = len(doc_ids - set(idmap.values()))
    return problems, orphans


def check_mapping(header: FaissHeader, idmap: dict[int, str], doc_ids: set[str]) -> tuple[list[str], int]:
    """Return (problems, orphan_document_count)."""
    items, orphans = check_mapping_items(header, idmap, doc_ids)
    return [text for _, _, text in items], orphans


def require_mapping(header: FaissHeader, idmap: dict[int, str], doc_ids: set[str]) -> int:
    problems, orphans = check_mapping(header, idmap, doc_ids)
    if problems:
        raise MalformedStoreError("; ".join(problems) + ".", code="mapping-mismatch")
    return orphans


def build_records(store: DecodedStore, *, limits: Limits = Limits()) -> list[dict[str, Any]]:
    """Turn a decoded store into JSONL records (typed tags applied, expanded size bounded by ``limits``)."""
    return records_for(
        ((pos, store.index_to_docstore_id[pos], store.docs[store.index_to_docstore_id[pos]])
         for pos in sorted(store.index_to_docstore_id)),
        limits,
    )


def records_for(items: Iterable[tuple[int, str, DecodedDoc]], limits: Limits = Limits()) -> list[dict[str, Any]]:
    """Encode ``(position, id, document)`` triples; one budget covers the whole store."""
    budget = _jsontags.Budget.from_limits(limits)
    records: list[dict[str, Any]] = []
    for pos, doc_id, doc in items:
        budget.start_document()
        records.append(_record(pos, doc_id, doc, limits.max_depth, budget))
    return records


def _record(pos: int, doc_id: str, doc: DecodedDoc, max_depth: int, budget: _jsontags.Budget | None = None) -> dict[str, Any]:
    try:
        if budget is not None:
            budget.size(len(doc_id) + len(doc.page_content) + len(doc.doc_id or "") + 64)
        rec: dict[str, Any] = {
            "pos": pos,
            "id": doc_id,
            "doc_id": doc.doc_id,
            "page_content": doc.page_content,
            "metadata": _jsontags.encode(doc.metadata, max_depth=max_depth, budget=budget),
        }
        if doc.extra:
            rec["extra"] = _jsontags.encode(doc.extra, max_depth=max_depth, budget=budget)
    except _jsontags.UnsupportedType as exc:
        raise _unsupported(exc.type_name, doc_id) from None
    return rec


def _unsupported(type_name: str, doc_id: str) -> UnsupportedContentError:
    from ._pickle_scan import Finding

    module, _, name = type_name.rpartition(".")
    finding = Finding("unsupported-value-type", "unsupported", -1, "", module, name,
                      f"metadata value of document {doc_id!r}")
    report = ScanReport("unsupported", [finding])
    return UnsupportedContentError(report, f"found in document {doc_id!r}.")


_JSONL_ENCODER = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), allow_nan=False)
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")
_WRITE_BATCH = 1 << 16  # characters collected before one write


def _utf8(text: str) -> bytes:
    """UTF-8 bytes of a JSON fragment.  Lone surrogates (only possible inside strings) become ``\\uXXXX`` escapes."""
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        return _LONE_SURROGATE.sub(lambda m: f"\\u{ord(m.group()):04x}", text).encode("utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]], limits: Limits = Limits()) -> None:
    """Write one JSON object per line, streaming.

    The JSON text is produced piece by piece (``iterencode``) and the UTF-8 bytes are counted
    as they are written, so no whole line is ever built in memory and the file stops at
    ``limits.max_docstore_bytes`` - the same byte limit :func:`read_jsonl` applies.
    """
    written = 0
    batch: list[str] = []
    batch_chars = 0
    with path.open("wb") as fh:
        for chunk in _chunks(records):  # small pieces of the lines, never a whole line
            batch.append(chunk)
            batch_chars += len(chunk)
            if batch_chars >= _WRITE_BATCH:
                written = _write_batch(fh, batch, written, limits)
                batch, batch_chars = [], 0
        _write_batch(fh, batch, written, limits)


def _chunks(records: list[dict[str, Any]]) -> Iterable[str]:
    for rec in records:
        yield from _JSONL_ENCODER.iterencode(rec)
        yield "\n"


def _write_batch(fh: Any, batch: list[str], written: int, limits: Limits) -> int:
    data = _utf8("".join(batch))
    written += len(data)
    if written > limits.max_docstore_bytes:  # checked before the bytes reach the file
        raise MalformedStoreError(
            f"{DOCSTORE_FILE} would be larger than {limits.max_docstore_bytes} bytes. "
            "Raise Limits.max_docstore_bytes if this is expected.",
            code="limit-exceeded", params={"limit": "max_docstore_bytes"},
        )
    fh.write(data)
    return written


def read_jsonl(path: Path, limits: Limits) -> list[dict[str, Any]]:
    if path.stat().st_size > limits.max_docstore_bytes:
        raise MalformedStoreError(f"{path.name} is larger than Limits.max_docstore_bytes.", code="limit-exceeded",
                                  params={"limit": "max_docstore_bytes"})
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="\n") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if not (isinstance(rec, dict) and type(rec["pos"]) is int and isinstance(rec["id"], str)
                        and isinstance(rec["page_content"], str) and "metadata" in rec):
                    raise ValueError("bad record shape")
                if rec.get("doc_id") is not None and not isinstance(rec["doc_id"], str):
                    raise ValueError("doc_id must be a string or null")
            except (ValueError, KeyError, UnicodeDecodeError) as exc:
                raise MalformedStoreError(f"{path.name} line {n} is not a valid record ({exc}).", code="bad-docstore") from None
            out.append(rec)
    return out


def records_to_store(records: list[dict[str, Any]], limits: Limits) -> tuple[dict[str, DecodedDoc], dict[int, str]]:
    docs: dict[str, DecodedDoc] = {}
    idmap: dict[int, str] = {}
    for rec in records:
        if rec["pos"] in idmap:
            raise MalformedStoreError(f"duplicate position {rec['pos']} in {DOCSTORE_FILE}.", code="bad-docstore")
        metadata = _jsontags.decode(rec["metadata"], max_depth=limits.max_depth)
        if not isinstance(metadata, dict):
            raise MalformedStoreError(f"metadata of {rec['id']!r} is not an object.", code="bad-docstore")
        extra = _jsontags.decode(rec.get("extra", {}), max_depth=limits.max_depth)
        idmap[rec["pos"]] = rec["id"]
        docs[rec["id"]] = DecodedDoc(rec["page_content"], metadata, rec.get("doc_id"), extra if isinstance(extra, dict) else {})
    return docs, idmap


# --------------------------------------------------------------------------- original form
@dataclass
class OriginalStore:
    report: ScanReport
    header: FaissHeader
    store: DecodedStore
    faiss_path: Path
    pickle_path: Path


def load_original(src: str | Path, *, index_name: str = "index", limits: Limits = Limits()) -> OriginalStore:
    """Scan + decode an original ``index.pkl`` directory and validate ``index.faiss`` (no faiss import)."""
    p = Path(src)
    pkl = resolve_pickle_path(p, index_name)
    if pkl.stat().st_size > limits.max_pickle_bytes:
        raise MalformedStoreError(f"{pkl.name} is larger than Limits.max_pickle_bytes.", code="limit-exceeded",
                                  params={"limit": "max_pickle_bytes"})
    report, store = decode_checked(pkl.read_bytes(), limits=limits, source=str(pkl))
    faiss_path = pkl.with_name(f"{index_name}.faiss") if p.is_dir() else pkl.with_suffix(".faiss")
    header = read_faiss_header(faiss_path, limits=limits)
    return OriginalStore(report, header, store, faiss_path, pkl)


# --------------------------------------------------------------------------- converted form
@dataclass
class ConvertedStore:
    manifest: Manifest
    header: FaissHeader
    records: list[dict[str, Any]]
    docs: dict[str, DecodedDoc]
    index_to_docstore_id: dict[int, str]
    faiss_path: Path


def read_manifest(directory: Path) -> Manifest:
    mpath = directory / MANIFEST_FILE
    try:
        if mpath.stat().st_size > _MAX_MANIFEST_BYTES:
            raise MalformedStoreError("manifest.json is unreasonably large.", code="bad-manifest")
        raw = json.loads(mpath.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MalformedStoreError(f"cannot read {mpath}: {exc.strerror or exc}", code="missing-file",
                                  params={"path": str(mpath)}) from None
    except ValueError as exc:
        raise MalformedStoreError(f"manifest.json is not valid JSON ({exc}).", code="bad-manifest") from None
    return Manifest.from_dict(raw)


def verify_hashes(directory: Path, manifest: Manifest) -> None:
    for name, meta in manifest.files.items():
        path = directory / name
        if not path.is_file():
            raise MalformedStoreError(f"{name} is listed in manifest.json but missing.", code="missing-file",
                                      params={"path": str(path)})
        size = path.stat().st_size
        if size != meta["size"]:
            raise ManifestMismatchError(name, f"{meta['size']} bytes", f"{size} bytes")
        actual = sha256_file(path)
        if actual != meta["sha256"]:
            raise ManifestMismatchError(name, meta["sha256"], actual)


def read_converted(directory: str | Path, *, limits: Limits = Limits()) -> ConvertedStore:
    """Read a pickle-free store.  Hashes are checked before any file is parsed or handed to faiss."""
    d = Path(directory)
    manifest = read_manifest(d)
    verify_hashes(d, manifest)
    header = read_faiss_header(d / FAISS_FILE, limits=limits)
    for key in ("fourcc", "d", "ntotal", "metric_type"):
        if getattr(header, key) != manifest.index[key]:
            raise MalformedStoreError(f"index {key} in manifest.json differs from {FAISS_FILE}.", code="manifest-index-mismatch")
    records = read_jsonl(d / DOCSTORE_FILE, limits)
    docs, idmap = records_to_store(records, limits)
    if len(records) != manifest.doc_count:
        raise MalformedStoreError(
            f"manifest says {manifest.doc_count} documents but {DOCSTORE_FILE} has {len(records)}.", code="mapping-mismatch"
        )
    require_mapping(header, idmap, set(docs))
    return ConvertedStore(manifest, header, records, docs, idmap, d / FAISS_FILE)


def is_converted(path: Path) -> bool:
    return path.is_dir() and (path / MANIFEST_FILE).is_file()


# --------------------------------------------------------------------------- writing
def _prepare_dst(dst: Path, overwrite: bool) -> bool:
    """Make ``dst`` ready for a new store.  Returns True if the directory was created here."""
    managed = [dst / n for n in (FAISS_FILE, DOCSTORE_FILE, MANIFEST_FILE)]
    created = not dst.exists()
    if dst.exists() and not dst.is_dir():
        raise OutputExistsError(f"{dst} exists and is not a directory.", code="not-a-directory", path=str(dst))
    if any(p.exists() for p in managed) and not overwrite:
        raise OutputExistsError(
            f"{dst} already holds a store. Pass overwrite=True (CLI: --overwrite) to replace it.",
            code="holds-store", path=str(dst),
        )
    dst.mkdir(parents=True, exist_ok=True)
    for p in managed:
        if p.exists():
            p.unlink()
    return created


def write_store(
    dst: Path,
    *,
    write_index: Callable[[Path], None],
    records: list[dict[str, Any]],
    options: dict[str, Any],
    warns: list[dict[str, Any]],
    origin: dict[str, Any],
    overwrite: bool,
    check: Callable[[FaissHeader], None] | None = None,
    limits: Limits = Limits(),
) -> Manifest:
    created = _prepare_dst(dst, overwrite)
    try:
        write_index(dst / FAISS_FILE)
        header = read_faiss_header(dst / FAISS_FILE, limits=limits)
        if check is not None:
            check(header)
        write_jsonl(dst / DOCSTORE_FILE, records, limits)
        manifest = Manifest(
            files={n: {"sha256": sha256_file(dst / n), "size": (dst / n).stat().st_size} for n in (FAISS_FILE, DOCSTORE_FILE)},
            index={k: v for k, v in header.to_dict().items() if k in ("fourcc", "index_type", "d", "ntotal", "is_trained", "metric_type")},
            doc_count=len(records),
            faiss_options=options,
            origin=origin,
            warnings=warns,
        )
        (dst / MANIFEST_FILE).write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except BaseException:
        for name in (FAISS_FILE, DOCSTORE_FILE, MANIFEST_FILE):  # leave no half-written store behind
            (dst / name).unlink(missing_ok=True)
        if created and not any(dst.iterdir()):  # nor an empty directory we made ourselves
            dst.rmdir()
        raise
    return manifest


def convert(
    src: str | Path,
    dst: str | Path,
    *,
    distance_strategy: Any = None,
    normalize_l2: bool | None = None,
    overwrite: bool = False,
    index_name: str = "index",
    limits: Limits = Limits(),
) -> Manifest:
    """Convert an original ``index.pkl`` store into the pickle-free format.

    The pickle is scanned first and decoded with inert stubs; it is never executed.
    """
    srcp, dstp = Path(src), Path(dst)
    if dstp.resolve() == srcp.resolve():
        raise OutputExistsError("destination must differ from the source directory.", code="same-as-source", path=str(dstp))
    original = load_original(srcp, index_name=index_name, limits=limits)
    require_mapping(original.header, original.store.index_to_docstore_id, set(original.store.docs))
    try:
        records = build_records(original.store, limits=limits)
    except UnsupportedContentError as exc:  # show the real scan summary (size, imports) next to the finding
        report = replace(original.report, verdict="unsupported", findings=[*original.report.findings, *exc.report.findings])
        raise UnsupportedContentError(report, exc.detail) from None
    options, warns = resolve_options(original.header, distance_strategy, normalize_l2)
    return write_store(
        dstp,
        write_index=lambda target: shutil.copyfile(original.faiss_path, target),
        records=records,
        options=options,
        warns=warns,
        origin={"kind": "pickle", "legacy": original.store.legacy, "protocol": original.report.protocol},
        overwrite=overwrite,
        limits=limits,
    )


# --------------------------------------------------------------------------- inspect
@dataclass
class StoreInfo:
    """What ``inspect`` found.  Any field may be ``None`` when it could not be determined safely."""

    kind: str  # "pickle" | "converted"
    path: str
    header: FaissHeader | None = None
    scan: ScanReport | None = None
    doc_count: int | None = None
    mapping_problems: list[str] = field(default_factory=list)
    orphan_docs: int = 0
    metadata_keys: dict[str, int] = field(default_factory=dict)
    module_generation: str | None = None  # "current" | "legacy" | "mixed"
    faiss_options: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    mapping_items: list[MappingProblem] = field(default_factory=list)  # same problems, with code + parameters
    header_error: MalformedStoreError | None = None  # why index.faiss could not be read (for localised output)

    @property
    def ok(self) -> bool:
        return (
            self.header is not None
            and (self.scan is None or self.scan.safe)
            and not self.mapping_problems
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "ok": self.ok,
            "header": self.header.to_dict() if self.header else None,
            "scan": self.scan.to_dict() if self.scan else None,
            "doc_count": self.doc_count,
            "mapping_problems": list(self.mapping_problems),
            "orphan_docs": self.orphan_docs,
            "metadata_keys": dict(self.metadata_keys),
            "module_generation": self.module_generation,
            "faiss_options": self.faiss_options,
            "warnings": list(self.warnings),
        }


def _key_counts(docs: dict[str, DecodedDoc]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for doc in docs.values():
        for key in doc.metadata:
            counts[str(key)] = counts.get(str(key), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def inspect(path: str | Path, *, index_name: str = "index", limits: Limits = Limits()) -> StoreInfo:
    """Describe a store without opening it with faiss or pickle."""
    p = Path(path)
    if is_converted(p):
        conv = read_converted(p, limits=limits)
        orphans = len(set(conv.docs) - set(conv.index_to_docstore_id.values()))
        return StoreInfo(
            kind="converted", path=str(p), header=conv.header, doc_count=len(conv.docs),
            orphan_docs=orphans, metadata_keys=_key_counts(conv.docs),
            module_generation=("legacy" if conv.manifest.origin.get("legacy") else "current")
            if conv.manifest.origin.get("kind") == "pickle" else None,
            faiss_options=conv.manifest.faiss_options,
            warnings=[w.get("code", "") for w in conv.manifest.warnings],
        )
    pkl = resolve_pickle_path(p, index_name)
    info = StoreInfo(kind="pickle", path=str(p))
    faiss_path = pkl.with_name(f"{index_name}.faiss") if p.is_dir() else pkl.with_suffix(".faiss")
    try:
        info.header = read_faiss_header(faiss_path, limits=limits)
    except MalformedStoreError as exc:
        info.warnings.append(str(exc))
        info.header_error = exc
    if pkl.stat().st_size > limits.max_pickle_bytes:
        info.warnings.append("pickle larger than Limits.max_pickle_bytes; not scanned")
        return info
    data = pkl.read_bytes()
    info.scan = scan_bytes(data, limits=limits, source=str(pkl))
    if not info.scan.safe:
        return info
    try:
        _, store = decode_checked(data, limits=limits, source=str(pkl))
    except (UnsafePickleError, UnsupportedContentError) as exc:  # pragma: no cover - scan already passed
        info.warnings.append(str(exc))
        return info
    info.doc_count = len(store.docs)
    info.metadata_keys = _key_counts(store.docs)
    info.module_generation = "mixed" if store.mixed else ("legacy" if store.legacy else "current")
    if info.header is not None:
        info.mapping_items, info.orphan_docs = check_mapping_items(info.header, store.index_to_docstore_id, set(store.docs))
        info.mapping_problems = [text for _, _, text in info.mapping_items]
    return info


