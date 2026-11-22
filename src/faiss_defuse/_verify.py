"""``verify``: integrity, content and top-k equivalence between a converted store and its original."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import _adapter
from ._store import build_records, records_for, load_original, read_converted, require_mapping
from .limits import Limits


@dataclass
class VerifyReport:
    """Outcome of :func:`verify`.  ``verified`` is true only if every executed check passed."""

    verified: bool
    docs_checked: int = 0
    mismatches: list[str] = field(default_factory=list)  # document ids whose content differs
    probes_run: int = 0
    k: int = 0
    topk_equal: bool | None = None  # None = not run
    compared_with_original: bool = False
    checks: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    warning_codes: list[str] = field(default_factory=list)  # one code per entry of ``warnings`` (for localised output)

    def warn(self, code: str, text: str) -> None:
        """Record a warning: English ``text`` for JSON/library users, ``code`` for the CLI's own language."""
        self.warnings.append(text)
        self.warning_codes.append(code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "docs_checked": self.docs_checked,
            "mismatches": list(self.mismatches),
            "probes_run": self.probes_run,
            "k": self.k,
            "topk_equal": self.topk_equal,
            "compared_with_original": self.compared_with_original,
            "checks": list(self.checks),
            "warnings": list(self.warnings),
        }


def _by_pos(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {r["pos"]: r for r in records}


def _compare_records(left: list[dict[str, Any]], right: list[dict[str, Any]], *, keys: tuple[str, ...]) -> list[str]:
    a, b = _by_pos(left), _by_pos(right)
    bad: list[str] = []
    for pos in sorted(set(a) | set(b)):
        ra, rb = a.get(pos), b.get(pos)
        if ra is None or rb is None or any(ra.get(k) != rb.get(k) for k in keys):
            bad.append((rb or ra)["id"])
    return bad


def _probe_positions(positions: list[int], probes: int) -> list[int]:
    if not positions:
        return []
    count = min(probes, len(positions))
    step = len(positions) / count
    return [positions[int(i * step)] for i in range(count)]


def verify(
    converted: str | Path,
    *,
    original: str | Path | None = None,
    trust_original: bool = False,
    k: int = 5,
    probes: int = 20,
    index_name: str = "index",
    limits: Limits = Limits(),
) -> VerifyReport:
    """Check a pickle-free store.

    Always: manifest hashes, header agreement, document/mapping consistency.  With
    ``original`` (scanned and decoded without executing it): identical ``index.faiss`` bytes,
    identical document content, and identical top-``k`` (document id, score) results for
    ``probes`` stored vectors used as queries (needs the ``[langchain]`` extra; no embedding
    model is called).  ``trust_original=True`` additionally compares against LangChain's own
    ``load_local`` result - this *executes* the original pickle, so use it only on stores you built.
    """
    if k < 1 or probes < 1:
        raise ValueError("k and probes must be at least 1.")
    if trust_original and original is None:
        raise ValueError("trust_original requires original=<path of the original store>.")
    report = VerifyReport(verified=False, k=k)
    conv = read_converted(converted, limits=limits)
    report.checks.append({"name": "manifest-hashes", "ok": True})
    report.docs_checked = len(conv.records)

    orig = None
    if original is not None:
        report.compared_with_original = True
        orig = load_original(original, index_name=index_name, limits=limits)
        require_mapping(orig.header, orig.store.index_to_docstore_id, set(orig.store.docs))
        same_index = orig.faiss_path.read_bytes() == conv.faiss_path.read_bytes()
        report.checks.append({"name": "index-bytes", "ok": same_index})
        orig_records = build_records(orig.store, limits=limits)
        report.docs_checked = len(orig_records)
        bad = _compare_records(orig_records, conv.records, keys=("id", "doc_id", "page_content", "metadata", "extra"))
        report.mismatches.extend(bad)
        report.checks.append({"name": "documents", "ok": not bad, "detail": f"{len(bad)} differ"})

    if trust_original and original is not None:
        try:
            docs, idmap = _adapter.load_local_trusted(original, conv.header.d, index_name)
        except Exception as exc:  # LangChain could not open it (for example legacy module paths)
            report.checks.append({"name": "trusted-load-local", "ok": False, "detail": f"{type(exc).__name__}: {exc}"})
        else:
            live = records_for(((pos, idmap[pos], docs[idmap[pos]]) for pos in sorted(idmap)), limits)
            bad_live = _compare_records(live, conv.records, keys=("id", "doc_id", "page_content", "metadata"))
            report.mismatches.extend(x for x in bad_live if x not in report.mismatches)
            report.checks.append({"name": "trusted-load-local", "ok": not bad_live, "detail": f"{len(bad_live)} differ"})

    if _adapter.faiss_available():
        _run_topk(report, conv, orig, k, probes)
    else:
        report.warn("topk-no-faiss", "top-k check skipped: faiss/numpy are not installed (pip install \"faiss-defuse[langchain]\")")

    report.verified = all(c["ok"] for c in report.checks) and not report.mismatches
    return report


def _run_topk(report: VerifyReport, conv: Any, orig: Any, k: int, probes: int) -> None:
    import numpy as np

    ntotal = conv.header.ntotal
    if ntotal == 0:
        report.warn("topk-empty", "top-k check skipped: the index is empty")
        return
    kk = min(k, ntotal)
    report.k = kk
    index_c = _adapter.read_index(conv.faiss_path)
    index_o = _adapter.read_index(orig.faiss_path) if orig is not None else None
    positions = _probe_positions(sorted(conv.index_to_docstore_id), probes)
    vectors = []
    for pos in positions:
        try:
            vectors.append(np.asarray(index_c.reconstruct(int(pos)), dtype="float32"))
        except RuntimeError:
            report.warn("topk-no-reconstruct", "top-k check partly skipped: this index type cannot reconstruct stored vectors")
            break
    if not vectors:
        return
    queries = np.stack(vectors)
    dc, lc = index_c.search(queries, kk)
    report.probes_run = len(vectors)
    if index_o is None:
        report.checks.append({"name": "search-runs", "ok": True})
        return
    do, lo = index_o.search(queries, kk)
    idmap_c, idmap_o = conv.index_to_docstore_id, orig.store.index_to_docstore_id
    ids_c = [[idmap_c.get(int(x)) for x in row] for row in lc]
    ids_o = [[idmap_o.get(int(x)) for x in row] for row in lo]
    equal = ids_c == ids_o and bool(np.array_equal(dc, do))
    report.topk_equal = equal
    report.checks.append({"name": "top-k", "ok": equal, "detail": f"{len(vectors)} probes, k={kk}"})
