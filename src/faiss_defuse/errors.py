"""Exception hierarchy.  Every error raised on purpose derives from :class:`DefuseError`."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._pickle_scan import ScanReport


class DefuseError(Exception):
    """Base class for all faiss-defuse errors."""


class UnsafePickleError(DefuseError):
    """The pickle contains something outside the allow-list (or cannot be proven harmless)."""

    def __init__(self, report: "ScanReport") -> None:
        self.report = report
        codes = ", ".join(sorted({f.code for f in report.findings})) or "unknown"
        super().__init__(
            f"refusing to open pickle: verdict 'unsafe' ({codes}). "
            "Nothing was executed. Run `faiss-defuse scan <path>` for the evidence."
        )


class UnsupportedContentError(DefuseError):
    """The pickle looks harmless but holds a value type this version cannot convert."""

    def __init__(self, report: "ScanReport", detail: str = "") -> None:
        self.report = report
        self.detail = detail
        names = ", ".join(
            sorted({f"{f.module}.{f.name}" for f in report.findings if f.severity == "unsupported"})
        )
        self.names = names or detail or "unknown"
        msg = f"cannot convert store: unsupported value type ({names or detail or 'unknown'})."
        if detail and names:
            msg += f" {detail}"
        msg += " This is not a safety verdict; please open an issue with the type name."
        super().__init__(msg)


class MalformedStoreError(DefuseError):
    """A store file is damaged, inconsistent or has an unexpected structure."""

    def __init__(self, message: str, *, code: str = "malformed-store", params: dict[str, str] | None = None) -> None:
        self.code = code
        self.params: dict[str, str] = dict(params or {})
        self.detail = message
        super().__init__(f"[{code}] {message}")


class ManifestMismatchError(DefuseError):
    """A file does not match the SHA-256 (or size) recorded in ``manifest.json``."""

    def __init__(self, file: str, expected: str, actual: str) -> None:
        self.file = file
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"{file} does not match manifest.json (expected {expected}, got {actual}). "
            "The store was modified after it was written; rebuild it with `faiss-defuse convert`."
        )


class OptionConflictError(DefuseError):
    """The caller asked for options that differ from the ones recorded in the store."""


class OutputExistsError(DefuseError):
    """The destination already holds files and ``overwrite`` was not requested.

    ``code`` is ``not-a-directory``, ``holds-store`` or ``same-as-source`` (used to localise the CLI message).
    """

    def __init__(self, message: str, *, code: str = "holds-store", path: str = "") -> None:
        self.code = code
        self.path = path
        super().__init__(message)


class MissingExtraError(DefuseError):
    """An optional dependency is missing."""

    def __init__(self, what: str) -> None:
        self.what = what
        super().__init__(
            f"{what} requires the optional dependencies. "
            'Install them with: pip install "faiss-defuse[langchain]"'
        )
