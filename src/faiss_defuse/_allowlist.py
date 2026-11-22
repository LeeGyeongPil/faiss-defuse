"""Allow-lists.  Everything not listed here is refused by the scanner.

Two groups:

* ``STRUCT`` - the classes LangChain itself writes into ``index.pkl``.
* ``VALUES`` - plain value types that may appear inside document metadata.

The decoder never imports any of these modules.  It maps each pair to an inert
stub or to a small function of our own (see ``_stub_decode``).
"""

from __future__ import annotations

DOCSTORE_NAMES = frozenset(
    {
        ("langchain_community.docstore.in_memory", "InMemoryDocstore"),
        ("langchain.docstore.in_memory", "InMemoryDocstore"),
    }
)

DOCUMENT_NAMES = frozenset(
    {
        ("langchain_core.documents.base", "Document"),
        ("langchain.schema.document", "Document"),
        ("langchain.docstore.document", "Document"),
    }
)

STRUCT = DOCSTORE_NAMES | DOCUMENT_NAMES

VALUES = frozenset(
    {
        ("builtins", "set"),
        ("__builtin__", "set"),
        ("builtins", "frozenset"),
        ("__builtin__", "frozenset"),
        ("builtins", "object"),
        ("__builtin__", "object"),
        ("collections", "OrderedDict"),
        ("copyreg", "_reconstructor"),
        ("_codecs", "encode"),
        ("datetime", "date"),
        ("datetime", "datetime"),
        ("datetime", "time"),
        ("datetime", "timedelta"),
        ("datetime", "timezone"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
    }
)

ALLOWED_GLOBALS = STRUCT | VALUES

# Types that are known and harmless but that this version cannot turn into JSON.
# They are reported as "unsupported", not "unsafe".
KNOWN_UNSUPPORTED = frozenset(
    {
        ("pandas._libs.tslibs.timestamps", "_unpickle_timestamp"),
        ("pandas._libs.tslibs.timestamps", "Timestamp"),
        ("pandas._libs.tslibs.timedeltas", "_timedelta_unpickle"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("decimal", "Decimal"),
        ("uuid", "UUID"),
    }
)

# Modules LangChain used before the split into langchain-core / -community.
LEGACY_MODULE_PREFIXES = ("langchain.",)
