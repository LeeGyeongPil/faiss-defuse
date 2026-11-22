from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from faiss_defuse import _jsontags, convert, load_safe
from faiss_defuse.errors import MalformedStoreError

UTC = dt.timezone.utc

SAMPLES = {
    "date": dt.date(2025, 3, 4),
    "naive": dt.datetime(2025, 3, 4, 5, 6, 7, 8),
    "aware": dt.datetime(2025, 3, 4, 5, 6, 7, tzinfo=dt.timezone(dt.timedelta(hours=9))),
    "utc": dt.datetime(2025, 1, 1, tzinfo=UTC),
    "time": dt.time(1, 2, 3),
    "delta": dt.timedelta(days=2, seconds=3, microseconds=4),
    "tuple": (1, "a", (2, 3)),
    "set": {3, 1, 2},
    "frozenset": frozenset({"x", "y"}),
    "bytes": b"\x00\xffabc",
    "intkeys": {1: "a", 2: "b"},
    "tuplekey": {(1, 2): "pair"},
    "dollar": {"$date": "not a tag", "x": 1},
    "single_dollar": {"$tuple": [1]},
    "inf": math.inf,
    "ninf": -math.inf,
    "none": None,
    "bool": True,
    "nested": [{"a": (1, {2})}, [dt.date(2020, 1, 1)]],
    "big": 2**80,
    "text": "한국어 ✓",
}


def test_roundtrip_all_tags() -> None:
    for name, value in SAMPLES.items():
        encoded = _jsontags.encode(value)
        again = _jsontags.decode(json.loads(json.dumps(encoded, allow_nan=False)))
        assert again == value, name
        assert type(again) is type(value), name
    nan = _jsontags.decode(json.loads(json.dumps(_jsontags.encode(math.nan), allow_nan=False)))
    assert isinstance(nan, float) and math.isnan(nan)


def test_encoding_is_deterministic_for_sets() -> None:
    assert _jsontags.encode({3, 1, 2}) == _jsontags.encode({1, 2, 3})


def test_unsupported_type_names_the_type() -> None:
    with pytest.raises(_jsontags.UnsupportedType) as exc:
        _jsontags.encode({"x": object()})
    assert exc.value.type_name == "builtins.object"


def test_decode_rejects_bad_tags() -> None:
    for bad in ({"$nope": 1}, {"$date": "garbage"}, {"$bytes": "!!!"}, {"$tuple": 5}, {"$date": "2020-01-01", "x": 1}):
        with pytest.raises(MalformedStoreError):
            _jsontags.decode(bad)


def test_depth_limit() -> None:
    deep: list = []
    cur = deep
    for _ in range(200):
        nxt: list = []
        cur.append(nxt)
        cur = nxt
    with pytest.raises(MalformedStoreError):
        _jsontags.encode(deep, max_depth=50)


def test_metadata_survives_convert_and_load_safe(tmp_path: Path, emb) -> None:
    """End to end: a real save_local store with awkward metadata (including numpy scalars)."""
    metadata = {
        "date": dt.date(2024, 1, 1),
        "when": dt.datetime(2024, 1, 1, 12, tzinfo=UTC),
        "pair": (1, 2),
        "tags": {"a", "b"},
        "raw": b"\x01\x02",
        "by_int": {1: "one"},
        "$weird": {"$date": "x"},
        "nan": math.nan,
        "f32": np.float32(1.5),
        "i64": np.int64(7),
    }
    FAISS.from_documents([Document(page_content="hello", metadata=metadata, id="d1"),
                          Document(page_content="world", metadata={}, id="d2")], emb).save_local(str(tmp_path / "src"))
    convert(tmp_path / "src", tmp_path / "dst")
    store = load_safe(tmp_path / "dst", emb)
    got = next(d for d in store.docstore._dict.values() if d.page_content == "hello").metadata
    assert math.isnan(got.pop("nan"))
    expected = dict(metadata)
    expected.pop("nan")
    expected["f32"], expected["i64"] = 1.5, 7
    assert got == expected
    assert type(got["f32"]) is float and type(got["i64"]) is int
    # the original-pickle path decodes the same values
    again = load_safe(tmp_path / "src", emb)
    got2 = next(d for d in again.docstore._dict.values() if d.page_content == "hello").metadata
    assert got2["pair"] == (1, 2) and got2["date"] == dt.date(2024, 1, 1)
