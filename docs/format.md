# The `faiss-defuse/1` pickle-free format

A store is a directory with exactly three files.

| File | Content |
|---|---|
| `index.faiss` | The FAISS index, byte-for-byte as `faiss.write_index` wrote it |
| `docstore.jsonl` | One JSON object per line, one line per vector position |
| `manifest.json` | Format version, SHA-256 and size of the other two files, index facts, FAISS options |

## manifest.json

```json
{
  "format": "faiss-defuse/1",
  "created_by": "faiss-defuse 0.1.0",
  "files": {
    "index.faiss":    {"sha256": "...", "size": 2605},
    "docstore.jsonl": {"sha256": "...", "size": 4120}
  },
  "index": {"fourcc": "IxF2", "index_type": "IndexFlatL2", "d": 32, "ntotal": 20, "is_trained": true, "metric_type": 1},
  "doc_count": 20,
  "faiss_options": {"distance_strategy": "EUCLIDEAN_DISTANCE", "normalize_L2": false, "source": "inferred"},
  "origin": {"kind": "pickle", "legacy": false, "protocol": 4},
  "warnings": [{"code": "normalize-l2-unknown", "detail": "..."}]
}
```

* `faiss_options.source` is `recorded` (written by `save_safe` from a live object), `user` (both options given explicitly) or `inferred` (at least one option was guessed from the index metric). LangChain does not store `normalize_L2` or `distance_strategy` in `index.pkl`, so a converted store can only know them if somebody says so.
* A loader reads `manifest.json` first, checks that `files` lists exactly `index.faiss` and `docstore.jsonl`, compares size and SHA-256, and only then parses the other files or hands `index.faiss` to FAISS.

## docstore.jsonl

```json
{"pos":0,"id":"3f1c...","doc_id":"id-0","page_content":"text","metadata":{"page":0,"tags":["a","b"]}}
```

* `pos` is the vector position (or the id for `IndexIDMap`), `id` the docstore key, `doc_id` the `Document.id` attribute (`null` when absent).
* `extra` (optional) keeps unknown fields of old Pydantic v1 documents such as `lookup_str`; loaders ignore it.

## Type tags

JSON cannot hold some Python values exactly. Those are written as one-key objects whose key starts with `$`.

| Python value | JSON |
|---|---|
| `tuple` | `{"$tuple": [..]}` |
| `set` / `frozenset` | `{"$set": [..]}` / `{"$frozenset": [..]}` (items sorted by their JSON text) |
| `bytes` | `{"$bytes": "<base64>"}` |
| `datetime.date` / `datetime` / `time` | `{"$date": "2025-03-04"}`, `{"$datetime": "2025-03-04T05:06:07+09:00"}`, `{"$time": "01:02:03"}` |
| `timedelta` | `{"$timedelta": [days, seconds, microseconds]}` |
| `nan`, `inf`, `-inf` | `{"$float": "nan"}` / `"inf"` / `"-inf"` |
| dict with non-string keys, or keys starting with `$` | `{"$map": [[key, value], ..]}` |

`str`, `int`, finite `float`, `bool`, `null` and `list` are plain JSON. numpy scalars (`bool`, `int8`..`int64`, `uint8`..`uint64`, `float16/32/64`) become the matching Python numbers. Arrays and pandas `Timestamp` are not supported yet and are reported as `unsupported-value-type`. So is an `int` with more decimal digits than Python's integer-to-text limit (`sys.get_int_max_str_digits()`, 4,300 by default), reported as `builtins.int(digits>4300)`.

`docstore.jsonl` is UTF-8 without a byte order mark; lone surrogates in strings are written as `\uXXXX` escapes. Its size in bytes is bounded by `Limits.max_docstore_bytes` both when it is written and when it is read.
