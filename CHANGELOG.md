# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-22

### Added
- `scan`: static opcode scan of `index.pkl` (`pickletools.genops` plus a symbolic stack that follows memoised strings), allow-list verdicts `safe` / `unsafe` / `unsupported`, byte offsets for every finding, size and opcode limits.
- Inert decoder that never imports LangChain: allowed globals map to stubs or to small value shims (sets, dates, numpy scalars).
- `read_faiss_header`: validates the `index.faiss` header with the standard library before `faiss.read_index` is called.
- `inspect`, `convert`, `save_safe`: pickle-free `faiss-defuse/1` format (`index.faiss` + `docstore.jsonl` + `manifest.json` with SHA-256 hashes and FAISS options).
- `load_safe`: flag-free loader for pickle-free stores and for original `save_local` directories.
- `verify`: hashes, content and top-k equivalence against the original; optional `--trust-original` comparison.
- CLI `faiss-defuse scan|inspect|convert|verify|demo` with `--json`, exit codes, and English / Korean output (`--lang`, `FAISS_DEFUSE_LANG`, `LC_ALL` / `LC_MESSAGES` / `LANG`).
- `Limits.max_doc_nodes` (100,000 per document) and `Limits.max_total_nodes` (10,000,000 per store); `Limits.max_docstore_bytes` also bounds the JSON written by `convert` / `save_safe`.
- CLI `--limit NAME=NUMBER` (repeatable) on `scan`, `inspect`, `convert` and `verify` to change any `Limits` field; `limit-exceeded` errors say which flag raises the limit.

### Security
- Memo-reference amplification is bounded: a tiny pickle that reuses one object through many memo references fails fast with `limit-exceeded` instead of expanding to gigabytes. Allow-listed constructors that copy their argument (`set`, `frozenset`, `OrderedDict`, `_codecs.encode`) are capped at twice the pickle size (`reference-amplification`), and shared metadata is counted per document against `Limits.max_total_nodes`.
- An integer with more digits than Python can turn into text (4,300 by default) is reported as an unsupported value type (exit 1) instead of a raw `ValueError`.
