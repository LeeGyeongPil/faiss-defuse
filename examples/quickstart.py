"""Build a small store, save it without pickle, scan it, and load it back without any flag.

The embedding class is a fake, deterministic stand-in (practice data): search quality is meaningless.
Run:  uv run python examples/quickstart.py
"""
import hashlib
import tempfile
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from faiss_defuse import load_safe, save_safe, scan


class FakeEmbeddings(Embeddings):
    """Hash-based vectors so the example runs offline."""

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:16]]


emb = FakeEmbeddings()
store = FAISS.from_texts(["cats purr", "dogs bark", "birds sing"], emb, metadatas=[{"n": 1}, {"n": 2}, {"n": 3}])

with tempfile.TemporaryDirectory() as tmp:
    store.save_local(tmp + "/pickled")              # LangChain's own format (index.faiss + index.pkl)
    report = scan(tmp + "/pickled")                  # static scan, nothing is executed
    print("scan:", report.verdict, sorted(report.imports))

    save_safe(store, tmp + "/safe")                  # pickle-free format
    print("files:", sorted(p.name for p in Path(tmp, "safe").iterdir()))

    loaded = load_safe(tmp + "/safe", emb)           # no allow_dangerous_deserialization
    hit = loaded.similarity_search("dogs bark", k=1)[0]
    print("top hit:", hit.page_content, hit.metadata)
