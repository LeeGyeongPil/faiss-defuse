"""Fixtures.  Every store and every hostile pickle is generated while the tests run."""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402


def make_docs(n: int = 20) -> list[Document]:
    return [
        Document(
            page_content=f"doc {i} about topic {i % 3}",
            metadata={"src": f"f{i}.txt", "page": i, "score": 0.5 * i, "tags": ["a", "b"]},
            id=f"id-{i}",
        )
        for i in range(n)
    ]


@pytest.fixture(scope="session")
def emb() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture(scope="session")
def clean_store(tmp_path_factory: pytest.TempPathFactory, emb: DeterministicFakeEmbedding) -> Path:
    """A real ``FAISS.save_local`` store with JSON-friendly metadata.  Read-only for tests."""
    path = tmp_path_factory.mktemp("clean") / "store"
    FAISS.from_documents(make_docs(), emb).save_local(str(path))
    return path


@pytest.fixture()
def work_store(clean_store: Path, tmp_path: Path) -> Path:
    """A private copy of the clean store that a test may damage."""
    dst = tmp_path / "work_store"
    shutil.copytree(clean_store, dst)
    return dst


@pytest.fixture()
def sentinel(tmp_path: Path) -> Path:
    """Path of the file a harmless payload would create if a pickle were executed."""
    return tmp_path / "PWNED"


@pytest.fixture(autouse=True)
def _english_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not depend on the developer's locale."""
    for name in ("FAISS_DEFUSE_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(name, raising=False)
