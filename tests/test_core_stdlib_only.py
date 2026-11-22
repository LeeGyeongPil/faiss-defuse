from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")

SCRIPT = textwrap.dedent(
    """
    import sys
    BLOCKED = {"faiss", "numpy", "langchain", "langchain_core", "langchain_community", "pydantic"}

    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError("blocked: " + name)

    sys.meta_path.insert(0, Block())
    sys.path.insert(0, sys.argv[1])
    import faiss_defuse

    store, out = sys.argv[2], sys.argv[3]
    assert faiss_defuse.scan(store).verdict == "safe"
    info = faiss_defuse.inspect(store)
    assert info.ok and info.doc_count == 20, info
    manifest = faiss_defuse.convert(store, out)
    assert manifest.doc_count == 20
    assert faiss_defuse.inspect(out).ok
    assert faiss_defuse.verify(out, original=store).verified
    leaked = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
    assert not leaked, leaked
    print("OK")
    """
)


def test_core_without_optional_imports(clean_store: Path, tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", SCRIPT, SRC, str(clean_store), str(tmp_path / "converted")],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
