"""The served path must import without the heavy, optional dependencies.

Phase 7 split `sentence-transformers` (516 MB of torch), `qdrant-client` and the Bedrock
SDK out of the core dependencies, on the measured grounds that `ragpipe serve` uses none
of them: retrieval is BM25 over scipy sparse matrices and verification is pure Python.
That claim is what takes the container from ~1.2 GB to 569 MB.

It was verified once, by hand, in a scratch script — which is exactly the pattern this
phase's own progress entry names as a recurring defect: *a measurement that runs somewhere
the project cannot re-run it is a measurement the project does not have.* The next
module-level `import sentence_transformers` anywhere in the serve path would silently make
the container unbuildable-as-designed, and nothing would notice.

So it runs here. Each case is a subprocess, because import side effects cannot be undone
in-process: once `torch` is in `sys.modules`, a blocker installed afterwards proves nothing.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

#: Every dependency that is NOT in `[project.dependencies]`. If one of these becomes
#: reachable at import time from the serve path, the container's dependency set is wrong.
OPTIONAL_ONLY = (
    "torch",
    "transformers",
    "sentence_transformers",
    "qdrant_client",
    "anthropic",
    "boto3",
    "botocore",
)

#: Modules the serve path must be able to import with all of the above blocked.
SERVE_PATH = (
    "ragpipe",
    "ragpipe.citations",
    "ragpipe.retrieval",
    "ragpipe.answer",
    "ragpipe.generation",
    "ragpipe.service",
    "ragpipe.dashboard",
    "ragpipe.cli",
)

_BLOCKER = """
import sys
class Blocker:
    def __init__(self, names): self.names = set(names)
    def find_module(self, name, path=None): return self.find_spec(name, path)
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.names:
            raise ImportError(f"BLOCKED: {name}")
        return None
sys.meta_path.insert(0, Blocker(%r))
"""


def _run(body: str, blocked=OPTIONAL_ONLY) -> subprocess.CompletedProcess[str]:
    script = (_BLOCKER % (list(blocked),)) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )


class TestServePathIsLight:
    def test_every_serve_path_module_imports_with_the_extras_blocked(self):
        proc = _run(
            f"""
            for name in {SERVE_PATH!r}:
                __import__(name)
            print("OK")
            """
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "OK" in proc.stdout

    def test_the_blocker_actually_blocks(self):
        """Otherwise the test above passes because nothing was ever blocked -- the
        failure mode that has produced several no-op tests in this project."""
        proc = _run("import torch")
        assert proc.returncode != 0
        assert "BLOCKED: torch" in proc.stderr

    def test_the_app_serves_every_route_with_the_extras_blocked(self):
        """Importing is necessary but not sufficient: a lazy import inside a handler
        would pass the import test and 500 on the first request."""
        proc = _run(
            """
            from fastapi.testclient import TestClient
            from ragpipe import service
            from ragpipe.retrieval import BM25Retriever

            body = ("The sponsor shall submit the annual report within 60 days of the "
                    "anniversary date of the effective date of the application.")
            chunk = {
                "chunk_id": "c1",
                "doc_id": "d1",
                "text": body,
                "embed_text": "Reporting\\n\\n" + body,
                "section_heading": "Reporting",
                "identifiers": [],
            }
            state = service.AppState(chunks=[chunk], chunk_index={"c1": chunk})
            state.retriever = BM25Retriever([chunk])
            client = TestClient(service.create_app(state))
            codes = {
                "/": client.get("/").status_code,
                "/health": client.get("/health").status_code,
                "/stats": client.get("/stats").status_code,
                "/openapi.json": client.get("/openapi.json").status_code,
                "/query": client.post(
                    "/query", json={"query": "annual report", "k": 1, "retrieve_only": True}
                ).status_code,
            }
            assert all(v == 200 for v in codes.values()), codes
            print("ROUTES OK")
            """
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "ROUTES OK" in proc.stdout

    def test_bm25_actually_retrieves_with_the_extras_blocked(self):
        """scipy is a *core* dependency precisely because bm25s degrades without it
        rather than failing to import, which would quietly change the baseline every
        number in this project rests on."""
        proc = _run(
            """
            from ragpipe.retrieval import BM25Retriever
            # `prepend` mode indexes `embed_text`, so it has to be real text: a
            # placeholder tokenizes to an empty vocabulary and bm25s raises an opaque
            # `max() arg is an empty sequence` from inside its indexer.
            a = "The sponsor shall submit the annual report within 60 days."
            b = "Reports may be submitted through the electronic submissions gateway."
            chunks = [
                {"chunk_id": "c1", "doc_id": "d", "text": a, "embed_text": a,
                 "section_heading": "", "identifiers": []},
                {"chunk_id": "c2", "doc_id": "d", "text": b, "embed_text": b,
                 "section_heading": "", "identifiers": []},
            ]
            hits = BM25Retriever(chunks).search("annual report", k=2)
            assert hits and hits[0].chunk_id == "c1", hits
            print("BM25 OK")
            """
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "BM25 OK" in proc.stdout

    def test_the_cli_still_runs_with_the_extras_blocked(self):
        proc = _run(
            """
            import sys
            from ragpipe.cli import main
            for argv in (["--help"], ["serve", "--help"], ["ann", "--help"]):
                try:
                    main(argv)
                except SystemExit as exc:
                    assert exc.code == 0, (argv, exc.code)
            print("CLI OK")
            """
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "CLI OK" in proc.stdout


class TestOptionalExtrasFailClearly:
    def test_ann_names_the_extra_rather_than_raising_modulenotfound(self):
        """`ragpipe ann` needs `qdrant-client`. Without it, it previously died with a
        bare `ModuleNotFoundError` from inside `QdrantStore.from_dense` -- after loading
        chunks and building the dense matrix. Nothing named the extra to install."""
        proc = _run(
            """
            from ragpipe import vectorstore as vs
            try:
                vs.QdrantStore.from_dense(None, url="http://localhost:6333")
            except ImportError as exc:
                print("MESSAGE:", exc)
            """,
            blocked=("qdrant_client",),
        )
        combined = proc.stdout + proc.stderr
        assert "vectorstore" in combined and "extra" in combined.lower(), combined[-1500:]
