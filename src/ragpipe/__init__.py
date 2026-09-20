"""Hybrid-search RAG over public regulatory documentation.

Phase 0 (this module set) is corpus acquisition only: discover documents from
public indexes, sample them deterministically, download them with integrity
checks, and characterise what we actually got. No chunking, embedding, or
retrieval lives here yet.
"""

__version__ = "0.1.0"

# Repo-relative paths. Committed artifacts live in corpus/ and reports/;
# everything derived and bulky lives in data/, which is gitignored.
import os
from pathlib import Path

#: Repo root. Derived from the source layout, overridable with `RAGPIPE_ROOT`.
#
# The override exists for the container. `parents[2]` walks out of `src/ragpipe/` to the
# checkout, which is correct when running from the tree and wrong once the package is
# installed into site-packages -- there it resolves to a directory inside the virtualenv,
# so every path below would point at nothing and the service would report zero chunks
# rather than failing to start.
ROOT = Path(os.environ.get("RAGPIPE_ROOT") or Path(__file__).resolve().parents[2])
CORPUS_DIR = ROOT / "corpus"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = ROOT / "reports"

MANIFEST_PATH = CORPUS_DIR / "manifest.jsonl"
FETCHED_PATH = DATA_DIR / "fetched.jsonl"

EXTRACTED_DIR = DATA_DIR / "extracted"
EXTRACT_INDEX_PATH = EXTRACTED_DIR / "index.jsonl"

CHUNKS_DIR = DATA_DIR / "chunks"

# Embedding caches, keyed on chunk content hash. Gitignored and reproducible: the
# corpus can be re-embedded from chunks, so these are a time saving, not an input.
VECTORS_DIR = DATA_DIR / "vectors"

EVALSET_PATH = CORPUS_DIR / "evalset.jsonl"
GOLDEN_PATH = CORPUS_DIR / "golden.jsonl"

# Human curation verdicts on flagged golden pairs. Committed, unlike the derived
# artifacts: these are judgement calls that cost a person's attention to make, and
# regenerating them is not possible -- only redoing them.
CURATION_VERDICTS_PATH = CORPUS_DIR / "curation_verdicts.json"
