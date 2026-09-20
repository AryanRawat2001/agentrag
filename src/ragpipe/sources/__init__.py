"""Document discovery from public indexes.

Each source module exposes `discover(client) -> list[SourceDoc]`, returning the
full candidate pool with normalised metadata. Sampling happens later, in
sample.py, so the discovery step stays deterministic and cacheable.
"""

from ragpipe.sources import ctgov, fda

__all__ = ["fda", "ctgov"]
