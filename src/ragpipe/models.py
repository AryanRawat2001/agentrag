"""Manifest record types and JSONL persistence.

Two record types, written by two separate stages so each is independently
re-runnable and inspectable:

  SourceDoc  -- the *plan*: what we intend to fetch and the metadata the source
                index gave us. Committed to corpus/manifest.jsonl.
  FetchedDoc -- the *result*: download outcome, sha256, and text-layer verdict.
                Written to data/fetched.jsonl (gitignored, derived).

The two are joined on `doc_id` by the stats stage.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self


@dataclass(slots=True)
class SourceDoc:
    """One document we intend to download, plus everything the index told us.

    Every metadata field here exists because it unlocks a specific downstream
    capability, not for completeness' sake:
      status        -> draft-vs-final eval slice ("which version is authoritative?")
      center        -> document-level ACL / permission-filtered retrieval
      docket        -> exact-identifier queries, where BM25 beats dense retrieval
      issue_date    -> temporal reasoning and superseded-document handling
      sponsor_class -> structural heterogeneity for the chunking comparison
    """

    doc_id: str
    source: str  # "fda_guidance" | "ctgov_protocol"
    title: str
    url: str  # PDF to download
    landing_url: str | None = None  # human-readable source page, for citations

    # Content pin. None at discovery time (the index does not publish hashes);
    # populated by the first successful fetch and committed with the manifest.
    # Thereafter every fetch verifies against it, so a changed remote document is
    # a loud failure rather than silent corpus drift. This is what makes the
    # corpus reproducible from a committed file rather than merely re-downloadable.
    sha256: str | None = None

    doc_type: str | None = None  # FDA communication type, or CT.gov doc kind (Prot/SAP/ICF)
    status: str | None = None  # "Draft" | "Final" (FDA only)
    center: str | None = None  # clean single-value FDA center
    offices: list[str] = field(default_factory=list)  # parsed multi-value office taxonomy
    docket: str | None = None
    issue_date: str | None = None  # ISO date, or None when absent/invalid
    product_areas: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)

    # Source-specific extras (nct_id, sponsor, sponsor_class, phase, conditions,
    # declared_bytes, sample_reason, ...). Kept loose so adding a third source
    # later doesn't require a schema migration.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(slots=True)
class FetchedDoc:
    """Download + PDF-inspection outcome for one SourceDoc."""

    doc_id: str
    ok: bool
    path: str | None = None
    sha256: str | None = None
    bytes: int | None = None
    error: str | None = None

    # PDF inspection (see pdfcheck.py)
    pages: int | None = None
    text_chars: int | None = None
    image_only_pages: int | None = None
    text_layer: str | None = None  # digital_native | mixed | image_only | unreadable
    table_like_lines: int | None = None
    docket_like_tokens: int | None = None  # FDA docket format, e.g. FDA-2016-D-0734
    registry_like_tokens: int | None = None  # ClinicalTrials.gov NCT IDs
    unreadable_reason: str | None = None  # why a PDF failed to open, when it did

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def write_jsonl(path: Path, records: list[SourceDoc] | list[FetchedDoc]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sort by doc_id so the committed manifest has a stable diff between runs.
    ordered = sorted(records, key=lambda r: r.doc_id)
    with path.open("w", encoding="utf-8") as fh:
        for rec in ordered:
            fh.write(rec.to_json() + "\n")


def write_jsonl_dicts(path: Path, rows: list[dict[str, Any]], sort_key: str = "doc_id") -> None:
    """Write plain dicts as JSONL, sorted for a stable diff between runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: str(r.get(sort_key, "")))
    with path.open("w", encoding="utf-8") as fh:
        for row in ordered:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_manifest(path: Path) -> list[SourceDoc]:
    return [SourceDoc.from_dict(d) for d in read_jsonl(path)]


def load_fetched(path: Path) -> dict[str, FetchedDoc]:
    return {d["doc_id"]: FetchedDoc.from_dict(d) for d in read_jsonl(path)}
