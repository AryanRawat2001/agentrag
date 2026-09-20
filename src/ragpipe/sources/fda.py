"""FDA guidance documents.

The entire guidance catalogue is published as one static JSON file backing the
"Search for FDA Guidance Documents" table — ~2,785 records, no pagination, no
scraping, no API key. Roughly 2,230 carry a PDF link; the rest are web-only or
withdrawn and get dropped.

Field semantics worth knowing, all verified against live data rather than
assumed:

  title                          HTML anchor -> display title + landing page path
  field_associated_media_2       HTML anchor -> /media/<id>/download (0 or 1 per record)
  field_final_guidance_1         "Draft" | "Final"          (the versioning slice)
  field_center                   single clean center name    (the ACL partition)
  field_issuing_office_taxonomy  <br>-delimited office list, sub-offices comma-delimited
  field_docket_number            HTML anchor -> regulations.gov, text is the docket ID
  field_issue_datetime           MM/DD/YYYY; 1900 appears as a placeholder, not a real date

The office taxonomy has two traps, both verified against live data:

  1. Strip the HTML *before* splitting and the values silently concatenate into
     strings like "Center for Drug Evaluation and ResearchCenter for Biologics
     Evaluation and Research". Split on the <br> tags while they are still there.
  2. Within a chunk, offices are separated by a comma with **no** following
     space, while individual office names contain commas *with* a space:

         Office of the Commissioner,Office of Policy, Legislation, and
         International Affairs,Office of Policy
         ^--- office 1 -----------^--- office 2 ----------------------^-- 3 --^

     Splitting on every comma shreds "Office of Policy, Legislation, and
     International Affairs" into three fake offices. The separator is `,(?!\\s)`.
"""

from __future__ import annotations

import html
import re
from datetime import date, datetime

from ragpipe.models import SourceDoc
from ragpipe.net import PoliteClient

INDEX_URL = "https://www.fda.gov/files/api/datatables/static/search-for-guidance.json"
FDA_BASE = "https://www.fda.gov"

_ANCHOR_RE = re.compile(r'<a\s[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.I | re.S)
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_MEDIA_ID_RE = re.compile(r"/media/(\d+)/download")

# Office separator: a comma NOT followed by whitespace. See the module docstring —
# office names legitimately contain ", " and must not be split on it.
_OFFICE_SEP_RE = re.compile(r",(?!\s)")

# Dates before this year are placeholders in the source data, not real issue
# dates (the index bottoms out at 1900). 1938 itself is accepted.
_MIN_PLAUSIBLE_YEAR = 1938  # year the FD&C Act was enacted; nothing predates it


def _strip_tags(value: str | None) -> str:
    """HTML -> plain text, preserving nothing but the words."""
    if not value:
        return ""
    return html.unescape(_TAG_RE.sub("", value)).strip()


def _first_anchor(value: str | None) -> tuple[str | None, str]:
    """Return (href, text) of the first anchor in an HTML fragment."""
    if not value:
        return None, ""
    m = _ANCHOR_RE.search(value)
    if not m:
        return None, _strip_tags(value)
    return m.group(1), _strip_tags(m.group(2))


def _split_offices(raw: str | None) -> list[str]:
    """Parse the <br>-delimited office taxonomy into a clean list.

    Split on <br> while the tags are still present, strip markup from each
    chunk, then split on `,(?!\\s)` — see the module docstring for why a plain
    comma split corrupts office names.
    """
    if not raw:
        return []
    offices: list[str] = []
    for chunk in _BR_RE.split(raw):
        for part in _OFFICE_SEP_RE.split(_strip_tags(chunk)):
            name = part.strip().rstrip(",")
            if name and name not in offices:
                offices.append(name)
    return offices


def _parse_date(raw: str | None) -> str | None:
    """MM/DD/YYYY -> ISO date string, or None if absent/implausible."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        parsed: date = datetime.strptime(raw, "%m/%d/%Y").date()
    except ValueError:
        return None
    if parsed.year < _MIN_PLAUSIBLE_YEAR or parsed.year > date.today().year + 1:
        return None
    return parsed.isoformat()


def _split_list(raw: str | None) -> list[str]:
    """Split a comma-delimited field (product areas, topics).

    Applies the same <br>-first rule as _split_offices. These two fields are
    comma-delimited in practice today, but the concatenation trap is a property
    of the source index, not of one field, so the rule is applied uniformly.
    """
    if not raw:
        return []
    out: list[str] = []
    for chunk in _BR_RE.split(raw):
        for part in _strip_tags(chunk).split(","):
            value = part.strip()
            if value and value not in out:
                out.append(value)
    return out


def _absolute(url: str | None) -> str | None:
    if not url:
        return None
    return url if url.startswith("http") else FDA_BASE + url


def discover(client: PoliteClient) -> list[SourceDoc]:
    """Fetch and normalise the full FDA guidance catalogue."""
    records = client.get_json(INDEX_URL)
    docs: list[SourceDoc] = []

    for rec in records:
        media_href, _ = _first_anchor(rec.get("field_associated_media_2"))
        if not media_href:
            continue  # web-only or withdrawn guidance — nothing to download

        media_id_match = _MEDIA_ID_RE.search(media_href)
        if not media_id_match:
            continue
        media_id = media_id_match.group(1)

        landing_href, title = _first_anchor(rec.get("title"))
        docket_href, docket_text = _first_anchor(rec.get("field_docket_number"))
        # "None found" is a real literal value in this field, not a docket number.
        docket = docket_text if docket_text and docket_text.lower() != "none found" else None

        status = _strip_tags(rec.get("field_final_guidance_1")) or None
        center = _strip_tags(rec.get("field_center")) or None
        issue_date = _parse_date(rec.get("field_issue_datetime"))

        docs.append(
            SourceDoc(
                doc_id=f"fda-{media_id}",
                source="fda_guidance",
                title=title or f"FDA guidance {media_id}",
                url=_absolute(media_href) or "",
                landing_url=_absolute(landing_href),
                doc_type=_strip_tags(rec.get("field_communication_type")) or None,
                status=status,
                center=center,
                offices=_split_offices(rec.get("field_issuing_office_taxonomy")),
                docket=docket,
                issue_date=issue_date,
                product_areas=_split_list(rec.get("field_regulated_product_field")),
                topics=_split_list(rec.get("field_topics")),
                extra={
                    "media_id": media_id,
                    "docket_url": docket_href,
                    "raw_issue_date": (rec.get("field_issue_datetime") or "").strip() or None,
                    "date_invalid": issue_date is None,
                    "open_comment": _strip_tags(rec.get("open-comment")) or None,
                },
            )
        )

    return docs


def normalise_title(title: str) -> str:
    """Aggressive title normalisation, used to pair drafts with their finals.

    Drops punctuation, edition markers, and the draft/final wording that
    otherwise prevents the same guidance from matching across revisions.
    """
    t = title.lower()
    t = re.sub(r"\b(draft|final|revision\s*\d*|rev\.?\s*\d*|guidance for industry)\b", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()
