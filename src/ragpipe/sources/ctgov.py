"""Clinical trial protocols from ClinicalTrials.gov.

The v2 REST API exposes study records as JSON, and `aggFilters=docs:prot`
restricts results to studies that actually have a protocol PDF attached (most
studies do not). Attached documents live on a CDN under a path keyed by the last
two characters of the NCT ID:

    https://cdn.clinicaltrials.gov/large-docs/<nct[-2:]>/<NCT_ID>/<filename>

Why protocols are in this corpus at all: FDA guidance is well-structured
government prose, which makes structure-aware chunking look easy. Protocols are
the opposite — templated but wildly inconsistent between sponsors, with
schedule-of-assessment tables that are the hardest extraction case in the
corpus. That contrast is what makes the chunking-strategy comparison meaningful
instead of a formality.

`largeDocs[].size` is a byte count, not a page count. A large byte count against
few pages is the signature of a scanned document, so we keep it for pre-download
triage.
"""

from __future__ import annotations

from typing import Any

from ragpipe.models import SourceDoc
from ragpipe.net import PoliteClient

API_URL = "https://clinicaltrials.gov/api/v2/studies"
CDN_BASE = "https://cdn.clinicaltrials.gov/large-docs"

# Which attached document kinds we care about. Protocol and protocol+SAP bundles
# are the substantive documents; informed-consent forms are short and templated,
# and statistical analysis plans are useful but secondary.
WANTED_DOC_TYPES = {"Prot", "Prot_SAP"}

PAGE_SIZE = 200


def doc_url(nct_id: str, filename: str) -> str:
    return f"{CDN_BASE}/{nct_id[-2:]}/{nct_id}/{filename}"


def _get(d: dict[str, Any], *path: str, default: Any = None) -> Any:
    """Safe nested lookup — the v2 schema omits modules rather than nulling them."""
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def discover(client: PoliteClient, max_studies: int = 1200) -> list[SourceDoc]:
    """Page through studies that have protocol documents attached.

    `max_studies` bounds the candidate pool, not the final sample. A pool of a
    few hundred per sponsor class is ample for stratified sampling, and there is
    no reason to walk the entire registry to pick 40 documents.
    """
    docs: list[SourceDoc] = []
    seen_studies = 0
    page_token: str | None = None

    while seen_studies < max_studies:
        params: dict[str, Any] = {
            "aggFilters": "docs:prot",
            "pageSize": PAGE_SIZE,
            "format": "json",
        }
        if page_token:
            params["pageToken"] = page_token

        payload = client.get_json(API_URL, params=params)
        studies = payload.get("studies", [])
        if not studies:
            break

        for study in studies:
            seen_studies += 1
            ident = _get(study, "protocolSection", "identificationModule", default={})
            nct_id = ident.get("nctId")
            if not nct_id:
                continue

            sponsor = _get(
                study, "protocolSection", "sponsorCollaboratorsModule", "leadSponsor", default={}
            )
            design = _get(study, "protocolSection", "designModule", default={})
            conditions = _get(
                study, "protocolSection", "conditionsModule", "conditions", default=[]
            )
            status_module = _get(study, "protocolSection", "statusModule", default={})

            large_docs = (
                _get(study, "documentSection", "largeDocumentModule", "largeDocs", default=[]) or []
            )

            for entry in large_docs:
                kind = entry.get("typeAbbrev")
                filename = entry.get("filename")
                if kind not in WANTED_DOC_TYPES or not filename:
                    continue

                docs.append(
                    SourceDoc(
                        doc_id=f"ctgov-{nct_id}-{filename.removesuffix('.pdf')}",
                        source="ctgov_protocol",
                        title=ident.get("briefTitle") or nct_id,
                        url=doc_url(nct_id, filename),
                        landing_url=f"https://clinicaltrials.gov/study/{nct_id}",
                        doc_type=kind,
                        # No draft/final concept here; a protocol's revision date
                        # is the closest analogue and lands in issue_date.
                        status=None,
                        center=None,
                        offices=[],
                        docket=nct_id,  # the registry ID plays the exact-identifier role
                        issue_date=entry.get("date"),
                        product_areas=[],
                        topics=list(conditions)[:8],
                        extra={
                            "nct_id": nct_id,
                            "sponsor": sponsor.get("name"),
                            "sponsor_class": sponsor.get("class"),
                            "phases": design.get("phases") or [],
                            "enrollment": _get(design, "enrollmentInfo", "count"),
                            "overall_status": status_module.get("overallStatus"),
                            "declared_bytes": entry.get("size"),
                            "filename": filename,
                        },
                    )
                )

        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    return docs
