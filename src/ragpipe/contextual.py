"""Contextual retrieval: an LLM-written situating blurb prepended to each chunk before embedding.

A chunk reading "The applicant shall submit within 15 days" is nearly unretrievable —
it names no drug, no regulation, no reporting regime. Contextual retrieval fixes that
by asking a model to read the *whole document* and write one or two sentences placing
each chunk inside it, then prepending that to the embedding input.

Phase 1b's `text` / `embed_text` split is what makes this safe: the blurb goes into
`embed_text` only, so `text` stays a byte-exact source slice and every span in the
golden set is unaffected. Same guarantee that let `embedding.py` normalize dot leaders.

## The cost architecture — where the interesting engineering is

The standard recipe (Anthropic's own cookbook included) is **one API call per chunk**,
with the document as a cached prefix. Measured by `estimate_cost` against the shipped
corpus — `structural` at the 1,024-char default, **13,423 chunks over 152 indexable
documents**, batched, Haiku 4.5:

    architecture      requests   all cache hits   all cache misses
    per-chunk           13,423          $39.02            $416.36
    grouped 40/call        417           $6.67             $15.38   <- this module
    one call per doc       152           $5.91              $5.91

Recomputed in Phase 8 against the corpus currently on disk. The previous table cited
13,706 chunks over 154 documents — the pre-quarantine corpus, before three documents were
excluded for broken text layers — which made every figure about 1.5% high. The ordering
and the argument are unchanged; the basis is now the corpus that exists.

Two independent reasons the grouped form wins, and only the first is about money:

1. **Batch parallelism undermines prompt caching.** A cache entry becomes readable
   only once the first response *begins*; a batch fires every request at once, so N
   concurrent requests sharing a prefix can all miss. The per-chunk architecture's
   10x saving is therefore a hope, not a guarantee — its real cost sits somewhere in a
   **$39–$416** band you do not control, and the spread *is* the risk. Grouping narrows
   the same band to $6.67–$15.38 and makes caching a bonus instead of a load-bearing
   assumption: 417 requests instead of 13,423.
2. **Bounded output per request.** One call per document would put 765 blurbs in a
   single response for the largest document — ~61K output tokens, and quality drifts
   across a generation that long. 40 per call caps it near 3,200 tokens. (An earlier
   version said "542 blurbs for the largest protocol": 542 was the 2,048-char figure,
   and the document is an FDA compliance-policy index, not a protocol.)

Grouping costs little against the all-in-one extreme ($6.67 vs $5.91 with hits) because
the cached document re-send is charged at 0.1x. Pick the group size for *quality*, and
let caching absorb the structural cost.

**The floor is not free either.** Prompt caching has a per-model minimum prefix — 4,096
tokens on Haiku 4.5, 512 on Opus 5 — and 33 of 152 documents are below the Haiku
minimum, so 33 groups never cache no matter how the work is arranged. Cost impact is
negligible (all 33 are single-group documents), but "grouping makes caching a bonus"
overstates what the mechanism delivers on the cheaper writer. `estimate_cost` reports
`cacheable_groups` so the claim is checkable rather than assumed.

## Per-model request shaping

The writer model is behind an interface for the same reason the embedder is — the
request shape is a property of the model, not of the caller:

  `haiku-4-5`  No `effort` parameter (it errors), no adaptive thinking.
  `opus-5`     Thinks by **default** — omitting `thinking` does not disable it. Run at
               `effort: "low"`; do *not* pass `thinking: {"type": "disabled"}`, which
               on this model can leak `<thinking>` tags into the visible response.
               Low effort is the cheaper and safer lever.

Both models are run as separate ablation rows, because "does the blurb-writer's
capability matter?" is a question a contextual-retrieval result is incomplete without —
if a 1x model matches a 5x one, that is the finding.

## Structured output, not prose parsing

Each response is constrained to a JSON object mapping chunk index to blurb via
`output_config.format`. Asking for prose and parsing it back would make the failure
mode "silently mismatched blurbs", which is unrecoverable and invisible: a blurb
attached to the wrong chunk degrades retrieval while every count still reconciles.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

# Writer models. `effort` and `thinking` are per-model request properties, so they
# live here rather than at the call site — the same reasoning as `embedding.MODELS`.
WRITERS: dict[str, dict[str, Any]] = {
    "haiku-4-5": {
        "model": "claude-haiku-4-5",
        # Haiku 4.5 rejects `output_config.effort`, and has no adaptive thinking.
        "effort": None,
        "price_in": 1.00,
        "price_out": 5.00,
        # Minimum cacheable prefix. **Not uniform across models, and not monotonic
        # across generations** — Haiku 4.5 needs 8x the prefix Opus 5 does. Below it
        # nothing caches, silently: `cache_creation_input_tokens` comes back 0 with no
        # error. Recorded per model because the caching design's value depends on it.
        "cache_min_tokens": 4096,
    },
    "opus-5": {
        "model": "claude-opus-5",
        # Thinking is ON by default on Opus 5 — omitting the parameter does not turn
        # it off. `effort: "low"` bounds the spend; disabling thinking outright is
        # the documented worse option (it can leak `<thinking>` tags into output).
        "effort": "low",
        "price_in": 5.00,
        "price_out": 25.00,
        "cache_min_tokens": 512,
    },
}

DEFAULT_WRITER = "haiku-4-5"

# Chunks per request. Chosen for bounded output quality, not cost: 40 blurbs is
# ~3,200 output tokens, short enough that the model does not drift, while the cached
# document re-send costs 0.1x. See the module docstring's cost table.
DEFAULT_GROUP_SIZE = 40

# Batch API discount and cache multipliers, for the cost estimator below.
BATCH_DISCOUNT = 0.50
CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute TTL; 1h would be 2.0
CACHE_READ_MULTIPLIER = 0.10

MAX_BLURB_CHARS = 400

SYSTEM_PROMPT = """\
You situate excerpts inside the regulatory document they came from, so that each \
excerpt can be found by search on its own.

You will be given one document, then a numbered list of verbatim excerpts from it. For \
each excerpt, write one or two sentences of context that a reader would need to \
understand what the excerpt is about without seeing the rest of the document.

State what the excerpt is about and where it sits in the document. Name the things a \
searcher would search for and the excerpt does not spell out: the document's subject, \
the regulation, drug, device, or trial it concerns, the section or requirement the \
excerpt belongs to, and who the obligation falls on. Prefer the document's own \
terminology over paraphrase, and resolve pronouns and bare references — "the \
Agency", "such reports", "this part" — to what they actually refer to.

Write context, not summary. Do not restate the excerpt's own wording back, do not \
quote it, and do not editorialise about its importance. If an excerpt is boilerplate \
or a table of contents fragment with no substantive content, say so plainly in one \
short sentence rather than inventing significance for it.\
"""

_DOC_HEADER = "<document title={title!r} source={source!r}>\n{text}\n</document>"


def _blurb_schema(n: int) -> dict[str, Any]:
    """A JSON schema requiring exactly one blurb per excerpt, keyed by index.

    Every key is `required` and `additionalProperties` is false, so a response that
    skips or invents an excerpt is rejected by the API rather than by us. Structured
    outputs need both to enforce the shape.
    """
    keys = [str(i) for i in range(1, n + 1)]
    return {
        "type": "object",
        "properties": {
            k: {"type": "string", "description": f"Context for excerpt {k}."} for k in keys
        },
        "required": keys,
        "additionalProperties": False,
    }


@dataclass(frozen=True, slots=True)
class ChunkRef:
    """One chunk awaiting a blurb."""

    chunk_id: str
    doc_id: str
    text: str


@dataclass
class ContextRequest:
    """One grouped API request: a document plus the excerpts to situate within it."""

    custom_id: str
    doc_id: str
    doc_title: str
    doc_source: str
    doc_text: str
    chunks: list[ChunkRef]

    def user_content(self) -> list[dict[str, Any]]:
        """Document block first (cacheable), excerpts after.

        Order is load-bearing. The document is the stable prefix shared by every group
        from this document, so it must precede the excerpts — putting the varying
        excerpt list first would make the prefix differ per request and cache nothing.
        """
        doc = _DOC_HEADER.format(title=self.doc_title, source=self.doc_source, text=self.doc_text)
        excerpts = "\n\n".join(
            f"<excerpt {i}>\n{c.text}\n</excerpt {i}>" for i, c in enumerate(self.chunks, start=1)
        )
        return [
            # The cache breakpoint sits on the document, not on the excerpts: the
            # excerpts change every request and would cache nothing but their own
            # single use, while paying the 1.25x write premium each time.
            {"type": "text", "text": doc, "cache_control": {"type": "ephemeral"}},
            {
                "type": "text",
                "text": (
                    f"Situate each of the following {len(self.chunks)} excerpts within "
                    f"the document above.\n\n{excerpts}"
                ),
            },
        ]


def group_requests(
    chunks: list[dict[str, Any]],
    docs: dict[str, dict[str, Any]],
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> list[ContextRequest]:
    """Group chunks by document, then into fixed-size batches within each document.

    Grouping by document is required, not an optimisation: the document *is* the
    context, so a request may only carry excerpts from one document. Chunk order
    within a document is preserved so a blurb can reference neighbouring structure.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")

    by_doc: dict[str, list[ChunkRef]] = {}
    for c in chunks:
        doc_id = c["doc_id"]
        if doc_id not in docs:
            continue  # no document text to situate against; caller reports these
        by_doc.setdefault(doc_id, []).append(
            ChunkRef(chunk_id=c["chunk_id"], doc_id=doc_id, text=c.get("text") or "")
        )

    requests: list[ContextRequest] = []
    for doc_id in sorted(by_doc):
        doc = docs[doc_id]
        refs = by_doc[doc_id]
        for start in range(0, len(refs), group_size):
            window = refs[start : start + group_size]
            requests.append(
                ContextRequest(
                    # Deterministic and decodable: results come back in arbitrary
                    # order, so the id has to identify the group on its own.
                    custom_id=f"{doc_id}::{start // group_size:04d}",
                    doc_id=doc_id,
                    doc_title=(doc.get("metadata") or {}).get("title") or doc_id,
                    doc_source=doc.get("source") or "unknown",
                    doc_text=doc.get("text") or "",
                    chunks=window,
                )
            )
    return requests


def build_params(req: ContextRequest, writer: str = DEFAULT_WRITER) -> dict[str, Any]:
    """Message-create params for one grouped request."""
    if writer not in WRITERS:
        raise ValueError(f"unknown writer {writer!r}; known: {sorted(WRITERS)}")
    spec = WRITERS[writer]

    n = len(req.chunks)
    params: dict[str, Any] = {
        "model": spec["model"],
        # Headroom for n blurbs plus, on a thinking model, its reasoning. Undersizing
        # this truncates the JSON object and loses the whole group.
        "max_tokens": max(2048, n * 220),
        # No `cache_control` here, deliberately. The system prompt is frozen across
        # every request, so caching it *looks* free — but at ~290 tokens it is far
        # below the minimum cacheable prefix (4,096 on Haiku 4.5, 512 on Opus 5), so a
        # breakpoint on it can never cache anything. It would have been a comment
        # asserting a mechanism that cannot fire. The system prompt still contributes
        # to the *document* block's prefix, which is where the breakpoint belongs.
        "system": [{"type": "text", "text": SYSTEM_PROMPT}],
        "messages": [{"role": "user", "content": req.user_content()}],
        "output_config": {"format": {"type": "json_schema", "schema": _blurb_schema(n)}},
    }
    if spec["effort"]:
        params["output_config"]["effort"] = spec["effort"]
    return params


_TAG = re.compile(r"</?thinking>", re.IGNORECASE)


def parse_blurbs(text: str, req: ContextRequest) -> dict[str, str]:
    """Map `chunk_id -> blurb` from one response.

    Raises on a response that does not cover the group. A silently partial parse is
    the dangerous failure here: blurbs would shift onto the wrong chunks and every
    count would still reconcile, so the damage would show up only as worse retrieval.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{req.custom_id}: response was not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{req.custom_id}: expected a JSON object, got {type(payload).__name__}")

    # A duplicate chunk_id inside one group would let the second blurb overwrite the
    # first, silently dropping one and landing the wrong text on the survivor — the
    # exact failure this function exists to prevent, arriving by a different route.
    # `chunking.build_chunks` mints unique ids, so this is a guard on the contract
    # rather than an observed bug.
    ids = [c.chunk_id for c in req.chunks]
    if len(set(ids)) != len(ids):
        dupes = sorted({cid for cid in ids if ids.count(cid) > 1})
        raise ValueError(f"{req.custom_id}: duplicate chunk_ids in group: {dupes}")

    out: dict[str, str] = {}
    for i, chunk in enumerate(req.chunks, start=1):
        raw = payload.get(str(i))
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{req.custom_id}: missing blurb for excerpt {i}")
        # Belt-and-braces: strip any leaked reasoning tags before the text is
        # embedded. `output_config.format` should prevent them, but a blurb is
        # cheap to clean and impossible to notice once it is inside a vector.
        cleaned = _TAG.sub("", raw).strip()[:MAX_BLURB_CHARS]
        out[chunk.chunk_id] = cleaned
    return out


def contextualize(chunk: dict[str, Any], blurb: str) -> str:
    """The embedding input for a contextualized chunk.

    Blurb, then the existing `embed_text` — which already carries the section heading
    under `structural`. `text` is never touched, so citation offsets are unchanged.
    """
    body = chunk.get("embed_text") or chunk.get("text") or ""
    return f"{blurb}\n\n{body}" if blurb else body


@dataclass
class CostReport:
    """Token accounting for a run, from the API's own usage numbers.

    Built from `usage` rather than estimated, because the estimate's weakest
    assumption — how many tokens a thinking model spends per blurb — is exactly the
    one worth measuring.
    """

    writer: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    blurbs: int = 0
    errors: list[str] = field(default_factory=list)

    def add_usage(self, usage: Any) -> None:
        self.requests += 1
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

    @property
    def effective_input_tokens(self) -> float:
        """Input tokens weighted by what each kind actually costs."""
        return (
            self.input_tokens
            + self.cache_write_tokens * CACHE_WRITE_MULTIPLIER
            + self.cache_read_tokens * CACHE_READ_MULTIPLIER
        )

    def cost_usd(self, *, batched: bool = True) -> float:
        spec = WRITERS[self.writer]
        cost = (
            self.effective_input_tokens / 1e6 * spec["price_in"]
            + self.output_tokens / 1e6 * spec["price_out"]
        )
        return cost * (BATCH_DISCOUNT if batched else 1.0)

    @property
    def cache_hit_rate(self) -> float:
        """Share of cacheable input served from cache.

        Reported because batch parallelism makes it genuinely uncertain — a low rate
        here is the measurement that justifies the grouped architecture.
        """
        total = self.cache_read_tokens + self.cache_write_tokens
        return self.cache_read_tokens / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "writer": self.writer,
            "model": WRITERS[self.writer]["model"],
            "requests": self.requests,
            "blurbs": self.blurbs,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "effective_input_tokens": round(self.effective_input_tokens, 1),
            "cost_usd_batched": round(self.cost_usd(batched=True), 4),
            "cost_usd_unbatched": round(self.cost_usd(batched=False), 4),
            "errors": self.errors,
        }


class ContextWriterClient(Protocol):
    """The slice of the Anthropic client this module uses.

    Narrow on purpose: it is what makes the whole pipeline testable without a key,
    the same role `FakeEmbedder` plays for Phase 3.
    """

    def submit(self, requests: list[tuple[str, dict[str, Any]]]) -> str: ...

    def results(self, batch_id: str) -> Any: ...


def estimate_cost(
    requests: list[ContextRequest],
    writer: str = DEFAULT_WRITER,
    *,
    chars_per_token: float = 4.0,
    out_tokens_per_blurb: int = 80,
    assume_cache_hits: bool = True,
) -> dict[str, Any]:
    """Pre-flight cost estimate, before spending anything.

    `assume_cache_hits=False` is a genuine **upper bound**: every group pays a cache
    *write* for the document. That is what an all-miss run actually costs, because
    `user_content()` sets `cache_control` unconditionally — a miss is still billed as
    `cache_creation` at 1.25x, never at 1.0x.

    An earlier version billed misses at 1.0x and was wrong in both directions. It
    understated the true worst case by 17% on this corpus, and — because a
    single-group document pays 1.25x once under "optimistic" and 1.0x once under
    "pessimistic" — the pessimistic figure came out *lower* than the optimistic one
    for 67 of 152 documents, i.e. it was not a bound at all. The plan says to quote
    this number when deciding whether to spend, so it has to actually bracket the
    outcome.

    Neither figure is the *floor*: prompt caching has a per-model minimum prefix
    (4,096 tokens on Haiku 4.5, 512 on Opus 5), and a document below it never caches
    no matter how many groups it has. See `cacheable_groups` in the returned dict.
    """
    spec = WRITERS[writer]
    minimum = spec["cache_min_tokens"]
    seen_docs: set[str] = set()
    eff_in = 0.0
    out = 0
    cacheable = 0
    system_tokens = len(SYSTEM_PROMPT) / chars_per_token

    for req in requests:
        doc_tokens = len(req.doc_text) / chars_per_token
        excerpt_tokens = sum(len(c.text) for c in req.chunks) / chars_per_token
        # The cached prefix is system + document, since the breakpoint sits on the
        # document block and the system prompt renders before it.
        prefix_cacheable = (system_tokens + doc_tokens) >= minimum
        if prefix_cacheable:
            cacheable += 1

        if assume_cache_hits and prefix_cacheable and req.doc_id in seen_docs:
            eff_in += doc_tokens * CACHE_READ_MULTIPLIER
        elif prefix_cacheable:
            eff_in += doc_tokens * CACHE_WRITE_MULTIPLIER
            seen_docs.add(req.doc_id)
        else:
            # Below the minimum nothing caches, and nothing is charged the write
            # premium either — plain input tokens at full price.
            eff_in += doc_tokens
            seen_docs.add(req.doc_id)
        eff_in += excerpt_tokens
        out += len(req.chunks) * out_tokens_per_blurb

    cost = eff_in / 1e6 * spec["price_in"] + out / 1e6 * spec["price_out"]
    return {
        "writer": writer,
        "requests": len(requests),
        "blurbs": sum(len(r.chunks) for r in requests),
        "effective_input_tokens": round(eff_in),
        "output_tokens": out,
        "assume_cache_hits": assume_cache_hits,
        "cacheable_groups": cacheable,
        "cache_min_tokens": minimum,
        "cost_usd": round(cost, 2),
        "cost_usd_batched": round(cost * BATCH_DISCOUNT, 2),
    }


# --- batch execution ---------------------------------------------------------


def submit_batch(client: Any, requests: list[ContextRequest], writer: str) -> str:
    """Submit one Message Batch and return its id.

    The Batch API is a 50% discount for work nobody is waiting on, which describes
    this exactly — blurbs are written once and then embedded. Batches accept up to
    100,000 requests, so this corpus (417 groups at the shipped 1,024-char default)
    fits in one submission.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    batch = client.messages.batches.create(
        requests=[
            Request(
                custom_id=req.custom_id,
                params=MessageCreateParamsNonStreaming(**build_params(req, writer)),
            )
            for req in requests
        ]
    )
    return batch.id


def collect_batch(
    client: Any,
    batch_id: str,
    requests: list[ContextRequest],
    writer: str,
) -> tuple[dict[str, str], CostReport]:
    """Read a finished batch into `chunk_id -> blurb`, plus real token accounting.

    Results arrive in **arbitrary order**, so they are keyed by `custom_id` and never
    by position — pairing by index here would silently attach every blurb to the wrong
    chunk while all the counts still reconciled.

    One failed group costs that group's blurbs, not the run: those chunks simply keep
    their uncontextualized `embed_text`, and the failure is recorded in the report.
    """
    by_id = {req.custom_id: req for req in requests}
    blurbs: dict[str, str] = {}
    report = CostReport(writer=writer)

    for result in client.messages.batches.results(batch_id):
        req = by_id.get(result.custom_id)
        if req is None:
            report.errors.append(f"{result.custom_id}: no matching request")
            continue

        kind = result.result.type
        if kind != "succeeded":
            # `errored` may be retryable (server-side) or not (invalid request); the
            # caller decides, so record which it was rather than collapsing them.
            detail = getattr(getattr(result.result, "error", None), "type", kind)
            report.errors.append(f"{req.custom_id}: {kind} ({detail})")
            continue

        message = result.result.message
        report.add_usage(message.usage)
        text = next((b.text for b in message.content if b.type == "text"), "")
        try:
            parsed = parse_blurbs(text, req)
        except ValueError as exc:
            report.errors.append(str(exc))
            continue
        blurbs.update(parsed)

    report.blurbs = len(blurbs)
    return blurbs, report


def apply_blurbs(
    chunks: list[dict[str, Any]], blurbs: dict[str, str]
) -> tuple[list[dict[str, Any]], int]:
    """Return chunks with blurbs folded into `embed_text`, and the count contextualized.

    `text`, `start`, and `end` are untouched, so every span in the golden set still
    resolves — the guarantee that makes contextual retrieval safe to add at all. A
    chunk with no blurb passes through unchanged rather than being dropped, so a
    partial run degrades coverage instead of corrupting the corpus.
    """
    out: list[dict[str, Any]] = []
    applied = 0
    for chunk in chunks:
        blurb = blurbs.get(chunk["chunk_id"], "")
        new = dict(chunk)
        if blurb:
            # Idempotent: re-applying to an already-contextualized chunk must not
            # double-prepend the blurb. A resume or re-run is the normal path for a
            # metered job that can fail partway, and a silently doubled blurb would
            # corrupt the embedding input while every count still reconciled — which
            # contradicts the "partial runs degrade coverage, never corrupt" guarantee.
            existing = chunk.get("context_blurb")
            base = (
                {**chunk, "embed_text": chunk["embed_text"][len(existing) + 2 :]}
                if existing and chunk.get("embed_text", "").startswith(existing)
                else chunk
            )
            new["embed_text"] = contextualize(base, blurb)
            new["context_blurb"] = blurb
            applied += 1
        out.append(new)
    return out, applied


# --- provider selection ------------------------------------------------------
#
# First-party Anthropic and Amazon Bedrock differ in three ways that matter here,
# so the provider is explicit rather than inferred from whichever credential
# happens to be in the environment:
#
#   1. **No Batch API on Bedrock.** The 50% discount is unavailable, so the same
#      work costs twice as much and runs synchronously. This is the whole reason
#      `run_sync` exists alongside the batch path.
#   2. **Model IDs carry an `anthropic.` prefix** on Bedrock. A bare id 400s there;
#      a prefixed id 400s on first-party.
#   3. **Automatic prompt caching is unavailable** on Bedrock — but explicit
#      `cache_control` breakpoints work, which is what this module already uses,
#      so the caching design carries over unchanged.

BEDROCK_PREFIX = "anthropic."


def model_id(writer: str, *, provider: str = "anthropic") -> str:
    """The model identifier for a writer on a given provider."""
    if writer not in WRITERS:
        raise ValueError(f"unknown writer {writer!r}; known: {sorted(WRITERS)}")
    base = WRITERS[writer]["model"]
    if provider == "bedrock":
        return f"{BEDROCK_PREFIX}{base}"
    if provider == "anthropic":
        return base
    raise ValueError(f"unknown provider {provider!r}; expected 'anthropic' or 'bedrock'")


def load_env_file(path: Any) -> dict[str, str]:
    """Read `KEY=value` lines from a dotenv-style file.

    Hand-rolled rather than pulling in `python-dotenv`: this needs to work when
    invoked from a heredoc, where `find_dotenv()` raises because there is no calling
    frame to walk. Values are not logged anywhere by this function.
    """
    import pathlib

    out: dict[str, str] = {}
    p = pathlib.Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip("'\"")
    return out


def make_client(provider: str = "anthropic", env: dict[str, str] | None = None) -> Any:
    """Construct the client for a provider, reading credentials from `env`.

    Bedrock needs a region and there is **no default** — `AnthropicBedrockMantle`
    raises at construction rather than sending a request, which is the failure you
    want (loud, before any spend).
    """
    import anthropic

    env = env or {}
    if provider == "bedrock":
        region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
        if not region:
            raise ValueError("bedrock requires AWS_REGION (no default is applied)")
        return anthropic.AnthropicBedrockMantle(
            aws_region=region,
            aws_access_key=env.get("AWS_ACCESS_KEY_ID"),
            aws_secret_key=env.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=env.get("AWS_SESSION_TOKEN"),
        )
    if provider == "anthropic":
        key = env.get("ANTHROPIC_API_KEY")
        # A bare constructor is correct when the key is absent: the SDK then resolves
        # an `ant auth login` profile, which is a supported path, not a failure.
        return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    raise ValueError(f"unknown provider {provider!r}")


def run_sync(
    client: Any,
    requests: list[ContextRequest],
    writer: str,
    *,
    provider: str = "anthropic",
    on_progress: Any = None,
) -> tuple[dict[str, str], CostReport]:
    """Run every group as an individual request, for providers without batching.

    Sequential on purpose. Prompt caching is a *prefix* match that only becomes
    readable once a response has begun, so firing these concurrently would make
    groups from the same document race each other and miss the cache the previous
    group just wrote — the same effect that makes the per-chunk architecture's saving
    unreliable in a batch. Going in order means every group after a document's first
    reads the cached document instead of re-paying for it.

    Requests are grouped by document already, and `group_requests` emits them in
    document order, so sequential execution keeps each document's groups adjacent.
    """
    blurbs: dict[str, str] = {}
    report = CostReport(writer=writer)

    for i, req in enumerate(requests, start=1):
        params = build_params(req, writer)
        params["model"] = model_id(writer, provider=provider)
        try:
            message = client.messages.create(**params)
        except Exception as exc:  # noqa: BLE001 - one bad group must not lose the run
            report.errors.append(f"{req.custom_id}: {type(exc).__name__}: {exc}")
            if on_progress:
                on_progress(i, len(requests), report)
            continue

        report.add_usage(message.usage)
        text = next((b.text for b in message.content if b.type == "text"), "")
        try:
            blurbs.update(parse_blurbs(text, req))
        except ValueError as exc:
            report.errors.append(str(exc))
        if on_progress:
            on_progress(i, len(requests), report)

    report.blurbs = len(blurbs)
    return blurbs, report
