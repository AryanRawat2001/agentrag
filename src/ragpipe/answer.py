"""Grounded answering: prompt, response envelope, refusal, and verification.

This is the layer that turns retrieved chunks into an answer whose every claim is
checkable. It owns three things the transport layer deliberately does not: what the
model is asked, what shape it must reply in, and what happens when the context does
not support an answer.

## Show the model exactly the string verification searches

Chunks are rendered from `chunk["text"]`, never `chunk["embed_text"]`. This looks
like a detail and is not. `embed_text` has the section heading prepended for 94.8%
of `structural` chunks (see `retrieval.py`), so it is not a contiguous slice of the
source document. A model shown `embed_text` can faithfully quote a heading that
appears at a different offset -- or twice -- in `text`, and `citations.locate_quote`
would then report a span that does not correspond to what the model was looking at.
Verification searches `text`, so the prompt shows `text`. The two must not diverge.

## Two independent refusal mechanisms, measured apart

The unanswerable slice exists to test refusal instead of fabrication, and there are
two ways to get it, with different costs and failure modes:

1. **Score-gated refusal**, before any generation. Phase 3 measured refusal
   separability -- how cleanly top retrieval scores separate answerable from
   unanswerable queries -- as **best-threshold accuracy** over 60 answerable and 60
   unanswerable queries, majority baseline 0.5. On `structural`, the chunking this
   phase retrieves over, BM25 alone scores **0.983**; 1.000 is `semantic` and 0.992
   is `fixed`. Min-max fusion recovers much less: **0.667** on `structural` (0.725 is
   its best anywhere, on `semantic`). And that is a property of min-max, not of fusion
   in general -- rank fusion plus reranking reaches **0.942** on `structural`.

   So a threshold on the top score can refuse without spending a token, and the
   mechanism is **retriever-specific** -- it must be re-validated against whatever
   Phase 7 serves rather than inherited from the best row of an earlier table. The
   gap is real but narrower than the sparse-versus-min-max pairing suggests.

   Three corrections worth carrying, and the third is the instructive one. Phase 5's
   audit gate caught "1.000 for BM25 alone" (true only of `semantic`) and "0.725 for any
   fusion configuration" (true only of min-max, understating the real fusion maximum by
   0.217). Phase 6's audit gate then caught that the *replacement* still quoted 0.725
   beside "on `structural`" -- 0.725 is `semantic`; structural's best min-max is 0.667.
   **The first fix corrected the scope of the claim and left its population wrong**,
   which is the same defect one level down. Finally, Phase 5's own separability figure
   is pairwise
   **AUC**, a different statistic from Phase 3's accuracy -- the two are not
   comparable, and this docstring formerly implied they were.
2. **Model-decided refusal**, via the `refused` field in the response envelope. This
   catches the case the threshold cannot: chunks that score well lexically but do
   not actually answer the question.

`GroundedAnswer.refusal_source` records which fired. Collapsing them into one
boolean would make a strong threshold look like a well-behaved model, and would hide
a regression in either one behind the other.

## Why quotes, and why the prompt is insistent about them

Tier-1 verification locates the model's quote in the source (deviation #14). That
makes verbatim quoting the single behaviour the prompt most needs to elicit -- a
paraphrase is correctly unverifiable, and the resulting failure rate is a real
measurement of the generator, not a bug to be prompted away. The instructions ask
for exact spans and warn that paraphrase fails; what they must not do is ask the
model to *report offsets*, which it cannot do reliably and which we no longer need.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ragpipe import citations as cit
from ragpipe.generation import DEFAULT_GENERATOR, GenerationResult, Generator, make_generator

#: Response envelope. Constrained server-side by `responseSchema`, so a truncated or
#: invented shape is a 400 rather than a silent misparse here. `propertyOrdering` is
#: a Gemini schema field: it fixes key order in the emitted JSON. Note what that does
#: and does not buy — `citations` is emitted *after* `answer`, so quotes are still
#: produced after the prose they support. The only ordering guarantee here is that
#: `reasoning` precedes `answer`, which is the weaker claim rule 7 makes. An earlier
#: version of this comment claimed the ordering stopped the model front-loading prose
#: before selecting evidence; it does not.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "refused": {"type": "boolean"},
        "reasoning": {"type": "string"},
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["chunk_id", "quote"],
                "propertyOrdering": ["chunk_id", "quote"],
            },
        },
    },
    "required": ["refused", "reasoning", "answer", "citations"],
    "propertyOrdering": ["refused", "reasoning", "answer", "citations"],
}

SYSTEM_PROMPT = """\
You answer questions about regulatory and clinical-trial documents using only the \
excerpts provided. You are evaluated on whether every claim you make can be traced \
to an exact span of a provided excerpt.

Rules:

1. Use only the provided excerpts. Do not use outside knowledge, even if you are \
confident it is correct.
2. Every substantive claim in your answer must be supported by a citation.
3. A citation's "quote" must be copied CHARACTER FOR CHARACTER from the excerpt it \
cites. Do not paraphrase, summarise, correct spelling, expand abbreviations, or \
tidy punctuation inside a quote. An automated check locates each quote in the source \
text; a paraphrased quote fails that check and counts against you. You may normalise \
runs of whitespace and line breaks, and nothing else.
4. Quote at least {min_quote} characters -- roughly a full clause. A quote too short \
to be distinctive is not evidence and is scored as a defect.
5. "chunk_id" must be copied exactly from the [chunk_id: ...] label of the excerpt \
the quote came from. Do not cite one excerpt for text that appears in another.
6. If the excerpts do not contain the answer, set "refused" to true, leave "answer" \
as a one-sentence statement that the provided documents do not address the question, \
and return an empty "citations" list. Refusing when the answer is absent is correct \
behaviour and is scored as such. Guessing is not.
7. Put your evidence selection in "reasoning" before writing "answer" -- name which \
excerpts bear on the question and which do not.
"""

_EXCERPT = "[chunk_id: {chunk_id}] (document {doc_id}{page})\n{text}"


def render_context(chunks: Sequence[Mapping[str, Any]]) -> str:
    """Render retrieved chunks as labelled excerpts.

    `text` is used verbatim, including its PDF whitespace artifacts. Cleaning it
    here would mean the model quotes a string that does not exist in the source that
    verification searches -- the normalized matching tier in `citations.py` exists
    precisely so the artifacts can be left alone.
    """
    blocks: list[str] = []
    for chunk in chunks:
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        if page_start is None:
            page = ""
        elif page_end is None or page_end == page_start:
            page = f", page {page_start}"
        else:
            page = f", pages {page_start}-{page_end}"
        blocks.append(
            _EXCERPT.format(
                chunk_id=chunk.get("chunk_id", ""),
                doc_id=chunk.get("doc_id", ""),
                page=page,
                text=chunk.get("text") or "",
            )
        )
    return "\n\n".join(blocks)


def build_prompt(query: str, chunks: Sequence[Mapping[str, Any]]) -> str:
    """The user turn: excerpts first, question last.

    Question last because it is the instruction the model should be holding when it
    starts generating, and because it keeps the long, reusable part of the prompt at
    the front -- the same ordering `contextual.py` uses to make its document prefix
    cacheable.
    """
    return (
        f"Excerpts:\n\n{render_context(chunks)}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the excerpts above, following every rule."
    )


def system_prompt(min_quote_chars: int = cit.MIN_QUOTE_CHARS) -> str:
    """The system turn, with the quote-length floor kept in sync with the verifier.

    Interpolated rather than written twice: a prompt that asks for 20 characters
    while the verifier requires 24 would manufacture `too_short` failures and read
    like a model defect.
    """
    return SYSTEM_PROMPT.format(min_quote=min_quote_chars)


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """An answer plus everything needed to score it."""

    query: str
    answer: str
    refused: bool
    citations: tuple[dict[str, str], ...]
    verification: cit.VerificationReport
    refusal_source: str | None = None  # "score_gate" | "model" | None
    reasoning: str = ""
    usage: GenerationResult | None = field(default=None, repr=False)
    top_score: float | None = None

    @property
    def citation_precision(self) -> float | None:
        return self.verification.citation_precision

    @property
    def fully_grounded(self) -> bool:
        return self.verification.fully_grounded

    @property
    def is_answered(self) -> bool:
        return not self.refused


def _score_gate_refusal(
    query: str, top_score: float | None, threshold: float | None
) -> GroundedAnswer | None:
    if threshold is None or top_score is None or top_score >= threshold:
        return None
    return GroundedAnswer(
        query=query,
        answer="The provided documents do not address this question.",
        refused=True,
        citations=(),
        verification=cit.VerificationReport(checks=()),
        refusal_source="score_gate",
        reasoning=(
            f"Top retrieval score {top_score:.4f} below refusal threshold {threshold:.4f}; "
            "refused before generation."
        ),
        top_score=top_score,
    )


def answer_query(
    query: str,
    chunks: Sequence[Mapping[str, Any]],
    *,
    generator: Generator | None = None,
    generator_spec: str = DEFAULT_GENERATOR,
    top_score: float | None = None,
    refusal_threshold: float | None = None,
    min_quote_chars: int = cit.MIN_QUOTE_CHARS,
    max_output_tokens: int | None = 4096,
    temperature: float | None = 0.0,
) -> GroundedAnswer:
    """Answer `query` from `chunks`, then verify every citation it produced.

    The score gate is checked first, so an unanswerable query that retrieval already
    identified costs nothing. `refusal_threshold=None` disables it, which is the
    right default until the threshold has been validated against the retriever that
    actually ships -- see the module docstring.
    """
    gated = _score_gate_refusal(query, top_score, refusal_threshold)
    if gated is not None:
        return gated

    gen = generator if generator is not None else make_generator(generator_spec)
    result = gen.generate(
        build_prompt(query, chunks),
        system=system_prompt(min_quote_chars),
        schema=ANSWER_SCHEMA,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    payload = result.parse_json()
    if not isinstance(payload, dict):
        from ragpipe.generation import GenerationError

        raise GenerationError(f"expected a JSON object, got {type(payload).__name__}")

    refused = bool(payload.get("refused"))
    raw_citations: Iterable[Mapping[str, Any]] = payload.get("citations") or []
    claimed = tuple(
        {"chunk_id": str(c.get("chunk_id", "")), "quote": str(c.get("quote", ""))}
        for c in raw_citations
        if isinstance(c, Mapping)
    )

    # Citations are verified even on a refusal. A model that refuses *and* cites is
    # inconsistent, and silently dropping the citations would hide that.
    verification = cit.verify_answer(claimed, chunks, min_quote_chars=min_quote_chars)

    return GroundedAnswer(
        query=query,
        answer=str(payload.get("answer") or ""),
        refused=refused,
        citations=claimed,
        verification=verification,
        refusal_source="model" if refused else None,
        reasoning=str(payload.get("reasoning") or ""),
        usage=result,
        top_score=top_score,
    )
