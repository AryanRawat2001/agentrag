"""Contextual retrieval tests.

No API key, no network, no spend. The properties worth defending here are the ones
whose failure is *invisible*: a blurb attached to the wrong chunk degrades retrieval
while every count still reconciles, and a cache breakpoint on the wrong block turns a
$4 run into a $268 one with no error either way.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from ragpipe import contextual as ctx


def chunk(cid: str, doc_id: str, text: str, *, heading: str | None = None) -> dict:
    return {
        "chunk_id": cid,
        "doc_id": doc_id,
        "text": text,
        "embed_text": f"{heading}\n\n{text}" if heading else text,
        "section_heading": heading,
        "strategy": "structural",
    }


def doc(doc_id: str, text: str = "Document body about adverse event reporting.") -> dict:
    return {
        "doc_id": doc_id,
        "source": "fda",
        "text": text,
        "metadata": {"title": f"Guidance {doc_id}"},
    }


# -- grouping ---------------------------------------------------------------


def test_groups_are_per_document():
    """A request may only carry excerpts from one document — the document *is* the
    context, so mixing two would situate excerpts against the wrong text."""
    chunks = [chunk(f"a{i}", "doc-a", f"t{i}") for i in range(3)]
    chunks += [chunk(f"b{i}", "doc-b", f"t{i}") for i in range(3)]
    reqs = ctx.group_requests(chunks, {"doc-a": doc("doc-a"), "doc-b": doc("doc-b")})
    assert len(reqs) == 2
    for r in reqs:
        assert {c.doc_id for c in r.chunks} == {r.doc_id}


def test_group_size_caps_chunks_per_request():
    chunks = [chunk(f"c{i}", "d1", f"t{i}") for i in range(95)]
    reqs = ctx.group_requests(chunks, {"d1": doc("d1")}, group_size=40)
    assert [len(r.chunks) for r in reqs] == [40, 40, 15]


def test_every_chunk_appears_exactly_once():
    """The invariant that keeps blurb counts honest: no chunk dropped, none duplicated."""
    chunks = [chunk(f"c{i}", f"d{i % 4}", f"t{i}") for i in range(87)]
    docs = {f"d{i}": doc(f"d{i}") for i in range(4)}
    reqs = ctx.group_requests(chunks, docs, group_size=7)
    seen = [c.chunk_id for r in reqs for c in r.chunks]
    assert sorted(seen) == sorted(c["chunk_id"] for c in chunks)
    assert len(seen) == len(set(seen))


def test_chunk_order_within_a_document_is_preserved():
    """Blurbs may reference neighbouring structure, so order must not be shuffled."""
    chunks = [chunk(f"c{i:02d}", "d1", f"t{i}") for i in range(10)]
    reqs = ctx.group_requests(chunks, {"d1": doc("d1")}, group_size=4)
    assert [c.chunk_id for r in reqs for c in r.chunks] == [c["chunk_id"] for c in chunks]


def test_chunks_without_document_text_are_skipped_not_faked():
    """Better to report a missing document than to situate a chunk against nothing."""
    chunks = [chunk("c1", "known", "t"), chunk("c2", "missing", "t")]
    reqs = ctx.group_requests(chunks, {"known": doc("known")})
    assert [c.chunk_id for r in reqs for c in r.chunks] == ["c1"]


def test_custom_ids_are_unique_and_deterministic():
    """Batch results come back in arbitrary order, so the id must identify the group
    on its own — and be stable across runs so a resume matches."""
    chunks = [chunk(f"c{i}", f"d{i % 3}", "t") for i in range(50)]
    docs = {f"d{i}": doc(f"d{i}") for i in range(3)}
    a = ctx.group_requests(chunks, docs, group_size=6)
    b = ctx.group_requests(chunks, docs, group_size=6)
    ids = [r.custom_id for r in a]
    assert len(ids) == len(set(ids))
    assert ids == [r.custom_id for r in b]


def test_rejects_bad_group_size():
    with pytest.raises(ValueError, match="group_size"):
        ctx.group_requests([], {}, group_size=0)


def test_empty_input_yields_no_requests():
    assert ctx.group_requests([], {}) == []


# -- request shape: the cache breakpoint ------------------------------------


def test_document_precedes_excerpts_and_carries_the_cache_breakpoint():
    """The cost architecture in one assertion.

    The document is the prefix shared across every group from that document, so it
    must come first and hold the breakpoint. Reversing the order, or marking the
    excerpt block instead, caches nothing while still paying the 1.25x write premium —
    silently, with no error and no wrong output.
    """
    req = ctx.group_requests([chunk("c1", "d1", "t")], {"d1": doc("d1")})[0]
    blocks = req.user_content()
    assert len(blocks) == 2
    assert "Document body" in blocks[0]["text"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[1], "excerpt block must not be cached"
    assert "<excerpt 1>" in blocks[1]["text"]


def test_document_prefix_is_byte_identical_across_groups_of_one_document():
    """Prompt caching is a prefix match — any byte difference invalidates it."""
    chunks = [chunk(f"c{i}", "d1", f"body {i}") for i in range(9)]
    reqs = ctx.group_requests(chunks, {"d1": doc("d1")}, group_size=3)
    prefixes = {r.user_content()[0]["text"] for r in reqs}
    assert len(prefixes) == 1, "document prefix differs between groups; caching would miss"


def test_system_prompt_is_frozen_and_carries_no_unreachable_breakpoint():
    """Frozen across requests, but deliberately *not* marked cacheable.

    An earlier version put a `cache_control` breakpoint here with the comment "caches
    once for the whole run". At ~290 tokens the system prompt is far below the minimum
    cacheable prefix (4,096 on Haiku 4.5, 512 on Opus 5), so the breakpoint could never
    fire — a comment asserting a mechanism that cannot work.
    """
    a = ctx.build_params(ctx.group_requests([chunk("c1", "d1", "t")], {"d1": doc("d1")})[0])
    b = ctx.build_params(ctx.group_requests([chunk("c2", "d2", "t")], {"d2": doc("d2")})[0])
    assert a["system"] == b["system"], "system prompt must not vary per request"
    assert "cache_control" not in a["system"][0]
    smallest = min(w["cache_min_tokens"] for w in ctx.WRITERS.values())
    assert len(ctx.SYSTEM_PROMPT) / 4 < smallest, (
        "system prompt now exceeds the smallest cache minimum; a breakpoint here "
        "would be reachable and this test's premise needs revisiting"
    )


def test_excerpts_are_numbered_from_one_to_match_the_schema():
    chunks = [chunk(f"c{i}", "d1", f"body {i}") for i in range(4)]
    req = ctx.group_requests(chunks, {"d1": doc("d1")})[0]
    text = req.user_content()[1]["text"]
    for i in range(1, 5):
        assert f"<excerpt {i}>" in text
    assert "<excerpt 0>" not in text and "<excerpt 5>" not in text


# -- per-model request shaping ---------------------------------------------


def test_haiku_omits_effort_and_opus_sets_it_low():
    """Haiku 4.5 rejects `effort`; Opus 5 thinks by default, so low effort is the
    bound — and is the documented-safer choice over disabling thinking."""
    req = ctx.group_requests([chunk("c1", "d1", "t")], {"d1": doc("d1")})[0]
    haiku = ctx.build_params(req, "haiku-4-5")
    opus = ctx.build_params(req, "opus-5")
    assert "effort" not in haiku["output_config"]
    assert opus["output_config"]["effort"] == "low"
    assert haiku["model"] == "claude-haiku-4-5"
    assert opus["model"] == "claude-opus-5"


def test_no_writer_disables_thinking():
    """Disabling thinking on Opus 5 can leak `<thinking>` tags into the response."""
    req = ctx.group_requests([chunk("c1", "d1", "t")], {"d1": doc("d1")})[0]
    for writer in ctx.WRITERS:
        assert "thinking" not in ctx.build_params(req, writer)


def test_max_tokens_scales_with_group_size():
    """Undersizing truncates the JSON object and loses the whole group, not one blurb."""
    small = ctx.group_requests([chunk("c0", "d1", "t")], {"d1": doc("d1")})[0]
    big = ctx.group_requests([chunk(f"c{i}", "d1", "t") for i in range(40)], {"d1": doc("d1")})[0]
    assert ctx.build_params(big)["max_tokens"] > ctx.build_params(small)["max_tokens"]
    assert ctx.build_params(big)["max_tokens"] >= 40 * 200


def test_rejects_unknown_writer():
    req = ctx.group_requests([chunk("c1", "d1", "t")], {"d1": doc("d1")})[0]
    with pytest.raises(ValueError, match="unknown writer"):
        ctx.build_params(req, "gpt-9")


def test_schema_requires_every_excerpt_and_forbids_extras():
    """The API rejects a short or invented response, so we never have to detect it."""
    schema = ctx._blurb_schema(3)
    assert schema["required"] == ["1", "2", "3"]
    assert schema["additionalProperties"] is False


# -- parsing: the invisible-failure guard ----------------------------------


def test_parses_blurbs_onto_the_right_chunks():
    chunks = [chunk("alpha", "d1", "a"), chunk("beta", "d1", "b")]
    req = ctx.group_requests(chunks, {"d1": doc("d1")})[0]
    out = ctx.parse_blurbs(json.dumps({"1": "About alpha.", "2": "About beta."}), req)
    assert out == {"alpha": "About alpha.", "beta": "About beta."}


def test_missing_blurb_raises_rather_than_shifting_the_rest():
    """The failure this guards is silent: without it, blurb 2 would land on chunk 3
    and every count would still reconcile while retrieval quietly got worse."""
    chunks = [chunk(f"c{i}", "d1", "t") for i in range(3)]
    req = ctx.group_requests(chunks, {"d1": doc("d1")})[0]
    with pytest.raises(ValueError, match="missing blurb for excerpt 2"):
        ctx.parse_blurbs(json.dumps({"1": "one", "3": "three"}), req)


@pytest.mark.parametrize("bad", ["", "   ", None, 42, [], {}])
def test_empty_or_non_string_blurb_is_rejected(bad):
    req = ctx.group_requests([chunk("c1", "d1", "t")], {"d1": doc("d1")})[0]
    with pytest.raises(ValueError, match="missing blurb"):
        ctx.parse_blurbs(json.dumps({"1": bad}), req)


def test_non_json_and_non_object_responses_raise():
    req = ctx.group_requests([chunk("c1", "d1", "t")], {"d1": doc("d1")})[0]
    with pytest.raises(ValueError, match="not JSON"):
        ctx.parse_blurbs("Here are the blurbs: ...", req)
    with pytest.raises(ValueError, match="expected a JSON object"):
        ctx.parse_blurbs(json.dumps(["one"]), req)


def test_leaked_thinking_tags_are_stripped():
    """Impossible to notice once a blurb is inside a vector, cheap to clean here."""
    req = ctx.group_requests([chunk("c1", "d1", "t")], {"d1": doc("d1")})[0]
    out = ctx.parse_blurbs(json.dumps({"1": "<thinking>hm</thinking>Real context."}), req)
    assert out["c1"] == "hmReal context."
    assert "<thinking>" not in out["c1"]


def test_overlong_blurb_is_truncated():
    req = ctx.group_requests([chunk("c1", "d1", "t")], {"d1": doc("d1")})[0]
    out = ctx.parse_blurbs(json.dumps({"1": "x" * 5000}), req)
    assert len(out["c1"]) == ctx.MAX_BLURB_CHARS


# -- contextualize: the citation-safety guarantee --------------------------


def test_blurb_prepends_to_embed_text_and_never_touches_text():
    """The Phase 1b split is what makes this safe. If `text` moved, every span in the
    golden set would silently point at the wrong characters."""
    c = chunk("c1", "d1", "The applicant shall submit.", heading="Subpart C")
    before = c["text"]
    embed = ctx.contextualize(c, "This concerns adverse event reporting under 21 CFR 314.")
    assert c["text"] == before
    assert embed.startswith("This concerns adverse event reporting")
    assert "Subpart C" in embed
    assert "The applicant shall submit." in embed


def test_empty_blurb_leaves_embed_text_unchanged():
    c = chunk("c1", "d1", "body", heading="H")
    assert ctx.contextualize(c, "") == c["embed_text"]


def test_contextualize_falls_back_to_text_when_no_embed_text():
    assert ctx.contextualize({"text": "body"}, "ctx") == "ctx\n\nbody"


# -- cost accounting ------------------------------------------------------


class FakeUsage:
    def __init__(self, i=0, o=0, cw=0, cr=0):
        self.input_tokens = i
        self.output_tokens = o
        self.cache_creation_input_tokens = cw
        self.cache_read_input_tokens = cr


def test_effective_input_weights_cache_reads_and_writes():
    r = ctx.CostReport(writer="haiku-4-5")
    r.add_usage(FakeUsage(i=1000, o=500, cw=10_000, cr=100_000))
    # 1000 + 10_000*1.25 + 100_000*0.10 = 23_500
    assert r.effective_input_tokens == pytest.approx(23_500)


def test_batch_discount_halves_cost():
    r = ctx.CostReport(writer="haiku-4-5")
    r.add_usage(FakeUsage(i=1_000_000, o=1_000_000))
    assert r.cost_usd(batched=False) == pytest.approx(1.0 + 5.0)
    assert r.cost_usd(batched=True) == pytest.approx((1.0 + 5.0) / 2)


def test_opus_costs_five_times_haiku_on_identical_usage():
    usage = dict(i=1_000_000, o=1_000_000)
    h = ctx.CostReport(writer="haiku-4-5")
    h.add_usage(FakeUsage(**usage))
    o = ctx.CostReport(writer="opus-5")
    o.add_usage(FakeUsage(**usage))
    assert o.cost_usd() == pytest.approx(h.cost_usd() * 5)


def test_cache_hit_rate_is_reported_and_safe_when_empty():
    """Reported because batch parallelism makes it genuinely uncertain — a low rate
    is the measurement that justifies grouping over per-chunk calls."""
    r = ctx.CostReport(writer="haiku-4-5")
    assert r.cache_hit_rate == 0.0
    r.add_usage(FakeUsage(cw=1000, cr=9000))
    assert r.cache_hit_rate == pytest.approx(0.9)


def test_report_serializes_for_the_artifact():
    r = ctx.CostReport(writer="opus-5")
    r.add_usage(FakeUsage(i=10, o=20, cw=30, cr=40))
    r.blurbs = 40
    d = r.as_dict()
    assert d["model"] == "claude-opus-5"
    assert d["requests"] == 1 and d["blurbs"] == 40
    assert d["cost_usd_batched"] < d["cost_usd_unbatched"]


# -- pre-flight estimate --------------------------------------------------


def test_pessimistic_estimate_exceeds_optimistic_one():
    """Quote the range, not the optimistic end: batch parallelism can defeat caching,
    so the no-cache-hits number is the one that bounds the decision."""
    chunks = [chunk(f"c{i}", "d1", "x" * 2000) for i in range(120)]
    reqs = ctx.group_requests(chunks, {"d1": doc("d1", "y" * 400_000)}, group_size=40)
    optimistic = ctx.estimate_cost(reqs, assume_cache_hits=True)
    pessimistic = ctx.estimate_cost(reqs, assume_cache_hits=False)
    assert pessimistic["cost_usd"] > optimistic["cost_usd"]
    assert optimistic["blurbs"] == pessimistic["blurbs"] == 120


def test_estimate_counts_every_blurb():
    chunks = [chunk(f"c{i}", f"d{i % 3}", "t") for i in range(31)]
    docs = {f"d{i}": doc(f"d{i}") for i in range(3)}
    reqs = ctx.group_requests(chunks, docs, group_size=5)
    assert ctx.estimate_cost(reqs)["blurbs"] == 31


def test_grouping_is_cheaper_than_one_call_per_chunk():
    """The module's central claim, asserted rather than asserted-in-prose."""
    chunks = [chunk(f"c{i}", "d1", "x" * 2000) for i in range(200)]
    docs = {"d1": doc("d1", "y" * 400_000)}
    per_chunk = ctx.estimate_cost(ctx.group_requests(chunks, docs, group_size=1))
    grouped = ctx.estimate_cost(ctx.group_requests(chunks, docs, group_size=40))
    assert grouped["cost_usd"] < per_chunk["cost_usd"]
    assert grouped["requests"] == 5 and per_chunk["requests"] == 200


# -- batch collection: the arbitrary-ordering hazard -----------------------


class FakeResultInner:
    def __init__(self, kind, message=None, error_type=None):
        self.type = kind
        self.message = message
        self.error = type("E", (), {"type": error_type})() if error_type else None


class FakeResult:
    def __init__(self, custom_id, kind, message=None, error_type=None):
        self.custom_id = custom_id
        self.result = FakeResultInner(kind, message, error_type)


class FakeMessage:
    def __init__(self, text, usage=None):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.usage = usage or FakeUsage(i=100, o=200, cw=1000, cr=5000)


class FakeBatches:
    def __init__(self, results):
        self._results = results
        self.submitted = None

    def create(self, requests):
        self.submitted = requests
        return type("B", (), {"id": "batch_fake"})()

    def results(self, batch_id):
        return iter(self._results)


class FakeClient:
    def __init__(self, results):
        self.messages = type("M", (), {"batches": FakeBatches(results)})()


def _reqs(n_docs=2, per_doc=3):
    chunks, docs = [], {}
    for d in range(n_docs):
        docs[f"d{d}"] = doc(f"d{d}")
        chunks += [chunk(f"d{d}-c{i}", f"d{d}", f"body {i}") for i in range(per_doc)]
    return ctx.group_requests(chunks, docs, group_size=per_doc), chunks


def test_results_are_keyed_by_custom_id_not_position():
    """Batch results arrive in arbitrary order. Pairing by index would attach every
    blurb to the wrong chunk while every count still reconciled — invisible."""
    reqs, _ = _reqs(n_docs=2, per_doc=2)
    payload = json.dumps({"1": "first", "2": "second"})
    # Deliberately reversed relative to `reqs`.
    results = [
        FakeResult(reqs[1].custom_id, "succeeded", FakeMessage(payload)),
        FakeResult(reqs[0].custom_id, "succeeded", FakeMessage(payload)),
    ]
    blurbs, report = ctx.collect_batch(FakeClient(results), "b", reqs, "haiku-4-5")
    assert blurbs[reqs[0].chunks[0].chunk_id] == "first"
    assert blurbs[reqs[1].chunks[0].chunk_id] == "first"
    assert report.errors == []
    assert report.blurbs == 4


def test_one_failed_group_does_not_lose_the_run():
    reqs, _ = _reqs(n_docs=2, per_doc=2)
    payload = json.dumps({"1": "a", "2": "b"})
    results = [
        FakeResult(reqs[0].custom_id, "succeeded", FakeMessage(payload)),
        FakeResult(reqs[1].custom_id, "errored", error_type="overloaded_error"),
    ]
    blurbs, report = ctx.collect_batch(FakeClient(results), "b", reqs, "haiku-4-5")
    assert len(blurbs) == 2
    assert len(report.errors) == 1 and "overloaded_error" in report.errors[0]


def test_unparseable_response_is_recorded_not_raised():
    reqs, _ = _reqs(n_docs=1, per_doc=2)
    results = [FakeResult(reqs[0].custom_id, "succeeded", FakeMessage("not json"))]
    blurbs, report = ctx.collect_batch(FakeClient(results), "b", reqs, "haiku-4-5")
    assert blurbs == {}
    assert len(report.errors) == 1


def test_unknown_custom_id_is_reported():
    reqs, _ = _reqs(n_docs=1, per_doc=2)
    results = [FakeResult("ghost::0000", "succeeded", FakeMessage(json.dumps({"1": "x"})))]
    _, report = ctx.collect_batch(FakeClient(results), "b", reqs, "haiku-4-5")
    assert "no matching request" in report.errors[0]


def test_usage_is_accumulated_from_the_api_not_estimated():
    reqs, _ = _reqs(n_docs=1, per_doc=2)
    msg = FakeMessage(json.dumps({"1": "a", "2": "b"}), FakeUsage(i=7, o=11, cw=13, cr=17))
    _, report = ctx.collect_batch(
        FakeClient([FakeResult(reqs[0].custom_id, "succeeded", msg)]), "b", reqs, "haiku-4-5"
    )
    assert (report.input_tokens, report.output_tokens) == (7, 11)
    assert (report.cache_write_tokens, report.cache_read_tokens) == (13, 17)


#: The two tests below reach code that builds a real Anthropic request type, so they
#: need the `bedrock` extra. Skipped rather than failed on a core-only install: that
#: install is the one the container uses, and a skip says "not exercised here" where a
#: failure would say "broken", which is a different and wrong claim.
requires_anthropic = pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is None,
    reason="needs the `bedrock` extra (uv sync --extra bedrock)",
)


@requires_anthropic
def test_submit_sends_one_request_per_group():
    reqs, _ = _reqs(n_docs=3, per_doc=4)
    client = FakeClient([])
    assert ctx.submit_batch(client, reqs, "haiku-4-5") == "batch_fake"
    submitted = client.messages.batches.submitted
    assert len(submitted) == len(reqs)
    assert [r["custom_id"] for r in submitted] == [r.custom_id for r in reqs]


# -- apply_blurbs: the citation guarantee at corpus scale ------------------


def test_apply_blurbs_never_mutates_text_or_offsets():
    chunks = [chunk("c1", "d1", "body one"), chunk("c2", "d1", "body two")]
    for c in chunks:
        c["start"], c["end"] = 10, 18
    out, applied = ctx.apply_blurbs(chunks, {"c1": "Context for one."})
    assert applied == 1
    assert out[0]["embed_text"].startswith("Context for one.")
    assert out[0]["text"] == "body one"
    assert (out[0]["start"], out[0]["end"]) == (10, 18)
    assert out[0]["context_blurb"] == "Context for one."


def test_chunks_without_a_blurb_pass_through_unchanged():
    """A partial run must degrade coverage, not corrupt or drop chunks."""
    chunks = [chunk("c1", "d1", "a"), chunk("c2", "d1", "b")]
    out, applied = ctx.apply_blurbs(chunks, {})
    assert applied == 0
    assert [c["embed_text"] for c in out] == [c["embed_text"] for c in chunks]
    assert all("context_blurb" not in c for c in out)


def test_apply_blurbs_does_not_mutate_its_input():
    chunks = [chunk("c1", "d1", "body")]
    before = chunks[0]["embed_text"]
    ctx.apply_blurbs(chunks, {"c1": "ctx"})
    assert chunks[0]["embed_text"] == before


# -- provider selection ----------------------------------------------------


def test_bedrock_prefixes_model_ids_and_anthropic_does_not():
    """A bare id 400s on Bedrock; a prefixed id 400s on first-party. Getting this
    wrong fails every request, which is at least loud — unlike the cache mistakes."""
    assert ctx.model_id("haiku-4-5", provider="anthropic") == "claude-haiku-4-5"
    assert ctx.model_id("haiku-4-5", provider="bedrock") == "anthropic.claude-haiku-4-5"
    assert ctx.model_id("opus-5", provider="bedrock") == "anthropic.claude-opus-5"


def test_unknown_provider_and_writer_are_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        ctx.model_id("haiku-4-5", provider="vertex")
    with pytest.raises(ValueError, match="unknown writer"):
        ctx.model_id("nope", provider="bedrock")


@requires_anthropic
def test_bedrock_requires_an_explicit_region():
    """`AnthropicBedrockMantle` has no default region, so this must fail at
    construction — before any spend — rather than on the first request."""
    with pytest.raises(ValueError, match="AWS_REGION"):
        ctx.make_client("bedrock", env={"AWS_ACCESS_KEY_ID": "x"})


def test_env_file_parsing_handles_comments_quotes_and_blanks(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "AWS_REGION=us-east-1",
                "AWS_SECRET_ACCESS_KEY='quoted/secret+value'",
                'AWS_SESSION_TOKEN="double=quoted=="',
                "MALFORMED_LINE_NO_EQUALS",
            ]
        )
    )
    env = ctx.load_env_file(p)
    assert env["AWS_REGION"] == "us-east-1"
    assert env["AWS_SECRET_ACCESS_KEY"] == "quoted/secret+value"
    # A session token contains '=' padding; partition on the first '=' only.
    assert env["AWS_SESSION_TOKEN"] == "double=quoted=="
    assert "MALFORMED_LINE_NO_EQUALS" not in env


def test_missing_env_file_is_empty_not_an_error(tmp_path):
    assert ctx.load_env_file(tmp_path / "absent") == {}


# -- synchronous runner (the Bedrock path) ---------------------------------


class SyncClient:
    """Records the params of every call and replays scripted outcomes."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **params):
        self.calls.append(params)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_sync_run_uses_the_prefixed_model_id():
    reqs, _ = _reqs(n_docs=1, per_doc=2)
    msg = FakeMessage(json.dumps({"1": "a", "2": "b"}))
    client = SyncClient([msg])
    ctx.run_sync(client, reqs, "haiku-4-5", provider="bedrock")
    assert client.calls[0]["model"] == "anthropic.claude-haiku-4-5"


def test_sync_run_issues_one_request_per_group_in_order():
    """Order matters: sequential execution is what lets each document's later groups
    read the cache its first group wrote."""
    reqs, _ = _reqs(n_docs=2, per_doc=2)
    payload = json.dumps({"1": "a", "2": "b"})
    client = SyncClient([FakeMessage(payload), FakeMessage(payload)])
    blurbs, report = ctx.run_sync(client, reqs, "haiku-4-5", provider="bedrock")
    assert len(client.calls) == 2
    assert report.requests == 2
    assert len(blurbs) == 4


def test_sync_run_survives_a_failing_request():
    reqs, _ = _reqs(n_docs=2, per_doc=2)
    payload = json.dumps({"1": "a", "2": "b"})
    client = SyncClient([RuntimeError("throttled"), FakeMessage(payload)])
    blurbs, report = ctx.run_sync(client, reqs, "haiku-4-5", provider="bedrock")
    assert len(blurbs) == 2
    assert len(report.errors) == 1 and "throttled" in report.errors[0]
    assert report.requests == 1  # only the successful call contributes usage


def test_sync_run_reports_progress_including_failures():
    """A long synchronous run needs visible progress, and a failure must still tick —
    otherwise a stalled run and a failing run look identical."""
    reqs, _ = _reqs(n_docs=2, per_doc=2)
    seen: list[tuple[int, int]] = []
    client = SyncClient([RuntimeError("x"), FakeMessage(json.dumps({"1": "a", "2": "b"}))])
    ctx.run_sync(
        client,
        reqs,
        "haiku-4-5",
        provider="bedrock",
        on_progress=lambda i, n, rep: seen.append((i, n)),
    )
    assert seen == [(1, 2), (2, 2)]


# -- regressions from the Phase 4 code-review gate -------------------------


def _corpus(n_docs: int, groups_per_doc: int, doc_chars: int = 400_000):
    """n_docs documents, each large enough to split into `groups_per_doc` groups."""
    chunks, docs = [], {}
    for d in range(n_docs):
        docs[f"d{d}"] = doc(f"d{d}", "y" * doc_chars)
        chunks += [
            chunk(f"d{d}-c{i}", f"d{d}", "x" * 2000)
            for i in range(groups_per_doc * ctx.DEFAULT_GROUP_SIZE)
        ]
    return ctx.group_requests(chunks, docs, group_size=ctx.DEFAULT_GROUP_SIZE)


def test_no_cache_hits_is_a_genuine_upper_bound():
    """The gate's most severe finding.

    `assume_cache_hits=False` is quoted when deciding whether to spend, so it has to
    bracket the outcome. It billed misses at 1.0x while `user_content()` sets
    `cache_control` unconditionally — so a miss is really billed as `cache_creation` at
    1.25x. For a single-group document that made the "pessimistic" figure *lower* than
    the optimistic one: not a bound at all, on 67 of 154 real documents.
    """
    for groups in (1, 2, 5):
        reqs = _corpus(1, groups)
        hi = ctx.estimate_cost(reqs, assume_cache_hits=False)["cost_usd"]
        lo = ctx.estimate_cost(reqs, assume_cache_hits=True)["cost_usd"]
        assert hi >= lo, f"{groups} group(s): all-miss {hi} < all-hit {lo}"


def test_single_group_document_is_the_exact_case_that_inverted():
    reqs = _corpus(1, 1)
    assert len(reqs) == 1
    hi = ctx.estimate_cost(reqs, assume_cache_hits=False)["cost_usd"]
    lo = ctx.estimate_cost(reqs, assume_cache_hits=True)["cost_usd"]
    # With one group there is nothing to read from cache, so the two must agree.
    assert hi == lo


def test_cache_minimum_is_per_model_and_reported():
    """Haiku 4.5 needs 8x the prefix Opus 5 does, so 'the document caches' is a
    per-writer claim. `cacheable_groups` makes it checkable."""
    assert ctx.WRITERS["haiku-4-5"]["cache_min_tokens"] == 4096
    assert ctx.WRITERS["opus-5"]["cache_min_tokens"] == 512
    small = _corpus(1, 2, doc_chars=4_000)  # ~1,000 tokens: cacheable on Opus, not Haiku
    assert ctx.estimate_cost(small, "haiku-4-5")["cacheable_groups"] == 0
    assert ctx.estimate_cost(small, "opus-5")["cacheable_groups"] == 2


def test_below_minimum_documents_pay_no_write_premium():
    """Nothing caches below the minimum, and nothing is charged for trying."""
    tiny = _corpus(1, 1, doc_chars=4_000)
    haiku = ctx.estimate_cost(tiny, "haiku-4-5")
    doc_tokens = 4_000 / 4
    # Input is the document at 1.0x plus the excerpts, with no 1.25x premium.
    assert haiku["effective_input_tokens"] < doc_tokens * ctx.CACHE_WRITE_MULTIPLIER + 40 * 500


def test_apply_blurbs_is_idempotent():
    """A metered job that fails partway gets resumed, and a doubled blurb would
    corrupt the embedding input while every count still reconciled."""
    chunks = [chunk("c1", "d1", "The applicant shall submit.", heading="Subpart C")]
    once, _ = ctx.apply_blurbs(chunks, {"c1": "About adverse event reporting."})
    twice, _ = ctx.apply_blurbs(once, {"c1": "About adverse event reporting."})
    assert once[0]["embed_text"] == twice[0]["embed_text"]
    assert twice[0]["embed_text"].count("About adverse event reporting.") == 1


def test_reapplying_a_different_blurb_replaces_rather_than_stacks():
    chunks = [chunk("c1", "d1", "body text here")]
    first, _ = ctx.apply_blurbs(chunks, {"c1": "First context."})
    second, _ = ctx.apply_blurbs(first, {"c1": "Corrected context."})
    assert second[0]["embed_text"].startswith("Corrected context.")
    assert "First context." not in second[0]["embed_text"]
    assert second[0]["text"] == "body text here"


def test_duplicate_chunk_ids_in_a_group_raise():
    """Overwriting on a duplicate id would drop one blurb and land the wrong text on
    the survivor — the same silent misassignment `parse_blurbs` exists to prevent."""
    chunks = [chunk("dup", "d1", "a"), chunk("dup", "d1", "b")]
    req = ctx.group_requests(chunks, {"d1": doc("d1")})[0]
    import json as _json

    with pytest.raises(ValueError, match="duplicate chunk_ids"):
        ctx.parse_blurbs(_json.dumps({"1": "one", "2": "two"}), req)
