"""Tests for the FastAPI serving path.

Weighted toward the two things a latency-and-cost endpoint gets wrong silently.

First, **a stage that never ran must not report a number.** An empty percentile that
comes back `0.0` reads as "generation takes no time", which is the most flattering
possible lie about a pipeline whose generation stage is seconds of somebody else's
network. Every latency assertion here checks `None` and `n == 0`, not a value.

Second, **cost must stay null until a real rate is supplied.** Phase 4's audit found four
invented per-token literals in this project's own reports, and this module is where a
fifth would go. `cost_usd` is asserted null by default, and asserted to bill thinking
tokens as output when rates *are* set -- thinking is invisible in the response and was
the single largest lever on Phase 5 spend.

The generator is always a fake. No test here touches the network or spends quota, and
`retrieve_only` is asserted to leave the fake uncalled rather than merely to return
early -- "did not generate" and "generated and discarded" are indistinguishable in the
response body but differ by a daily quota unit.
"""

from __future__ import annotations

import json
import re

import pytest

# A plain import, not `importorskip`. fastapi is a **core** dependency -- the serving
# path is the phase's deliverable -- so if it is missing, this module's 40-odd tests
# vanishing while the suite reports green is the wrong outcome. Optional-extra modules
# (`qdrant_client`, `anthropic`) are the ones that get skip guards.
from fastapi.testclient import TestClient

from ragpipe import service
from ragpipe.generation import GenerationError, GenerationResult
from ragpipe.retrieval import Hit

SOURCE = (
    "The sponsor shall submit the annual report within 60 days of the anniversary date "
    "of the effective date of the application."
)


def _chunk(chunk_id="c1", doc_id="d1", text=SOURCE, heading="Reporting"):
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "embed_text": f"{heading}\n\n{text}",
        "section_heading": heading,
        "identifiers": [],
    }


class FakeRetriever:
    """Returns a fixed ranking. Records queries so route behaviour is observable."""

    name = "fake"

    def __init__(self, chunk_ids=("c1",)):
        self._chunk_ids = list(chunk_ids)
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int = 10) -> list[Hit]:
        self.calls.append((query, k))
        return [
            Hit(chunk_id=cid, score=10.0 - i, rank=i + 1)
            for i, cid in enumerate(self._chunk_ids[:k])
        ]


class FakeGenerator:
    """A generator whose response is scripted, and which counts its own invocations."""

    name = "fake-model"

    def __init__(self, payload=None, error: Exception | None = None, tokens=(100, 50, 7)):
        self.payload = payload if payload is not None else _answer_payload()
        self.error = error
        self.tokens = tokens
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        prompt_tokens, output_tokens, thinking_tokens = self.tokens
        return GenerationResult(
            text=json.dumps(self.payload),
            model=self.name,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            latency_s=1.5,
        )


def _answer_payload(quote=None, chunk_id="c1", refused=False, answer="Within 60 days."):
    return {
        "answer": answer,
        "refused": refused,
        "reasoning": "",
        "citations": [] if refused else [{"chunk_id": chunk_id, "quote": quote or SOURCE[:60]}],
    }


def _state(chunks=None, retriever=None, generator=None, **kw):
    chunks = list(chunks) if chunks is not None else [_chunk()]
    state = service.AppState(
        chunks=chunks,
        chunk_index={c["chunk_id"]: c for c in chunks},
        retriever=retriever if retriever is not None else FakeRetriever(),
        generator=generator,
        **kw,
    )
    return state


def _client(state):
    return TestClient(service.create_app(state))


class TestTimings:
    def test_a_stage_that_never_ran_reports_none_not_zero(self):
        """The whole point of the table. 0.0 ms would read as "instant"."""
        t = service.Timings()
        t.record("retrieve", 1.0)
        pct = t.percentiles()
        assert pct["retrieve"]["n"] == 1
        assert pct["generate"] == {"n": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}

    def test_every_stage_key_is_present_even_when_unused(self):
        pct = service.Timings().percentiles()
        assert set(pct) == set(service.STAGES)

    def test_p50_is_interpolated_and_p95_is_nearest_rank(self):
        """Two conventions, used knowingly. Mixing them silently is the defect."""
        t = service.Timings()
        for v in (1.0, 2.0, 3.0, 4.0):
            t.record("total", v)
        pct = t.percentiles()
        assert pct["total"]["p50_ms"] == 2.5  # interpolated, not a member of the sample
        assert pct["total"]["p95_ms"] == 4.0  # nearest rank
        assert pct["total"]["max_ms"] == 4.0

    def test_samples_are_bounded(self):
        t = service.Timings()
        for i in range(service.MAX_SAMPLES + 250):
            t.record("retrieve", float(i))
        assert t.percentiles()["retrieve"]["n"] == service.MAX_SAMPLES

    def test_bound_drops_oldest_so_percentiles_track_recent_latency(self):
        t = service.Timings()
        for _ in range(service.MAX_SAMPLES):
            t.record("retrieve", 1.0)
        for _ in range(service.MAX_SAMPLES):
            t.record("retrieve", 9.0)
        assert t.percentiles()["retrieve"]["p50_ms"] == 9.0

    def test_out_of_order_samples_do_not_disturb_percentiles(self):
        t = service.Timings()
        for v in (9.0, 1.0, 5.0):
            t.record("verify", v)
        assert t.percentiles()["verify"]["p50_ms"] == 5.0


class TestUsage:
    def test_cost_is_null_without_rates_and_says_why(self, monkeypatch):
        monkeypatch.delenv("RAGPIPE_PRICE_IN", raising=False)
        monkeypatch.delenv("RAGPIPE_PRICE_OUT", raising=False)
        u = service.Usage(prompt_tokens=1000, output_tokens=500, thinking_tokens=100, n_generated=1)
        d = u.as_dict()
        assert d["cost_usd"] is None
        assert d["cost_usd_per_query"] is None
        assert "null by design" in d["cost_note"]

    def test_rates_bill_thinking_tokens_as_output(self, monkeypatch):
        monkeypatch.setenv("RAGPIPE_PRICE_IN", "1.0")
        monkeypatch.setenv("RAGPIPE_PRICE_OUT", "10.0")
        u = service.Usage(
            prompt_tokens=1_000_000,
            output_tokens=1_000_000,
            thinking_tokens=1_000_000,
            n_generated=1,
        )
        # 1 * $1 + (1 + 1) * $10 = $21, not $11.
        assert u.as_dict()["cost_usd"] == pytest.approx(21.0)

    def test_one_rate_alone_is_not_enough(self, monkeypatch):
        """Half a rate card would produce a number that is wrong in a plausible
        direction, which is worse than no number."""
        monkeypatch.setenv("RAGPIPE_PRICE_IN", "1.0")
        monkeypatch.delenv("RAGPIPE_PRICE_OUT", raising=False)
        u = service.Usage(prompt_tokens=1_000_000, n_generated=1)
        assert u.as_dict()["cost_usd"] is None

    def test_unparseable_rate_is_ignored_rather_than_crashing_the_endpoint(self, monkeypatch):
        monkeypatch.setenv("RAGPIPE_PRICE_IN", "one dollar")
        monkeypatch.setenv("RAGPIPE_PRICE_OUT", "10.0")
        assert service.Usage(prompt_tokens=10, n_generated=1).as_dict()["cost_usd"] is None

    def test_per_query_divides_by_generations_not_requests(self, monkeypatch):
        monkeypatch.delenv("RAGPIPE_PRICE_IN", raising=False)
        u = service.Usage()
        u.add(None)  # a retrieve-only request: no generation happened
        assert u.as_dict()["n_generated"] == 0
        assert u.as_dict()["tokens_per_query"] is None

    def test_tokens_per_query_averages_over_generations(self):
        u = service.Usage()
        u.add(GenerationResult(text="", model="m", prompt_tokens=100, output_tokens=50))
        u.add(GenerationResult(text="", model="m", prompt_tokens=200, output_tokens=50))
        assert u.as_dict()["tokens_per_query"] == pytest.approx(200.0)


class TestHealth:
    def test_missing_generator_is_not_unready(self):
        """Retrieval is local. A demo that cannot start without a third-party
        credential is a worse demo."""
        r = _client(_state()).get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ready"] is True
        assert body["generator_available"] is False
        assert body["chunks"] == 1

    def test_no_key_is_distinguishable_from_broken(self):
        body = _client(_state(generator_error="GEMINI_API_KEY not set")).get("/health").json()
        assert body["generator_available"] is False
        assert "GEMINI_API_KEY" in body["generator_error"]

    def test_generator_error_is_null_when_there_is_no_error(self):
        assert (
            _client(_state(generator=FakeGenerator())).get("/health").json()["generator_error"]
            is None
        )

    def test_unloaded_index_reports_not_ready(self):
        state = service.AppState()
        assert _client(state).get("/health").json()["ready"] is False


class TestQueryRoute:
    def test_unloaded_index_is_503_not_a_500_or_an_empty_answer(self):
        r = _client(service.AppState()).post("/query", json={"query": "reporting deadline"})
        assert r.status_code == 503

    def test_empty_query_is_rejected_by_validation(self):
        r = _client(_state()).post("/query", json={"query": ""})
        assert r.status_code == 422

    def test_k_is_bounded(self):
        assert _client(_state()).post("/query", json={"query": "x", "k": 500}).status_code == 422

    def test_retrieve_only_never_calls_the_generator(self):
        """Not just "returns early" -- a call that happened and was discarded costs a
        quota unit and is invisible in the response body."""
        gen = FakeGenerator()
        state = _state(generator=gen)
        body = (
            _client(state)
            .post("/query", json={"query": "reporting deadline", "retrieve_only": True})
            .json()
        )
        assert gen.calls == 0
        assert body["answered"] is False
        assert body["reason"] == "retrieve_only"
        assert body["retrieved"][0]["chunk_id"] == "c1"

    def test_retrieve_only_does_not_pollute_the_generation_stats(self):
        state = _state(generator=FakeGenerator())
        client = _client(state)
        client.post("/query", json={"query": "q", "retrieve_only": True})
        latency = client.get("/stats").json()["latency"]
        assert latency["retrieve"]["n"] == 1
        assert latency["total"]["n"] == 1
        assert latency["generate"]["n"] == 0
        assert latency["generate"]["p50_ms"] is None

    def test_without_a_generator_the_reason_names_the_cause(self):
        body = _client(_state()).post("/query", json={"query": "q"}).json()
        assert body["answered"] is False
        assert "no generator" in body["reason"]

    def test_answer_carries_verified_citations_and_the_source_text(self):
        state = _state(generator=FakeGenerator())
        body = _client(state).post("/query", json={"query": "reporting deadline"}).json()
        assert body["answered"] is True
        assert body["verification"]["claimed"] == 1
        assert body["verification"]["verified"] == 1
        assert body["verification"]["fully_grounded"] is True
        (citation,) = body["citations"]
        assert citation["verified"] is True
        # What the document says, which under whitespace normalisation can differ from
        # what the model wrote. Both are returned; neither is inferred from the other.
        assert citation["source_text"] in SOURCE
        assert citation["doc_id"] == "d1"

    def test_a_fabricated_quote_is_reported_unverified_rather_than_dropped(self):
        gen = FakeGenerator(_answer_payload(quote="The sponsor shall submit within 30 days."))
        body = _client(_state(generator=gen)).post("/query", json={"query": "q"}).json()
        assert body["verification"]["verified"] == 0
        assert body["verification"]["fully_grounded"] is False
        assert body["citations"][0]["verified"] is False

    def test_a_model_refusal_is_answered_false_with_its_source(self):
        gen = FakeGenerator(_answer_payload(refused=True, answer="Not addressed."))
        body = _client(_state(generator=gen)).post("/query", json={"query": "q"}).json()
        assert body["answered"] is False
        assert body["refused"] is True
        assert body["refusal_source"] == "model"

    def test_all_four_stages_are_timed_on_a_generated_answer(self):
        state = _state(generator=FakeGenerator())
        client = _client(state)
        client.post("/query", json={"query": "reporting deadline"})
        latency = client.get("/stats").json()["latency"]
        for stage in service.STAGES:
            assert latency[stage]["n"] == 1, stage
            assert latency[stage]["p50_ms"] is not None, stage

    def test_response_timings_are_present_per_request(self):
        state = _state(generator=FakeGenerator())
        timings = _client(state).post("/query", json={"query": "q"}).json()["timings_ms"]
        assert set(timings) == {"retrieve", "generate", "verify", "total"}
        assert timings["total"] >= timings["retrieve"]

    def test_tokens_accumulate_into_stats(self):
        state = _state(generator=FakeGenerator(tokens=(100, 50, 7)))
        client = _client(state)
        client.post("/query", json={"query": "q"})
        client.post("/query", json={"query": "q"})
        usage = client.get("/stats").json()["usage"]
        assert usage["n_generated"] == 2
        assert usage["prompt_tokens"] == 200
        assert usage["thinking_tokens"] == 14
        assert usage["tokens_per_query"] == pytest.approx(157.0)


class TestGenerationFailure:
    def test_upstream_failure_returns_retrieval_rather_than_a_500(self):
        gen = FakeGenerator(error=GenerationError("429 daily quota exhausted"))
        r = _client(_state(generator=gen)).post("/query", json={"query": "q"})
        assert r.status_code == 200
        body = r.json()
        assert body["answered"] is False
        assert "generation failed" in body["reason"]
        assert body["retrieved"][0]["chunk_id"] == "c1"

    def test_a_failed_generation_is_still_timed(self):
        """Otherwise an upstream that fails slowly is invisible in the latency table."""
        gen = FakeGenerator(error=GenerationError("boom"))
        state = _state(generator=gen)
        client = _client(state)
        client.post("/query", json={"query": "q"})
        assert client.get("/stats").json()["latency"]["generate"]["n"] == 1

    def test_a_failed_generation_adds_no_tokens(self):
        gen = FakeGenerator(error=GenerationError("boom"))
        state = _state(generator=gen)
        client = _client(state)
        client.post("/query", json={"query": "q"})
        assert client.get("/stats").json()["usage"]["n_generated"] == 0

    def test_the_reason_is_truncated_so_an_upstream_body_cannot_flood_the_response(self):
        gen = FakeGenerator(error=GenerationError("x" * 5000))
        body = _client(_state(generator=gen)).post("/query", json={"query": "q"}).json()
        assert len(body["reason"]) <= 300


class TestContextAssembly:
    def test_a_hit_with_no_loaded_chunk_is_reported_but_not_sent_to_the_model(self):
        """Retrieval truth and model context are different things. The response shows
        what the retriever ranked; the model only ever sees chunks that exist."""
        state = _state(chunks=[_chunk("c1")], retriever=FakeRetriever(["c1", "ghost"]))
        state.generator = FakeGenerator()
        body = _client(state).post("/query", json={"query": "q", "k": 2}).json()
        ranked = [r["chunk_id"] for r in body["retrieved"]]
        assert ranked == ["c1", "ghost"]
        assert body["retrieved"][1]["doc_id"] is None
        assert body["verification"]["verified"] == 1


class TestDashboard:
    def test_the_page_is_served_from_root(self):
        r = _client(_state()).get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        # `agentrag` is the user-facing name (what you type to run it); `ragpipe` stays
        # the Python package. Asserted so the two cannot silently swap back.
        assert "<title>agentrag" in r.text
        assert "<h1>agentrag</h1>" in r.text

    def test_the_page_calls_the_same_endpoints_this_service_exposes(self):
        """A dashboard that fetches a route the API does not serve is broken in a way
        no Python test would otherwise notice."""
        page = _client(_state()).get("/").text
        for route in ('fetch("/health")', 'fetch("/query"'):
            assert route in page, route

    def test_the_pages_verified_list_matches_the_verifier(self):
        """The page duplicates `VERIFIED_METHODS` in JavaScript to colour its badges.
        If the verifier's definition changes, a green badge in the browser would keep
        claiming a citation is verified after the report stopped counting it."""
        from ragpipe import dashboard

        assert dashboard.verified_methods_match_page()

    def test_the_page_never_treats_the_ambiguous_tier_as_verified(self):
        """Phase 5 demoted `line_number_ambiguous` after it verified a meaning change
        (a model's "40 CFR" against a document's "21 CFR"). It must not be green."""
        from ragpipe import citations as cit
        from ragpipe import dashboard

        assert cit.LINE_NUMBER_AMBIGUOUS not in cit.VERIFIED_METHODS
        assert f'AMBIGUOUS = ["{cit.LINE_NUMBER_AMBIGUOUS}"]' in dashboard.DASHBOARD_HTML

    def test_the_page_does_not_promise_a_dollar_figure(self):
        """Cost is null by design everywhere else in this project, so the UI is the one
        place a per-token rate could quietly get invented to fill a field.

        The previous version of this test was `re.search(r"\\$\\s*[\\d{]", page)`, which only
        catches `$` immediately followed by a digit -- i.e. a literal. The way a dollar
        figure actually reaches this page is JS concatenation, where `$` is followed by a
        quote: `'est. cost $' + (tokens * rate).toFixed(4)`. That passed. Same shape as the
        `\\bcomment\\b` regex that could never match "comments".

        So the check is now structural: every `$` in the page must be the DOM helper,
        which appears only as `$(` or in its own declaration. Any other `$` fails.
        """
        page = _client(_state()).get("/").text
        for match in re.finditer(r"\$", page):
            after = page[match.end() : match.end() + 1]
            before = page[max(0, match.start() - 8) : match.start()]
            is_helper_call = after == "("
            is_helper_decl = before.endswith("const ")
            context = page[max(0, match.start() - 40) : match.end() + 40]
            assert is_helper_call or is_helper_decl, f"a `$` that is not the DOM helper: {context}"
        assert "USD" not in page

    def test_the_page_never_computes_a_cost_from_a_rate_of_its_own(self):
        """The narrower, truer version of the check above.

        An earlier version asserted `"cost_usd" not in page`, which forbade the page from
        *reading* the server's field. That is the wrong target: `/stats` returns
        `cost_usd: null` with a note saying why, and surfacing that null is a feature of
        this project rather than a risk. The risk is the page **inventing a rate** —
        multiplying a token count by a number of its own and printing the product.

        So: reading and displaying the server's value is allowed; client-side arithmetic
        on a token count is not.
        """
        page = _client(_state()).get("/").text
        offenders = re.findall(r"(?:tokens?|prompt|output|thinking)\w*\s*\*\s*[\d.]", page)
        assert not offenders, f"the page computes a cost from its own rate: {offenders}"

    def test_that_arithmetic_check_catches_an_invented_rate(self):
        """Break-test, so narrowing the guard above did not hollow it out."""
        from ragpipe import dashboard

        injected = dashboard.DASHBOARD_HTML.replace(
            "    const bits = [];",
            '    const bits = ["cost " + (u.prompt_tokens * 3e-7).toFixed(4)];',
            1,
        )
        assert injected != dashboard.DASHBOARD_HTML, "injection point not found"
        offenders = re.findall(r"(?:tokens?|prompt|output|thinking)\w*\s*\*\s*[\d.]", injected)
        assert offenders, "the arithmetic guard would not catch an invented rate"

    def test_the_page_says_cost_is_in_tokens_not_dollars(self):
        page = _client(_state()).get("/").text
        assert "tokens, not dollars" in page

    def test_that_dollar_check_catches_a_concatenated_cost(self):
        """The guard above is only worth having if it fails on the realistic regression."""
        from ragpipe import dashboard

        injected = dashboard.DASHBOARD_HTML.replace(
            "out.push('<span class=\"pill\">tokens <b>'",
            "out.push('<span class=\"pill\">est. cost $' + (d.tokens.prompt * 3e-7).toFixed(4) "
            "+ '</b></span>');\n    out.push('<span class=\"pill\">tokens <b>'",
            1,
        )
        assert injected != dashboard.DASHBOARD_HTML, "injection point not found"
        offenders = [
            m
            for m in re.finditer(r"\$", injected)
            if injected[m.end() : m.end() + 1] != "("
            and not injected[max(0, m.start() - 8) : m.start()].endswith("const ")
        ]
        assert offenders, "the structural check would not catch a concatenated cost"

    def test_both_sides_of_a_citation_render_unconditionally(self):
        """The dashboard's central design decision, and it was untested.

        `dashboard.py` states that both the model's quote and the document's text are
        shown even when identical, because showing the source only on a mismatch would
        make `verified` the one state in which the reader cannot check the work. A
        reviewer implemented exactly the rejected behaviour and the whole suite stayed
        green, so the rationale was load-bearing and unguarded.
        """
        from ragpipe import dashboard

        page = dashboard.DASHBOARD_HTML
        start = page.index("function renderCites")
        body = page[start : page.index("\nfunction ", start + 1)]
        assert "document says" in body
        # The source panel must not be gated on the verdict, in either spelling.
        assert "c.verified" not in body
        assert "VERIFIED.includes" not in body
        # And the only branch on source_text is "did we locate it", not "did it verify".
        assert "c.source_text ?" in body

    def test_the_page_only_fetches_routes_the_app_actually_serves(self):
        """Derived from the app, not from a hardcoded list.

        The previous version asserted two literal strings were present, so it could not
        notice the direction its own docstring named: the page fetching a route that does
        not exist. Nothing else in the suite covers that.
        """
        state = _state()
        app = service.create_app(state)
        served = {getattr(r, "path", None) for r in app.routes}
        page = _client(state).get("/").text
        fetched = set(re.findall(r'fetch\(\s*"([^"]+)"', page))
        assert fetched, "no fetch() calls found; the extraction is broken"
        assert fetched <= served, f"page fetches routes the app does not serve: {fetched - served}"


class TestEmptyContext:
    """A query that retrieves nothing must not reach the model.

    The bug this guards was worse than a wasted request: the empty context went to the
    generator, which spent a daily quota unit and then *answered*, citing a quote that
    verified `unverified`. Ungrounded output is the exact failure this project exists to
    prevent, and it arrived through the one path with no retrieval evidence at all.
    """

    def test_a_zero_hit_query_never_calls_the_generator(self):
        gen = FakeGenerator()
        state = _state(retriever=FakeRetriever([]), generator=gen)
        body = _client(state).post("/query", json={"query": "zzzqqqxyzzy"}).json()
        assert gen.calls == 0
        assert body["answered"] is False
        assert body["refused"] is True
        assert body["refusal_source"] == "no_context"
        assert body["retrieved"] == []

    def test_hits_that_map_to_no_loaded_chunk_also_refuse(self):
        """Retrieval succeeded, but every ranked id is missing from the index, so there
        is still nothing to ground on. The response keeps the ranking as evidence."""
        gen = FakeGenerator()
        state = _state(
            chunks=[_chunk("c1")], retriever=FakeRetriever(["ghost1", "ghost2"]), generator=gen
        )
        body = _client(state).post("/query", json={"query": "q", "k": 2}).json()
        assert gen.calls == 0
        assert body["refusal_source"] == "no_context"
        assert [r["chunk_id"] for r in body["retrieved"]] == ["ghost1", "ghost2"]

    def test_the_refusal_is_still_timed(self):
        state = _state(retriever=FakeRetriever([]), generator=FakeGenerator())
        client = _client(state)
        client.post("/query", json={"query": "q"})
        latency = client.get("/stats").json()["latency"]
        assert latency["retrieve"]["n"] == 1
        assert latency["total"]["n"] == 1
        assert latency["generate"]["n"] == 0

    def test_no_tokens_are_billed_for_a_refusal(self):
        state = _state(retriever=FakeRetriever([]), generator=FakeGenerator())
        client = _client(state)
        client.post("/query", json={"query": "q"})
        assert client.get("/stats").json()["usage"]["n_generated"] == 0


class TestPriceEnvIsValidated:
    """A nonsense rate must leave `cost_usd` null, not produce a nonsense cost."""

    def test_a_negative_rate_is_rejected(self, monkeypatch):
        monkeypatch.setenv("RAGPIPE_PRICE_IN", "-5")
        monkeypatch.setenv("RAGPIPE_PRICE_OUT", "10")
        assert service.Usage(prompt_tokens=1_000_000, n_generated=1).as_dict()["cost_usd"] is None

    def test_nan_is_rejected_because_it_is_not_even_valid_json(self, monkeypatch):
        monkeypatch.setenv("RAGPIPE_PRICE_IN", "nan")
        monkeypatch.setenv("RAGPIPE_PRICE_OUT", "10")
        d = service.Usage(prompt_tokens=1_000_000, n_generated=1).as_dict()
        assert d["cost_usd"] is None
        json.dumps(d, allow_nan=False)  # would raise if a NaN reached the body

    def test_infinity_is_rejected(self, monkeypatch):
        monkeypatch.setenv("RAGPIPE_PRICE_IN", "inf")
        monkeypatch.setenv("RAGPIPE_PRICE_OUT", "10")
        assert service.Usage(prompt_tokens=10, n_generated=1).as_dict()["cost_usd"] is None

    def test_zero_is_a_legitimate_rate(self, monkeypatch):
        """The free tier really does bill nothing. Zero must compute, not be rejected
        as falsy -- that would make the one rate this project actually knows unusable."""
        monkeypatch.setenv("RAGPIPE_PRICE_IN", "0")
        monkeypatch.setenv("RAGPIPE_PRICE_OUT", "0")
        d = service.Usage(prompt_tokens=1_000_000, output_tokens=5, n_generated=1).as_dict()
        assert d["cost_usd"] == 0.0


class TestDashboardComponents:
    """One test per UI component, because the page has no runtime of its own here.

    These are structural: they assert the markup and the script still contain the wiring
    each component needs. That is weaker than driving a browser, and it is what catches
    the failure that actually happens — a render function reading a response field the
    API stopped returning, or an element id that exists in one half only.
    """

    @staticmethod
    def _page():
        return _client(_state()).get("/").text

    def test_every_element_id_exists_in_both_the_markup_and_the_script(self):
        """An id in the markup that nothing writes to is dead weight; an id the script
        writes to that the markup lacks is a silent no-op at runtime."""
        page = self._page()
        declared = set(re.findall(r'id="([^"]+)"', page))
        used = set(re.findall(r'\$\("([^"]+)"\)', page))
        assert not declared - used, f"declared but never written: {sorted(declared - used)}"
        assert not used - declared, f"written but never declared: {sorted(used - declared)}"

    def test_the_four_result_cards_are_present(self):
        page = self._page()
        for card in ("answerCard", "citeCard", "retrCard", "timeCard"):
            assert f'id="{card}"' in page, card

    def test_result_cards_start_hidden_so_the_first_paint_is_not_empty_boxes(self):
        page = self._page()
        for card in ("answerCard", "citeCard", "retrCard", "timeCard"):
            match = re.search(r'class="([^"]*)" id="' + card + '"', page)
            assert match and "hidden" in match.group(1), card

    def test_the_diff_is_wired_into_the_citation_renderer(self):
        page = self._page()
        assert "function diffTokens" in page
        assert "function renderDiff" in page
        start = page.index("function renderCites")
        body = page[start : page.index("\nfunction ", start + 1)]
        assert "renderDiff(c.quote, c.source_text" in body

    def test_the_diff_marks_both_directions_distinctly(self):
        """`<ins>` and `<del>` rather than one highlight colour: which side a token is
        missing from is the whole content of the comparison."""
        page = self._page()
        assert "<del>" in page and "<ins>" in page
        assert "ins { background" in page and "del { background" in page

    def test_an_identical_pair_says_so_instead_of_rendering_an_empty_diff(self):
        page = self._page()
        assert "function diffIsClean" in page
        assert "nothing to " in page and "diff" in page

    def test_verdicts_carry_a_glyph_and_not_only_a_colour(self):
        """Colour-only encoding fails a colour-blind reader and a greyscale screenshot."""
        page = self._page()
        assert "GLYPH" in page
        assert "\\u2713" in page and "\\u2717" in page

    def test_each_refusal_source_is_described_as_what_actually_happened(self):
        """The API distinguishes three refusals and only one of them involved a model.

        Found by cross-checking the page's field reads against every response branch: the
        banner originally said "the model was told to answer only from the retrieved text
        and reported that it could not" for *all* refusals, which is a false statement
        about `no_context` and `score_gate` — both refuse before any request is sent.
        """
        page = self._page()
        assert 'refusal_source === "no_context"' in page
        assert 'refusal_source === "score_gate"' in page
        # The pre-generation refusals must say no request was made, not that a model spoke.
        no_ctx = page[page.index('refusal_source === "no_context"') :][:600]
        assert "no request was sent" in no_ctx or "No quota spent" in no_ctx

    def test_the_example_queries_include_one_that_should_refuse(self):
        """A refusal is a feature of this system, so the demo has to be able to show one
        without the visitor having to invent an unanswerable question."""
        page = self._page()
        assert "const EXAMPLES" in page
        block = page[page.index("const EXAMPLES") : page.index("const $ =")]
        assert block.count("{ label:") >= 3
        assert "refus" in block.lower()

    def test_retrieved_rows_show_the_chunk_text_and_whether_it_was_cited(self):
        """A retrieval demo showing only ids and scores cannot be judged: "was that a
        good hit?" is a question about the text."""
        page = self._page()
        assert "r.text" in page and "r.cited" in page
        assert "n_chars" in page and "truncated" in page
        assert 'class="badge"' in page

    def test_the_permalink_reproduces_a_query(self):
        page = self._page()
        assert "function permalink" in page
        assert "URLSearchParams" in page
        assert "history.replaceState" in page

    def test_the_status_line_is_announced_to_assistive_tech(self):
        page = self._page()
        assert re.search(r'id="status"[^>]*aria-live="polite"', page)
        assert re.search(r'id="health"[^>]*aria-live="polite"', page)

    def test_decorative_glyphs_are_hidden_from_assistive_tech(self):
        page = self._page()
        assert 'aria-hidden="true"' in page

    def test_motion_is_disabled_for_readers_who_ask_for_that(self):
        page = self._page()
        assert "prefers-reduced-motion" in page

    def test_focus_is_visible_for_keyboard_users(self):
        page = self._page()
        assert "focus-visible" in page

    def test_the_elapsed_timer_exists_because_generation_takes_seconds(self):
        page = self._page()
        assert "function startTicker" in page and "function stopTicker" in page
        assert "setInterval" in page and "clearInterval" in page

    def test_the_cumulative_strip_reads_stats_rather_than_recomputing_it(self):
        page = self._page()
        assert 'fetch("/stats")' in page
        assert "p50_ms" in page and "p95_ms" in page

    def test_the_page_never_indexes_embed_text(self):
        """`embed_text` carries a prepended heading the model never sees as body content,
        so showing it would misrepresent what was retrieved."""
        assert "embed_text" not in self._page()


class TestDashboardReadsPlainly:
    """The page is for a visitor who has never seen this project.

    A screenshot review found it written for someone who already knew the system: a
    dropdown labelled `k=5`, a checkbox labelled "retrieve only", a footer strip reading
    `retrieve p50 0.373ms / p95 0.373ms (n=1)`, and a raw API token printed mid-sentence
    ("No answer requested. retrieve_only Retrieval ran and is shown below."). These lock
    in the fixes.
    """

    @staticmethod
    def _visible(page: str) -> str:
        """Roughly what a reader sees: markup with script and style stripped."""
        return re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S)

    def _page(self):
        return _client(_state()).get("/").text

    def test_the_user_facing_name_is_agentrag(self):
        page = self._page()
        assert "<title>agentrag" in page
        assert "<h1>agentrag</h1>" in page

    def test_no_bare_k_label_on_the_control(self):
        """`k` is a variable name from the code. The control says what it does."""
        visible = self._visible(self._page())
        assert "k=5" not in visible and "k=3" not in visible
        assert "passages to read" in visible.lower()

    def test_the_search_only_switch_explains_itself(self):
        visible = self._visible(self._page())
        assert "retrieve only" not in visible.lower()
        assert "search only" in visible.lower()
        assert "skip the AI answer" in visible

    def test_the_unit_is_called_a_passage_in_prose(self):
        """One word for the searchable unit. The teaching card said "passages" while the
        results table said "chunks", which reads as two different things.

        The rule distinguishes prose from code: the English word is preceded by
        whitespace, whereas every legitimate use is part of an identifier
        (`chunk_id`, `chunktext`, `h.chunks`, `details.chunk`, `class="chunk"`), where the
        preceding character is a dot, quote or word character. An earlier version of this
        test tried to extract quoted strings with a regex, which matched across the whole
        CSS block because `"` appears there only sparsely.
        """
        page = self._page()
        prose_uses = re.findall(r"\s(chunks?)\b", page)
        assert not prose_uses, f"prose still says chunk instead of passage: {prose_uses}"
        assert "passages ranked" in page
        assert "passages, searched by keyword" in page

    def test_that_vocabulary_rule_would_catch_a_regression(self):
        """Otherwise the rule above could be passing because it matches nothing."""
        page = self._page().replace("passages ranked", "chunks ranked", 1)
        assert re.findall(r"\s(chunks?)\b", page)

    def test_there_is_a_plain_language_explainer_before_any_query_runs(self):
        visible = self._visible(self._page())
        assert "What happens when you ask" in visible
        assert "What do the words mean?" in visible

    def test_the_jargon_that_remains_is_defined_on_the_page(self):
        visible = self._visible(self._page())
        for term in ("BM25", "passage", "verified"):
            assert term in visible, term
        # Each has a definition, not just a mention.
        assert "<dt>BM25 / keyword search</dt>" in visible
        assert "<dt>passage</dt>" in visible
        assert "<dt>verified</dt>" in visible

    def test_developer_detail_is_folded_away_not_dumped_on_the_visitor(self):
        visible = self._visible(self._page())
        devbox = visible[visible.index('class="devbox"') :]
        for route in ("/health", "/query", "/stats", "/docs"):
            assert route in devbox, route
        assert "RAGPIPE_PRICE_IN" in devbox

    def test_the_latency_strip_withholds_percentile_language_at_tiny_n(self):
        """A p50 over one observation is the observation. The strip printed
        "p50 0.373ms / p95 0.373ms (n=1)" -- jargon around a single number."""
        page = self._page()
        assert "MIN_TYPICAL" in page and "MIN_TAIL" in page
        assert "p50 " not in self._visible(page)
        assert "STAGE_WORDS" in page

    def test_no_raw_api_token_is_printed_as_prose(self):
        """The original defect was `retrieve_only` appearing mid-sentence: "No answer
        requested. retrieve_only Retrieval ran and is shown below."

        Surfacing the server's reason is not the same thing, and the fallback branch does
        it on purpose — for an *unrecognised* reason, showing what the service said beats
        inventing a cause, which is how a quota failure got reported as a missing API key.
        So the rule is about framing: a reason may be shown under an explicit "Reason:"
        label, never spliced into a sentence.
        """
        page = self._page()
        assert 'esc(d.reason || "")' not in page, "reason spliced into prose unlabelled"
        assert 'd.reason === "retrieve_only"' in page or 'reason === "retrieve_only"' in page
        # Wherever the raw reason is rendered, it must be introduced as one.
        for match in re.finditer(r"esc\(reason[^)]*\)", page):
            window = page[max(0, match.start() - 200) : match.start()]
            assert "Reason:" in window, "raw reason rendered without a label"


class TestCitationDeepLink:
    """The deep link has to *execute*, not merely exist.

    The first version of this test asserted the string `location.hash === "#citations"`
    appeared in the page — and passed for days against code that could never run.
    `permalink()` calls `history.replaceState(null, "", "?" + params)`, and a query-only
    relative reference **drops the fragment**, so by the time the fetch resolved
    `location.hash` was always `""`. `bin/build-demo` depended on it and silently captured
    the default viewport instead.

    These tests execute the two pieces of logic under node rather than grepping for them.
    """

    @staticmethod
    def _page():
        return _client(_state()).get("/").text

    def _run(self, js):
        import shutil
        import subprocess

        node = shutil.which("node")
        if node is None:
            pytest.skip("needs node to execute the page's JS")
        proc = subprocess.run([node, "-e", js], capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr[-600:]
        return proc.stdout.strip()

    def test_the_rewritten_url_keeps_the_fragment(self):
        """The actual defect: this is what `permalink` produces, resolved by a real URL
        parser rather than by reading the code and hoping."""
        out = self._run(
            'const p=new URLSearchParams();p.set("q","x");p.set("k","5");'
            'const hash="#citations";'
            'console.log(new URL("?"+p.toString()+hash,'
            ' "http://127.0.0.1:8000/?q=y&k=5#citations").hash);'
        )
        assert out == "#citations", f"the fragment is dropped: {out!r}"

    def test_a_query_only_rewrite_would_drop_it(self):
        """Break-test. Without this, the test above could pass for the wrong reason."""
        out = self._run(
            'console.log(JSON.stringify(new URL("?q=x&k=5",'
            ' "http://127.0.0.1:8000/?q=y&k=5#citations").hash));'
        )
        assert out == '""', f"expected the fragment to be lost, got {out}"

    def test_permalink_appends_the_fragment(self):
        page = self._page()
        assert "p.toString() + location.hash" in page

    def test_the_intent_is_captured_before_the_url_is_rewritten(self):
        """Reading `location.hash` after the round trip is fragile by construction: it asks
        the address bar what the *request* wanted, seconds later."""
        page = self._page()
        assert "const wantCitations = location.hash" in page
        ask = page[page.index("async function ask()") :]
        capture = ask.index("wantCitations")
        rewrite = ask.index("permalink(query")
        assert capture < rewrite, "the fragment is read after permalink() rewrites the URL"

    def test_the_scroll_still_waits_for_the_card_to_exist(self):
        page = self._page()
        block = page[page.index("if (wantCitations") :][:200]
        assert 'classList.contains("hidden")' in block
        assert "scrollIntoView" in block


class TestNoAnswerReasons:
    """A keyless service and a quota-exhausted one look identical on this page and are
    fixed completely differently. The first version reported both as "running without an
    API key", which is a confident false statement in the quota case — and that case is
    the common one on a free tier capped at 20 requests a day."""

    def test_a_quota_failure_is_not_reported_as_a_missing_key(self):
        gen = FakeGenerator(
            error=GenerationError(
                "gemini-3.6-flash: daily free-tier quota exhausted (20 requests/day/model)",
                daily_quota=True,
            )
        )
        body = _client(_state(generator=gen)).post("/query", json={"query": "q"}).json()
        assert "quota" in body["reason"].lower()
        assert body["refusal_source"] == "daily_quota"

    def test_the_daily_quota_decision_travels_as_a_field_not_as_prose(self):
        """A per-minute 429's body also reads "you exceeded your current quota", so a
        client scanning the message cannot tell "come back tomorrow" from "retry in a
        moment". The upstream already decides this off the structured `quotaId`; the
        service has to carry the decision rather than re-encode it in English."""
        transient = FakeGenerator(
            error=GenerationError(
                "gemini-3.6-flash: giving up after 6 attempts. http 429: "
                "You exceeded your current quota, please check your plan and billing details."
            )
        )
        body = _client(_state(generator=transient)).post("/query", json={"query": "q"}).json()
        assert "quota" in body["reason"].lower()
        assert body["refusal_source"] == "generation_error"

    def test_the_two_quota_conditions_get_opposite_advice_in_the_page(self):
        page = _client(_state()).get("/").text
        assert 'd.refusal_source === "daily_quota"' in page
        assert "The daily AI quota is used up." in page
        assert "Rate limited, briefly." in page
        assert "Ask again in a few seconds." in page

    def test_the_daily_branch_is_tested_before_the_substring_fallback(self):
        """Order is the whole defect. With the substring first, every transient 429 was
        reported as an exhausted day."""
        page = _client(_state()).get("/").text
        assert page.index('d.refusal_source === "daily_quota"') < page.index(
            "/quota/i.test(reason)"
        )

    def test_the_missing_key_wording_belongs_to_exactly_one_branch(self):
        """The original defect was one message serving three causes. Rather than slice a
        character window -- which silently reaches into the *next* branch, and did -- pin
        that the keyless wording occurs once and that the branch owning it is the one
        keyed on `no generator`."""
        page = _client(_state()).get("/").text
        # Comment lines mention the wording while explaining the original defect, so count
        # only what a reader could actually be shown.
        shown = [ln for ln in page.splitlines() if "without an API key" in ln and "//" not in ln]
        assert len(shown) == 1
        before = page[: page.index(shown[0])]
        assert before.rindex("/no generator/i.test(reason)") > before.rindex(
            "/quota/i.test(reason)"
        )

    def test_the_three_named_reasons_are_all_handled(self):
        page = _client(_state()).get("/").text
        for probe in ('reason === "retrieve_only"', "/quota/i.test(reason)", "/no generator/i"):
            assert probe in page, probe

    def test_an_unrecognised_reason_is_shown_rather_than_guessed(self):
        """The fallback must surface the server's actual reason instead of inventing one."""
        page = _client(_state()).get("/").text
        tail = page[page.index("/no generator/i.test(reason)") :][:700]
        assert "Reason:" in tail
        assert "esc(reason" in tail


class TestTheCitationsOffsetIsPublished:
    """`bin/build-demo` crops scene 3 to `body[data-cite-top]`.

    It has to, because headless Chrome's `--screenshot` renders from the document origin
    and ignores the scroll the `#citations` deep link performs -- confirmed over CDP, where
    the page reports `scrollY: 610` and `citeTopViewport: 0`. A screenshot tool cannot ask
    a layout engine where a card is, and a hardcoded offset against a page whose height
    depends on the answer is the guess the deep link was added to remove. So the page
    publishes the number, and these tests keep it published: drop it and the build fails
    loudly, but rename it and the build fails at the crop instead, after spending a
    generation request.
    """

    def test_the_page_publishes_the_offset(self):
        from ragpipe import dashboard

        assert "dataset.citeTop" in dashboard.DASHBOARD_HTML

    def test_the_offset_is_a_document_position_not_a_viewport_one(self):
        """`getBoundingClientRect().top` alone is relative to the viewport, so on a scrolled
        page it is short by exactly the scroll -- which is the bug this replaced."""
        from ragpipe import dashboard

        page = dashboard.DASHBOARD_HTML
        assert "getBoundingClientRect().top + window.scrollY" in page

    def test_the_attribute_name_matches_what_the_build_script_greps_for(self):
        """The two are a contract across a process boundary with nothing else holding them
        together: `dataset.citeTop` serialises to `data-cite-top`."""
        from pathlib import Path

        from ragpipe import dashboard

        script = Path(__file__).resolve().parents[1] / "bin" / "build-demo"
        if not script.exists():  # pragma: no cover - the script ships with the repo
            pytest.skip("bin/build-demo is not present")
        assert "data-cite-top" in script.read_text()
        assert "dataset.citeTop" in dashboard.DASHBOARD_HTML

    def test_the_offset_is_written_after_the_citations_are_rendered(self):
        """Measured before `renderCites` it would describe an empty card, and the crop
        would land on whatever happened to be at that offset."""
        from ragpipe import dashboard

        page = dashboard.DASHBOARD_HTML
        assert page.index("renderCites(d.citations)") < page.index("dataset.citeTop")

    def test_the_deep_link_scroll_is_instant(self):
        """A deep link is a request for the proof, not a reading gesture: animating past
        everything above it is motion with no information in it.

        Comment lines are excluded because the comment above the call *explains* the
        choice by quoting it, so a plain substring search passes on the prose alone --
        which it did, and a mutation that changed the real call went uncaught.
        """
        from ragpipe import dashboard

        code = [
            line
            for line in dashboard.DASHBOARD_HTML.splitlines()
            if not line.lstrip().startswith("//")
        ]
        assert any("scrollIntoView" in line and 'behavior: "instant"' in line for line in code)
