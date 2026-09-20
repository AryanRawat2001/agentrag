"""Tests for the generator transport.

Weighted toward quota handling and request shaping, because that is where this module
actually failed -- twice, and the second failure was a misdiagnosis of the first.

The first Phase 5 eval run scored 3 of 80 queries. Every retry spends a quota unit, so
~480 attempts went into a budget that turned out to be **20 requests per day per
model**, not per minute. The 429 text says "Please retry in 4.411926172s.", which is
what produced the wrong reading; the structured `quotaId` says `PerDay`. Hence two
distinct behaviours under test below: pacing (for the unpublished per-minute cap) and
immediate failure without retries (for the per-day cap, which no backoff can clear).

Nothing about either is visible in a happy-path test.

No network. A fake transport records the requests it was given.

"""

from __future__ import annotations

import json

import httpx
import pytest

from ragpipe import generation as gen

REAL_429_BODY = json.dumps(
    {
        "error": {
            "code": 429,
            "message": (
                "You exceeded your current quota, please check your plan and billing "
                "details. * Quota exceeded for metric: "
                "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
                "limit: 20, model: gemini-3.6-flash\nPlease retry in 4.411926172s."
            ),
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "4s",
                }
            ],
        }
    }
)

OK_BODY = {
    "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}, "finishReason": "STOP"}],
    "usageMetadata": {
        "promptTokenCount": 100,
        "candidatesTokenCount": 20,
        "thoughtsTokenCount": 0,
    },
}


def _client(responses: list[httpx.Response]) -> tuple[httpx.Client, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0) if queue else httpx.Response(200, json=OK_BODY)

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def _generator(responses=None, **kwargs):
    client, seen = _client(responses or [])
    slept: list[float] = []
    g = gen.GeminiGenerator(
        api_key="test-key",
        env_path=None,
        client=client,
        sleep=slept.append,
        min_interval_s=kwargs.pop("min_interval_s", 0.0),
        **kwargs,
    )
    return g, seen, slept


class TestParseRetryDelay:
    def test_reads_structured_retry_info(self):
        assert gen.parse_retry_delay(REAL_429_BODY) == 4.0

    def test_falls_back_to_the_message_text(self):
        """RetryInfo is not always present; the message text carries it too."""
        body = json.dumps({"error": {"message": "Please retry in 12.5s.", "details": []}})
        assert gen.parse_retry_delay(body) == 12.5

    def test_returns_none_when_absent(self):
        assert gen.parse_retry_delay(json.dumps({"error": {"message": "nope"}})) is None

    def test_survives_non_json_body(self):
        assert gen.parse_retry_delay("<html>502 Bad Gateway</html>") is None

    def test_survives_json_that_is_not_an_object(self):
        assert gen.parse_retry_delay("[1, 2, 3]") is None


class TestBackoff:
    def test_header_wins_over_body(self):
        """The floor applies here too. An earlier version returned the raw header
        value, and this test asserted exactly 7.0 -- pinning the inconsistency in
        place rather than catching it."""
        assert gen._backoff_seconds(0, "7", REAL_429_BODY) == pytest.approx(7.5)

    def test_body_delay_used_when_no_header(self):
        """Honouring the server's number beats guessing: it knows when the quota
        window rolls over. A small floor is added so we do not arrive early."""
        assert gen._backoff_seconds(0, None, REAL_429_BODY) == pytest.approx(4.5)

    def test_exponential_fallback_when_server_says_nothing(self):
        waits = [gen._backoff_seconds(i, None, "") for i in range(5)]
        assert all(w > 0 for w in waits)
        assert max(waits) <= gen.MAX_BACKOFF_S

    def test_malformed_header_falls_through(self):
        assert gen._backoff_seconds(0, "not-a-number", REAL_429_BODY) == pytest.approx(4.5)

    def test_never_exceeds_the_cap(self):
        huge = json.dumps({"error": {"message": "Please retry in 99999s.", "details": []}})
        assert gen._backoff_seconds(0, None, huge) == gen.MAX_BACKOFF_S


class TestPacing:
    """The fix for the retry storm: stay under the cap instead of reacting to it."""

    def test_first_request_is_not_delayed(self):
        g, _, slept = _generator(min_interval_s=3.21)
        g.generate("hi")
        assert slept == []

    def test_second_request_waits_the_interval(self):
        # Only `_pace` reads the injected clock -- `generate` times latency off
        # `time.monotonic` directly. So: 0.0 for the first pace, 0.5 for the second
        # (0.5s elapsed), 3.21 for the re-read after sleeping.
        clock = iter([0.0, 0.5, 3.21])
        client, _ = _client([])
        slept: list[float] = []
        g = gen.GeminiGenerator(
            api_key="k",
            env_path=None,
            client=client,
            sleep=slept.append,
            min_interval_s=3.21,
            clock=lambda: next(clock),
        )
        g.generate("one")
        g.generate("two")
        assert slept and slept[0] == pytest.approx(2.71)

    def test_daily_cap_is_recorded_as_a_named_constant(self):
        """The binding free-tier constraint is 20 requests per DAY per model, read
        off `quotaId=GenerateRequestsPerDayPerProjectPerModel-FreeTier`. An earlier
        version of this module read the same 429 as a per-minute limit, because the
        message text suggests a 4-second retry."""
        assert gen.FREE_TIER_REQUESTS_PER_DAY == 20
        assert gen.MIN_REQUEST_INTERVAL_S > 0

    def test_zero_interval_disables_pacing(self):
        g, _, slept = _generator(min_interval_s=0.0)
        g.generate("a")
        g.generate("b")
        assert slept == []

    def test_retries_are_paced_too(self):
        """A retry spends a quota unit exactly like a first attempt -- which is why
        the daily budget vanished in one run."""
        responses = [httpx.Response(429, text=REAL_429_BODY), httpx.Response(200, json=OK_BODY)]
        client, seen = _client(responses)
        slept: list[float] = []
        ticks = iter([0.0, 0.0, 0.1, 10.0, 10.0, 20.0])
        g = gen.GeminiGenerator(
            api_key="k",
            env_path=None,
            client=client,
            sleep=slept.append,
            min_interval_s=3.21,
            clock=lambda: next(ticks),
        )
        g.generate("x")
        assert len(seen) == 2
        # One backoff sleep for the 429, plus at least one pacing sleep.
        assert len(slept) >= 2


class TestRetryBehaviour:
    def test_recovers_after_a_429(self):
        g, seen, _ = _generator(
            [httpx.Response(429, text=REAL_429_BODY), httpx.Response(200, json=OK_BODY)]
        )
        result = g.generate("q")
        assert result.attempts == 2
        assert len(seen) == 2

    def test_recovers_after_a_503(self):
        """503 "high demand" hit 2 of ~10 real probe calls -- transient, not a
        rejection. An earlier probe run misread these as invalid parameters."""
        g, _, _ = _generator(
            [httpx.Response(503, text="high demand"), httpx.Response(200, json=OK_BODY)]
        )
        assert g.generate("q").attempts == 2

    def test_gives_up_after_max_attempts_and_names_the_last_error(self):
        g, seen, _ = _generator([httpx.Response(429, text=REAL_429_BODY)] * 6, max_attempts=6)
        with pytest.raises(gen.GenerationError, match="gave up after 6"):
            g.generate("q")
        assert len(seen) == 6

    def test_400_is_not_retried(self):
        """A bad request retried six times buries the actual error under a backoff
        loop. Only 429 and 5xx are transient."""
        g, seen, _ = _generator([httpx.Response(400, text="INVALID_ARGUMENT")])
        with pytest.raises(gen.GenerationError, match="400"):
            g.generate("q")
        assert len(seen) == 1

    def test_transport_errors_are_retried(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("boom")
            return httpx.Response(200, json=OK_BODY)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        g = gen.GeminiGenerator(
            api_key="k", env_path=None, client=client, sleep=lambda _: None, min_interval_s=0.0
        )
        assert g.generate("q").attempts == 2

    def test_empty_candidates_raises_rather_than_returning_empty_text(self):
        """A safety block must not score as an unhelpful-but-valid answer."""
        g, _, _ = _generator([httpx.Response(200, json={"promptFeedback": {"blockReason": "X"}})])
        with pytest.raises(gen.GenerationError, match="no candidates"):
            g.generate("q")


class TestRequestShaping:
    def test_thinking_level_defaults_to_minimal(self):
        """Thinking is on by default and `thinkingBudget: 0` is a 400, so this is
        the only setting that reaches zero thinking tokens."""
        g, _, _ = _generator()
        body = g.build_body("hi")
        assert body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "minimal"

    def test_schema_sets_json_mime_type(self):
        g, _, _ = _generator()
        body = g.build_body("hi", schema={"type": "object"})
        cfg = body["generationConfig"]
        assert cfg["responseMimeType"] == "application/json"
        assert cfg["responseSchema"] == {"type": "object"}

    def test_no_schema_leaves_response_format_unset(self):
        g, _, _ = _generator()
        cfg = g.build_body("hi")["generationConfig"]
        assert "responseSchema" not in cfg
        assert "responseMimeType" not in cfg

    def test_system_instruction_is_separate_from_the_user_turn(self):
        g, _, _ = _generator()
        body = g.build_body("question", system="rules")
        assert body["systemInstruction"]["parts"][0]["text"] == "rules"
        assert body["contents"][0]["parts"][0]["text"] == "question"

    def test_no_system_instruction_key_when_absent(self):
        g, _, _ = _generator()
        assert "systemInstruction" not in g.build_body("q")

    def test_api_key_goes_in_the_header_not_the_url(self):
        g, seen, _ = _generator()
        g.generate("q")
        assert seen[0].headers["x-goog-api-key"] == "test-key"
        assert "test-key" not in str(seen[0].url)


class TestGenerationResult:
    def test_thinking_tokens_are_reported_separately(self):
        g, _, _ = _generator(
            [
                httpx.Response(
                    200,
                    json={
                        **OK_BODY,
                        "usageMetadata": {
                            "promptTokenCount": 10,
                            "candidatesTokenCount": 5,
                            "thoughtsTokenCount": 105,
                        },
                    },
                )
            ]
        )
        r = g.generate("q")
        assert (r.prompt_tokens, r.output_tokens, r.thinking_tokens) == (10, 5, 105)
        assert r.total_tokens == 120

    def test_parse_json_returns_the_payload(self):
        g, _, _ = _generator()
        assert g.generate("q").parse_json() == {"ok": True}

    def test_parse_json_error_names_finish_reason_and_length(self):
        """A truncated response is the realistic failure; the error must say so."""
        g, _, _ = _generator(
            [
                httpx.Response(
                    200,
                    json={
                        "candidates": [
                            {
                                "content": {"parts": [{"text": '{"a": '}]},
                                "finishReason": "MAX_TOKENS",
                            }
                        ]
                    },
                )
            ]
        )
        with pytest.raises(gen.GenerationError, match="MAX_TOKENS"):
            g.generate("q").parse_json()

    def test_missing_usage_metadata_defaults_to_zero(self):
        g, _, _ = _generator([httpx.Response(200, json={"candidates": OK_BODY["candidates"]})])
        assert g.generate("q").total_tokens == 0


class TestConstruction:
    def test_unknown_generator_is_rejected(self):
        with pytest.raises(ValueError, match="unknown generator"):
            gen.GeminiGenerator("gpt-4")

    def test_missing_key_names_where_to_put_one(self):
        with pytest.raises(gen.GenerationError, match="aistudio.google.com"):
            gen.GeminiGenerator(api_key=None, env_path="/nonexistent/.env.gemini")

    def test_make_generator_dispatches_on_provider(self):
        g = gen.make_generator(api_key="k", env_path=None)
        assert isinstance(g, gen.GeminiGenerator)
        assert g.name == gen.DEFAULT_GENERATOR

    def test_make_generator_rejects_unknown_spec(self):
        with pytest.raises(ValueError, match="unknown generator"):
            gen.make_generator("claude-opus-5")

    def test_registry_does_not_invent_prices(self):
        """A cost column computed from a guessed rate is worse than none. Phase 4's
        audit found four such literals in a report generator."""
        assert all(spec["price_per_mtok"] is None for spec in gen.GENERATORS.values())


DAILY_429_BODY = json.dumps(
    {
        "error": {
            "code": 429,
            "message": (
                "You exceeded your current quota. * Quota exceeded for metric: "
                "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
                "limit: 20, model: gemini-3.6-flash\nPlease retry in 4.411926172s."
            ),
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {
                            "quotaMetric": (
                                "generativelanguage.googleapis.com/"
                                "generate_content_free_tier_requests"
                            ),
                            "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                            "quotaValue": "20",
                        }
                    ],
                }
            ],
        }
    }
)


class TestDailyQuotaIsNotRetried:
    """The distinction that cost this project a day of quota.

    A per-day 429 carries the same "Please retry in 4.4s" text as any other, so
    treating the message as authoritative sends six attempts into an exhausted
    budget. The structured `quotaId` is what actually says `PerDay`.
    """

    def test_detects_a_per_day_quota_from_the_structured_field(self):
        assert gen._is_daily_quota_error(DAILY_429_BODY)

    def test_does_not_flag_an_ordinary_rate_limit(self):
        assert not gen._is_daily_quota_error(REAL_429_BODY)

    def test_survives_a_non_json_body(self):
        assert not gen._is_daily_quota_error("<html>429</html>")

    def test_substring_fallback_when_details_are_absent(self):
        body = json.dumps({"error": {"message": "quota GenerateRequestsPerDay... exceeded"}})
        assert gen._is_daily_quota_error(body)

    def test_raises_immediately_without_burning_retries(self):
        g, seen, slept = _generator([httpx.Response(429, text=DAILY_429_BODY)] * 6)
        with pytest.raises(gen.GenerationError, match="daily free-tier quota exhausted"):
            g.generate("q")
        assert len(seen) == 1, "a per-day 429 must cost exactly one request"
        assert slept == []

    def test_error_names_the_two_real_options(self):
        """Neither of which is waiting a few seconds."""
        g, _, _ = _generator([httpx.Response(429, text=DAILY_429_BODY)])
        with pytest.raises(gen.GenerationError) as exc:
            g.generate("q")
        assert "per model" in str(exc.value)
        assert "20 requests/day/model" in str(exc.value)

    def test_transient_429_still_retries(self):
        """The fix must not turn every rate limit into a hard failure."""
        g, seen, _ = _generator(
            [httpx.Response(429, text=REAL_429_BODY), httpx.Response(200, json=OK_BODY)]
        )
        assert g.generate("q").attempts == 2
        assert len(seen) == 2


class TestRegistryCoversIndependentQuotas:
    def test_multiple_models_are_registered(self):
        """Quota is per model, so a second model is a second budget -- the only way
        a by-generator comparison is affordable on a free tier."""
        assert len(gen.GENERATORS) >= 3

    def test_every_registered_model_is_a_gemini_flash_variant(self):
        assert all(spec["provider"] == "gemini" for spec in gen.GENERATORS.values())
        assert all("flash" in spec["model"] for spec in gen.GENERATORS.values())

    def test_every_model_defaults_to_minimal_thinking(self):
        assert all(spec["thinking_level"] == "minimal" for spec in gen.GENERATORS.values())


PER_MINUTE_429_BODY = json.dumps(
    {
        "error": {
            "code": 429,
            "message": "Quota exceeded. Please retry in 4.4s.",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}
                    ],
                },
                {
                    # Google attaches a help link whose text mentions the *per-day*
                    # quota id even for a per-minute violation. An unconditional
                    # substring scan read that as a daily exhaustion.
                    "@type": "type.googleapis.com/google.rpc.Help",
                    "links": [
                        {
                            "description": "see GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                            "url": "https://ai.google.dev/gemini-api/docs/rate-limits",
                        }
                    ],
                },
            ],
        }
    }
)


class TestPerMinuteIsNotMistakenForPerDay:
    def test_structured_per_minute_violation_is_not_daily(self):
        assert not gen._is_daily_quota_error(PER_MINUTE_429_BODY)

    def test_per_minute_429_is_retried_not_aborted(self):
        """The regression: a transient rate limit must not raise "daily quota
        exhausted" and abandon the query."""
        g, seen, _ = _generator(
            [httpx.Response(429, text=PER_MINUTE_429_BODY), httpx.Response(200, json=OK_BODY)]
        )
        assert g.generate("q").attempts == 2
        assert len(seen) == 2

    def test_substring_fallback_still_works_without_structured_violations(self):
        body = json.dumps({"error": {"message": "quota GenerateRequestsPerDay exceeded"}})
        assert gen._is_daily_quota_error(body)

    def test_non_dict_violations_do_not_raise(self):
        """A list of strings where objects were expected used to raise AttributeError
        out of the retry path."""
        body = json.dumps(
            {
                "error": {
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                            "violations": ["PerMinute", "whatever"],
                        }
                    ]
                }
            }
        )
        assert gen._is_daily_quota_error(body) is False


class TestHostileDelaysNeverReachSleep:
    """`time.sleep` rejects negative and NaN. Both escaped as raw ValueError, and
    `run_gen_eval`'s blanket handler then recorded a retryable 503 as a lost query."""

    def test_negative_retry_after_is_clamped(self):
        assert gen._backoff_seconds(0, "-3", "") >= 0.0

    def test_nan_retry_after_is_replaced(self):
        d = gen._backoff_seconds(0, "nan", "")
        assert d == d and d >= 0.0

    def test_infinite_retry_after_is_capped(self):
        assert gen._backoff_seconds(0, "inf", "") <= gen.MAX_BACKOFF_S

    def test_negative_body_delay_is_clamped(self):
        body = json.dumps({"error": {"message": "Please retry in 0.0s.", "details": []}})
        assert gen._backoff_seconds(0, None, body) >= 0.0

    def test_every_backoff_is_sleepable(self):
        import math
        import time

        bodies = ["", REAL_429_BODY, "not json", json.dumps({"error": {}})]
        for attempt in range(6):
            for hdr in (None, "-1", "nan", "inf", "abc", "5"):
                for body in bodies:
                    d = gen._backoff_seconds(attempt, hdr, body)
                    assert isinstance(d, float) and not math.isnan(d)
                    assert 0.0 <= d <= gen.MAX_BACKOFF_S
        time.sleep(0)  # the contract we are protecting

    def test_a_503_with_a_hostile_header_still_recovers(self):
        g, seen, _ = _generator(
            [
                httpx.Response(503, text="busy", headers={"retry-after": "-5"}),
                httpx.Response(200, json=OK_BODY),
            ]
        )
        assert g.generate("q").attempts == 2


class TestKeyResolution:
    """Precedence matters once there are two places a key can come from: a container
    passes an environment variable and has no dotfile to mount."""

    def test_environment_is_used_when_there_is_no_dotfile(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        made = gen.GeminiGenerator(env_path=str(tmp_path / "missing.env"))
        assert made._key == "from-env"

    def test_an_explicit_argument_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        assert gen.GeminiGenerator(api_key="explicit")._key == "explicit"

    def test_the_environment_beats_the_dotfile(self, monkeypatch, tmp_path):
        env_file = tmp_path / ".env.gemini"
        env_file.write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        assert gen.GeminiGenerator(env_path=str(env_file))._key == "from-env"

    def test_the_dotfile_is_used_when_the_environment_is_empty(self, monkeypatch, tmp_path):
        env_file = tmp_path / ".env.gemini"
        env_file.write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert gen.GeminiGenerator(env_path=str(env_file))._key == "from-file"

    def test_no_key_anywhere_names_both_places(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(gen.GenerationError, match="GEMINI_API_KEY"):
            gen.GeminiGenerator(env_path=str(tmp_path / "missing.env"))

    def test_an_empty_environment_variable_does_not_count_as_a_key(self, monkeypatch, tmp_path):
        """An exported-but-blank variable is the classic way a container looks configured
        and is not. It must fall through, not become the key."""
        env_file = tmp_path / ".env.gemini"
        env_file.write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
        monkeypatch.setenv("GEMINI_API_KEY", "")
        assert gen.GeminiGenerator(env_path=str(env_file))._key == "from-file"
