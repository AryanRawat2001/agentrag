"""Generator interface and the Gemini implementation.

Same shape as `retrieval.Retriever`: every generator answers the same question --
given a prompt and an optional response schema, return text plus token accounting --
so the answering layer and the eval harness never learn which vendor produced an
answer. That is what makes "citation-verification failure rate by generator" a row
in a table instead of a rewrite (plan deviation #14).

## Why Gemini, and what it cost us

The plan's locked decision was Claude, for native citations returning character
offsets. Organisation policy blocks first-party access, and the available Bedrock
role permits only models AWS has already retired -- 21 Claude model IDs probed, and
every model the policy allows returns 404 while every model that exists returns 403.
There is no intersection, so the project cannot depend on it.

Gemini 3.6 Flash on the free tier is the substitute. It has no citation feature, so
`citations.py` locates quoted spans by search instead of trusting returned offsets.

## Measured request-shaping facts (2026-08-17, probed not assumed)

- **`gemini-2.5-flash` is retired for new keys.** It returns 404 with a message
  naming `gemini-3.6-flash` as the replacement. Same for `gemini-2.5-flash-lite`.
  The retired ID is still listed by `GET /models`, so the model list is not a
  reliable availability check -- only a real request is.
- **Thinking is on by default, and `thinkingBudget: 0` does not disable it** -- that
  returns 400 INVALID_ARGUMENT. `thinkingLevel: "minimal"` is the only setting that
  reaches zero thinking tokens. Both matter: the default spent **105 thinking tokens
  to emit the single token `OK`**, which across a few hundred eval queries is the
  dominant cost. This is structurally the same trap as Opus 5 in `contextual.py`,
  where omitting `thinking` does not turn it off.
- `thinkingLevel` accepts `minimal` / `low` / `high`; `none` and `off` are 400s.
  `low` and `high` both spent ~80 tokens on a trivial prompt, so the level caps
  thinking rather than targeting it.
- **Structured output works and is compatible with our citation approach.** Claude's
  native `citations` returns 400 alongside `output_config.format` (deviation #3),
  which is precisely why that design had to assemble its JSON envelope in
  application code. Because we now carry quotes in the response body rather than in
  a vendor citation field, `responseSchema` constrains the whole envelope -- so a
  malformed or truncated answer is rejected by the API rather than misparsed here.
- **The free tier allows 20 requests per DAY, per model, per project** -- not per
  minute, which is what the 429's own "Please retry in 4.4s" text implies. The
  authoritative field is `quotaId=GenerateRequestsPerDayPerProjectPerModel-FreeTier`.
  Quota being per *model* is the one piece of good news: each registered model is an
  independent 20-request budget.
- 503 "high demand" is common and transient -- it hit 2 of ~10 probe calls. It is
  not a rejection, and an earlier probe run misread two 503s as invalid parameters.
  Retrying through them is required to tell a bad request from a busy model.

## Cost

Deliberately not priced. The free tier bills nothing, and I do not have confirmed
paid per-token rates for these models -- writing plausible numbers into a cost table
is exactly the defect Phase 4's audit found four times. `GenerationResult` therefore
reports measured token counts, and `price_per_mtok` stays None until real rates are
looked up. A cost figure computed from a guessed rate is worse than no cost figure.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from ragpipe.contextual import load_env_file  # shared dotenv reader; see note below

# `load_env_file` lives in contextual.py because that is where it was first needed.
# If a third consumer appears it should move to its own module; importing it here
# beats a second hand-rolled parser that can drift from the tested one.

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

#: Known-good generators. `context_tokens` is what `GET /models` reports.
#: `thinking_level` is the default for this project: "minimal", because thinking is
#: on by default and cannot be zeroed any other way (see module docstring).
GENERATORS: dict[str, dict[str, Any]] = {
    "gemini-3.6-flash": {
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "context_tokens": 1_048_576,
        "max_output_tokens": 65_536,
        "thinking_level": "minimal",
        "price_per_mtok": None,  # free tier; paid rates not confirmed, so not guessed
    },
    "gemini-3-flash-preview": {
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
        "context_tokens": 1_048_576,
        "max_output_tokens": 65_536,
        "thinking_level": "minimal",
        "price_per_mtok": None,
    },
    # Lite variants. Present because free-tier quota is per model, so these are
    # independent 20-request budgets rather than alternatives competing for one --
    # which is what makes a by-generator comparison affordable at all here.
    "gemini-3.5-flash-lite": {
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "context_tokens": 1_048_576,
        "max_output_tokens": 65_536,
        "thinking_level": "minimal",
        "price_per_mtok": None,
    },
    "gemini-3.1-flash-lite": {
        "provider": "gemini",
        "model": "gemini-3.1-flash-lite",
        "context_tokens": 1_048_576,
        "max_output_tokens": 65_536,
        "thinking_level": "minimal",
        "price_per_mtok": None,
    },
}
DEFAULT_GENERATOR = "gemini-3.6-flash"

#: Retried rather than raised. 429 is rate limiting, 5xx is transient capacity.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 6
BASE_BACKOFF_S = 1.0
MAX_BACKOFF_S = 32.0
RETRY_FLOOR_S = 0.5  # added to any server-supplied delay; window edges are not exact

# Free-tier request cap, read off the 429 rather than guessed:
#
#   quotaId = GenerateRequestsPerDayPerProjectPerModel-FreeTier
#   metric  = generativelanguage.googleapis.com/generate_content_free_tier_requests
#   value   = 20
#
# **Twenty requests per DAY, per model, per project.** Not per minute. The error text
# says "Please retry in 4.411926172s.", which reads exactly like a rate limit and is
# what an earlier version of this comment concluded -- wrongly. The structured
# `quotaId` is the authoritative field, and it says `PerDay`.
#
# That distinction is the difference between a stall and a wall. The first Phase 5 eval
# run made 80 queries with up to 6 retries each; those ~480 attempts consumed the
# entire daily budget for `gemini-3.6-flash` in one pass, and it scored 3 of 80. A
# rejected request still spends a quota unit, so retrying into an exhausted quota is
# pure loss -- and against a daily cap the loss does not recover until midnight
# Pacific.
#
# Two consequences, both implemented below:
#
# 1. A per-day 429 is **not retried at all** (`_is_daily_quota_error`). No wait
#    available to this process can clear it, so the six attempts that a transient 429
#    deserves would just burn five more units of a budget already at zero.
# 2. Requests are still paced. The free tier also imposes an unpublished
#    per-minute cap, and pacing costs nothing when the daily budget only permits 20
#    requests anyway.
#
# The practical consequence for this project is a hard ceiling on eval size: a
# stratified 4-per-slice sample is 16 queries, which is what fits. Quota is per
# *model*, so the "citation precision by generator" comparison is affordable in a way
# that a larger single-model sample is not.
FREE_TIER_REQUESTS_PER_DAY = 20
MIN_REQUEST_INTERVAL_S = 3.21

# Google returns the retry delay in the response *body* as a `RetryInfo` detail and
# in the message text -- not in a `Retry-After` header. Reading only the header (the
# first implementation here) found nothing and fell back to blind exponential waits.
_RETRY_DELAY_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)


class GenerationError(RuntimeError):
    """A request that failed in a way retrying will not fix, or ran out of retries.

    `daily_quota` marks the one failure that no caller can retry its way out of. It is a
    flag rather than something downstream re-derives from the message, because the obvious
    re-derivation -- scanning the text for "quota" -- also matches a *per-minute* 429,
    whose body reads "You exceeded your current quota". Those two need opposite advice:
    retry in a moment, versus stop and come back tomorrow. `_is_daily_quota_error` already
    makes the distinction correctly off the structured `quotaId`; this carries that
    decision instead of inviting every consumer to guess at it again.
    """

    def __init__(self, *args: object, daily_quota: bool = False) -> None:
        super().__init__(*args)
        self.daily_quota = daily_quota


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """One generation, with the accounting the eval report needs.

    `thinking_tokens` is separate from `output_tokens` on purpose: it is billed,
    invisible in the response text, and the single biggest lever on spend here. A
    result that folded it into output would hide the 105-tokens-to-say-OK problem.
    """

    text: str
    model: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 1
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens + self.thinking_tokens

    def parse_json(self) -> Any:
        """Parse the response as JSON, with the raw text in the error on failure.

        Used when a schema was supplied. The schema makes malformed JSON an API-side
        rejection rather than a silent misparse, but a truncated response can still
        arrive if the output cap is hit -- hence `finish_reason` on this class.
        """
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as exc:
            raise GenerationError(
                f"{self.model}: response was not valid JSON "
                f"(finish_reason={self.finish_reason!r}, {len(self.text)} chars): "
                f"{self.text[:200]!r}"
            ) from exc


class Generator(Protocol):
    """Anything that can turn a prompt into text."""

    name: str

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult: ...


def _is_daily_quota_error(body: str) -> bool:
    """True when a 429 names a per-day quota, which retrying cannot clear.

    Matches on the structured `quotaId` (`...PerDay...`) rather than the message
    text, because the text advertises a small "Please retry in Ns" delay even when the
    exhausted budget is daily -- following that advice is what burned this project's
    quota. Falls back to a substring scan so a body without QuotaFailure details is
    still classified rather than silently treated as transient.
    """
    try:
        details = (json.loads(body).get("error") or {}).get("details") or []
    except (json.JSONDecodeError, AttributeError):
        return "PerDay" in body

    saw_quota_failure = False
    for detail in details:
        if not isinstance(detail, dict):
            continue
        violations = detail.get("violations") or []
        if violations:
            saw_quota_failure = True
        for violation in violations:
            # Violations are not guaranteed to be objects. A list of strings used to
            # reach `.get` and raise AttributeError out of a retry path.
            if not isinstance(violation, dict):
                continue
            if "PerDay" in str(violation.get("quotaId", "")):
                return True

    # Only fall back to a substring scan when the response carried no structured
    # violations at all. Scanning unconditionally misread a *per-minute* 429 as a
    # per-day one whenever the word appeared elsewhere in the body -- Google's
    # `google.rpc.Help` link text mentions the per-day quota id -- which turned a
    # retryable rate limit into an immediate hard failure.
    if saw_quota_failure:
        return False
    return "PerDay" in body


def _clamp_delay(seconds: float) -> float:
    """Bound a delay into something `time.sleep` will accept.

    A server is not a trusted source of arithmetic. `Retry-After: -3` and
    `Retry-After: nan` both reached `time.sleep` and raised `ValueError` out of the
    retry loop, where `run_gen_eval`'s blanket handler turned a retryable 503 into a
    permanently lost query excluded from every denominator. NaN fails every
    comparison, so it is tested for explicitly rather than via min/max.
    """
    if seconds != seconds:  # NaN
        return BASE_BACKOFF_S
    return max(0.0, min(float(seconds), MAX_BACKOFF_S))


def parse_retry_delay(body: str) -> float | None:
    """Pull the server's requested retry delay out of an error body.

    Checks the structured `RetryInfo` detail first and the human-readable message
    second, because the two do not always both appear. Returns None when neither is
    present, which is the signal to fall back to exponential backoff.
    """
    try:
        details = (json.loads(body).get("error") or {}).get("details") or []
    except (json.JSONDecodeError, AttributeError):
        details = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        if detail.get("@type", "").endswith("RetryInfo"):
            raw = str(detail.get("retryDelay", ""))
            try:
                return float(raw.rstrip("s"))
            except ValueError:
                pass
    match = _RETRY_DELAY_RE.search(body)
    return float(match.group(1)) if match else None


def _backoff_seconds(attempt: int, retry_after: str | None, body: str = "") -> float:
    """How long to wait before retrying.

    Preference order: the server's own `Retry-After` header, then the retry delay it
    embeds in the error body, then exponential backoff with jitter. Honouring the
    server's number matters most for 429s -- it knows when the quota window rolls
    over and guessing longer just wastes wall time.

    Jitter matters even single-threaded: without it, a run that hits a capacity spike
    retries on the same cadence every time and tends to land in the same spikes. A
    small floor is added to a server-specified delay because the window boundary is
    not instantaneous, and arriving a few milliseconds early spends a quota unit for
    a guaranteed rejection.
    """
    server: float | None = None
    if retry_after:
        try:
            server = float(retry_after)
        except ValueError:
            server = None
    if server is None:
        server = parse_retry_delay(body)

    if server is not None:
        # The floor applies to *both* server-supplied paths. It was previously added
        # only to the body-derived one, so a `Retry-After` header returned the raw
        # value -- an inconsistency a test had pinned rather than caught.
        return _clamp_delay(server + RETRY_FLOOR_S)
    return _clamp_delay(BASE_BACKOFF_S * (2**attempt) * (0.5 + random.random()))


class GeminiGenerator:
    """Gemini via the REST API over `httpx`.

    Raw HTTP rather than `google-genai` deliberately: the project already depends on
    `httpx` (see `net.py`), the surface used here is one endpoint, and it keeps the
    dependency footprint of a portfolio repo honest. The tradeoff is that request
    shaping is hand-written, which is why the shapes above were probed rather than
    recalled.
    """

    def __init__(
        self,
        spec: str = DEFAULT_GENERATOR,
        *,
        api_key: str | None = None,
        env_path: str | None = ".env.gemini",
        thinking_level: str | None = None,
        client: httpx.Client | None = None,
        max_attempts: int = MAX_ATTEMPTS,
        sleep: Any = time.sleep,
        min_interval_s: float = MIN_REQUEST_INTERVAL_S,
        clock: Any = time.monotonic,
    ) -> None:
        if spec not in GENERATORS:
            raise ValueError(f"unknown generator {spec!r}; known: {sorted(GENERATORS)}")
        self.name = spec
        self.spec = GENERATORS[spec]
        self.model = self.spec["model"]
        self.thinking_level = thinking_level or self.spec["thinking_level"]
        self.max_attempts = max_attempts
        self._sleep = sleep
        self._clock = clock
        self.min_interval_s = min_interval_s
        self._last_request_at: float | None = None
        self._client = client
        self._owns_client = client is None

        # Explicit argument, then the environment, then the dotfile. The environment
        # step exists for the container, which has no `.env.gemini` to mount and passes
        # secrets as variables; it precedes the file because that is the precedence every
        # SDK and 12-factor deployment uses, so an operator overriding a baked-in file
        # gets what they asked for. Nothing here logs or echoes the key.
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key and env_path:
            key = (load_env_file(env_path) or {}).get("GEMINI_API_KEY")
        if not key:
            raise GenerationError(
                "no Gemini API key. Set GEMINI_API_KEY in the environment or in "
                ".env.gemini (get one at https://aistudio.google.com/apikey), or pass "
                "api_key=."
            )
        self._key = key

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> GeminiGenerator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(120.0))
        return self._client

    def _pace(self) -> None:
        """Block until `min_interval_s` has passed since the last request.

        Applied to retries as well as first attempts, because a retry spends a quota
        unit exactly like any other request -- that equivalence is what made the
        first eval run fail. Set `min_interval_s=0` to disable (paid tiers, or tests).
        """
        if self.min_interval_s <= 0:
            return
        now = self._clock()
        if self._last_request_at is not None:
            wait = self.min_interval_s - (now - self._last_request_at)
            if wait > 0 and wait == wait:  # NaN-safe
                self._sleep(wait)
                now = self._clock()
        self._last_request_at = now

    def build_body(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """The request body. Separated from the call so tests can assert its shape
        without a network round trip -- the shaping is where the vendor-specific
        traps live."""
        gen_config: dict[str, Any] = {"thinkingConfig": {"thinkingLevel": self.thinking_level}}
        if schema is not None:
            # Constrains the whole envelope, so a short or invented response shape is
            # rejected by the API rather than misassigned downstream.
            gen_config["responseMimeType"] = "application/json"
            gen_config["responseSchema"] = schema
        if max_output_tokens is not None:
            gen_config["maxOutputTokens"] = max_output_tokens
        if temperature is not None:
            gen_config["temperature"] = temperature

        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen_config,
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        body = self.build_body(
            prompt,
            system=system,
            schema=schema,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        url = f"{GEMINI_BASE}/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self._key, "content-type": "application/json"}

        started = time.monotonic()
        last: str = ""
        for attempt in range(self.max_attempts):
            self._pace()
            try:
                response = self.client.post(url, headers=headers, json=body)
            except httpx.HTTPError as exc:
                last = f"transport error: {type(exc).__name__}: {exc}"
                if attempt == self.max_attempts - 1:
                    break
                self._sleep(_backoff_seconds(attempt, None))
                continue

            if response.status_code in RETRY_STATUS:
                last = f"http {response.status_code}: {response.text[:200]}"
                if response.status_code == 429 and _is_daily_quota_error(response.text):
                    # Nothing this process can wait for will clear a daily quota, and
                    # every further attempt spends a unit of a budget already at zero.
                    raise GenerationError(
                        f"{self.model}: daily free-tier quota exhausted "
                        f"({FREE_TIER_REQUESTS_PER_DAY} requests/day/model). Not retried — "
                        f"no backoff can clear a per-day cap. Use a different model "
                        f"(quota is per model) or wait for the reset. "
                        f"Body: {response.text[:200]}",
                        daily_quota=True,
                    )
                if attempt == self.max_attempts - 1:
                    break
                self._sleep(
                    _backoff_seconds(attempt, response.headers.get("retry-after"), response.text)
                )
                continue

            if response.status_code != 200:
                # A 4xx that is not 429 is a bad request. Retrying is pointless and
                # would bury the actual error under a backoff loop.
                raise GenerationError(
                    f"{self.model}: http {response.status_code}: {response.text[:400]}"
                )

            return self._parse(response.json(), time.monotonic() - started, attempt + 1)

        raise GenerationError(
            f"{self.model}: gave up after {self.max_attempts} attempts. Last: {last}"
        )

    def _parse(self, payload: dict[str, Any], latency_s: float, attempts: int) -> GenerationResult:
        candidates = payload.get("candidates") or []
        if not candidates:
            # Usually a safety block or a prompt the model declined to continue.
            # Surfaced rather than returned as empty text, which would silently
            # score as an unhelpful-but-valid answer.
            raise GenerationError(
                f"{self.model}: no candidates returned "
                f"(promptFeedback={payload.get('promptFeedback')})"
            )
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)

        usage = payload.get("usageMetadata") or {}
        return GenerationResult(
            text=text,
            model=self.model,
            prompt_tokens=int(usage.get("promptTokenCount") or 0),
            output_tokens=int(usage.get("candidatesTokenCount") or 0),
            thinking_tokens=int(usage.get("thoughtsTokenCount") or 0),
            latency_s=latency_s,
            attempts=attempts,
            finish_reason=str(candidate.get("finishReason") or ""),
            raw=payload,
        )


def make_generator(spec: str = DEFAULT_GENERATOR, **kwargs: Any) -> Generator:
    """Build a generator from its registry name."""
    if spec not in GENERATORS:
        raise ValueError(f"unknown generator {spec!r}; known: {sorted(GENERATORS)}")
    provider = GENERATORS[spec]["provider"]
    if provider == "gemini":
        return GeminiGenerator(spec, **kwargs)
    raise ValueError(f"no implementation for provider {provider!r}")
