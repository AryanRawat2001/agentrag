"""Polite, resumable, integrity-checked HTTP.

Three things this module is careful about, because we are hitting public
government infrastructure with no API key and no rate-limit contract:

  1. Per-host throttling. Requests to the same host are spaced by a minimum
     interval regardless of how fast the caller's loop runs.
  2. Backoff that respects the server. 429 and 5xx are retried with exponential
     backoff; an explicit Retry-After header (seconds *or* HTTP-date form) wins
     over our own schedule. This applies to file downloads too, not just index
     calls — the downloads are the overwhelming majority of the traffic.
  3. Content integrity. A download whose sha256 does not match the pinned value
     fails loudly rather than silently replacing the corpus.

Set RAGPIPE_USER_AGENT to something with a real contact address before running
this at any volume.
"""

from __future__ import annotations

import email.utils
import hashlib
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
MAX_RETRY_AFTER = 60.0  # cap on how long we honour a server-supplied delay

# ClinicalTrials.gov sits behind a WAF that rejects requests whose User-Agent
# does not begin with the true client token. Verified empirically:
#
#   python-httpx/0.28.1                                    -> 200
#   python-httpx/0.28.1 ragpipe/0.1 (contact@example.com)  -> 200
#   ragpipe/0.1 (contact@example.com)                      -> 403
#   curl/8.7.1            (sent from httpx)                -> 403
#   Mozilla/5.0                                            -> 403
#   python-requests/2.32                                   -> 403
#
# Any UA that misrepresents the client is treated as spoofing, so we *prefix*
# the honest token and append our own identifier rather than replacing it. That
# satisfies the WAF and still tells the operator who we are and how to reach us.
# Replacing this prefix will produce 403s that look like an outage.
_HTTPX_TOKEN = f"python-httpx/{httpx.__version__}"
_FALLBACK_SUFFIX = "ragpipe/0.1 (set RAGPIPE_USER_AGENT with a contact address)"


class HashMismatch(RuntimeError):
    """A downloaded file's sha256 did not match the value pinned in the manifest."""

    def __init__(self, url: str, expected: str, actual: str) -> None:
        super().__init__(
            f"sha256 mismatch for {url}\n  pinned:   {expected}\n  received: {actual}\n"
            "The remote document changed since the manifest was pinned. Re-pin "
            "deliberately with `ragpipe fetch --allow-drift` if that is expected."
        )
        self.url = url
        self.expected = expected
        self.actual = actual


def build_user_agent(suffix: str | None = None) -> str:
    """Compose a WAF-acceptable User-Agent: honest client token + our identifier."""
    suffix = suffix or os.environ.get("RAGPIPE_USER_AGENT") or _FALLBACK_SUFFIX
    if suffix.startswith(_HTTPX_TOKEN):
        return suffix
    return f"{_HTTPX_TOKEN} {suffix}"


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header. RFC 9110 permits seconds *or* an HTTP-date.

    Rejecting the date form means hammering a server that explicitly asked for a
    long wait, so both are handled. Returns None when absent or unparseable.
    """
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return min(float(value), MAX_RETRY_AFTER)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    delay = (when - datetime.now(UTC)).total_seconds()
    return min(max(delay, 0.0), MAX_RETRY_AFTER)


class PoliteClient:
    """httpx wrapper with per-host throttling, retry/backoff, and hash checks."""

    def __init__(
        self,
        min_interval: float = 0.4,
        timeout: float = 60.0,
        max_retries: int = 4,
        user_agent: str | None = None,
    ) -> None:
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last_request: dict[str, float] = {}
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": build_user_agent(user_agent)},
        )

    def __enter__(self) -> PoliteClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _wait_turn(self, url: str) -> None:
        host = httpx.URL(url).host or ""
        elapsed = time.monotonic() - self._last_request.get(host, 0.0)
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request[host] = time.monotonic()

    def _sleep_before_retry(self, attempt: int, retry_after: float | None) -> None:
        """Wait before the next attempt. Only called when a retry will actually happen."""
        time.sleep(retry_after if retry_after is not None else min(2.0**attempt, 30.0))

    def get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """GET with throttling and retries. Raises on final failure."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self._wait_turn(url)
            retry_after: float | None = None
            try:
                resp = self._client.get(url, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
            else:
                if resp.status_code not in RETRY_STATUS:
                    resp.raise_for_status()
                    return resp
                last_error = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
                retry_after = parse_retry_after(resp.headers.get("Retry-After"))

            # Guard the sleep on there being a further attempt, so a permanently
            # failing URL waits max_retries times, not max_retries + 1.
            if attempt < self.max_retries:
                self._sleep_before_retry(attempt, retry_after)

        raise RuntimeError(
            f"GET failed after {self.max_retries + 1} attempts: {url}"
        ) from last_error

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return self.get(url, params=params).json()

    def download(
        self,
        url: str,
        dest: Path,
        pinned_sha256: str | None = None,
        skip_if_present: bool = True,
    ) -> tuple[str, int]:
        """Download to `dest`, returning (sha256, byte_count).

        `pinned_sha256` does double duty: it lets an already-correct file short
        circuit the request, and it is *verified* against what we actually
        receive. A mismatch raises HashMismatch rather than overwriting the file,
        so a changed remote document is a loud failure instead of silent corpus
        drift.

        Pass skip_if_present=False to force a real request (see `fetch --refetch`).
        """
        if skip_if_present and dest.exists() and pinned_sha256:
            existing = sha256_file(dest)
            if existing == pinned_sha256:
                return existing, dest.stat().st_size

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self._wait_turn(url)
            retry_after: float | None = None
            digest = hashlib.sha256()
            size = 0
            try:
                with self._client.stream("GET", url) as resp:
                    if resp.status_code in RETRY_STATUS:
                        retry_after = parse_retry_after(resp.headers.get("Retry-After"))
                        last_error = httpx.HTTPStatusError(
                            f"HTTP {resp.status_code}", request=resp.request, response=resp
                        )
                        resp.close()
                    else:
                        resp.raise_for_status()
                        with tmp.open("wb") as fh:
                            for chunk in resp.iter_bytes(chunk_size=65536):
                                digest.update(chunk)
                                fh.write(chunk)
                                size += len(chunk)

                        actual = digest.hexdigest()
                        if pinned_sha256 and actual != pinned_sha256:
                            # Do not promote a mismatched file into place.
                            tmp.unlink(missing_ok=True)
                            raise HashMismatch(url, pinned_sha256, actual)

                        # Rename last, so an interrupted download never leaves a
                        # truncated file where a later run would trust it.
                        tmp.replace(dest)
                        return actual, size
            except HashMismatch:
                raise  # a content mismatch is not a transient error; do not retry
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
            finally:
                # Never leave a partial file behind for the next run to trip over.
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

            if attempt < self.max_retries:
                self._sleep_before_retry(attempt, retry_after)

        raise RuntimeError(
            f"download failed after {self.max_retries + 1} attempts: {url}"
        ) from last_error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
