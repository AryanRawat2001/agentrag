"""Execute the dashboard's diff algorithm, rather than asserting that it exists.

`renderDiff` is the one piece of real logic on the page: an LCS over whitespace-split
tokens, deciding what a reader sees when a model's quote and the document disagree. Every
other dashboard test is structural — it checks the wiring is present — which cannot tell a
working diff from one that highlights the whole pane.

So the functions are lifted out of the shipped HTML string and run under node. Skipped
when node is absent, because it is a development convenience and not a dependency of the
service: nothing in `ragpipe` needs a JS runtime, and adding one to the container to test
a string would be the wrong trade.

The cases are the ones this corpus actually produces, and the ones the project has been
burned by: a doubled space left by stripping a PDF line number (a real match that must not
look like a failure), and the single-token meaning changes — `40 CFR` for `21 CFR`,
`60 days` for `30 days`, `may` for `shall` — that Phase 6 found lexical coverage scoring
1.000 and accepting.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from ragpipe.dashboard import DASHBOARD_HTML

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs node to execute the page's JS")

#: Lifted verbatim from the shipped page, so this cannot drift from what users run.
_START = "function diffTokens"
_END = "/* ------"


def _js_functions() -> str:
    start = DASHBOARD_HTML.index(_START)
    end = DASHBOARD_HTML.index(_END, start)
    return DASHBOARD_HTML[start:end]


def _run(quote: str, source: str) -> dict:
    harness = (
        """
const esc = (s) => (s == null ? "" : String(s).replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])));
"""
        + _js_functions()
        + """
// Via the environment, not argv: `node -e` shifts argv, so the input landed at a
// different index than expected and JSON.parse got "undefined".
const [q, s] = JSON.parse(process.env.DIFF_INPUT);
const model = renderDiff(q, s, "model");
const source = renderDiff(q, s, "source");
console.log(JSON.stringify({
  clean: diffIsClean(q, s),
  model: model,
  source: source,
  dels: (model.match(/<del>/g) || []).length,
  inses: (source.match(/<ins>/g) || []).length,
}));
"""
    )
    proc = subprocess.run(
        [NODE, "-e", harness],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "DIFF_INPUT": json.dumps([quote, source])},
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


class TestDiffMarksOnlyWhatDiffers:
    def test_identical_text_produces_no_marks(self):
        text = "The sponsor shall submit the annual report within 60 days."
        out = _run(text, text)
        assert out["clean"] is True
        assert out["dels"] == 0 and out["inses"] == 0

    @pytest.mark.parametrize(
        ("quote", "source", "expect_in_model", "expect_in_source"),
        [
            # The meaning changes. Each must be isolated to the one token that moved --
            # a diff that marks the whole sentence is as useless as no diff.
            (
                "as required under 40 CFR 314.50 the sponsor shall report",
                "as required under 21 CFR 314.50 the sponsor shall report",
                "40",
                "21",
            ),
            (
                "submit the annual report within 60 days",
                "submit the annual report within 30 days",
                "60",
                "30",
            ),
            (
                "the sponsor shall submit a report",
                "the sponsor may submit a report",
                "shall",
                "may",
            ),
        ],
    )
    def test_a_single_token_change_is_isolated(
        self, quote, source, expect_in_model, expect_in_source
    ):
        out = _run(quote, source)
        assert out["clean"] is False
        assert out["dels"] == 1, out["model"]
        assert out["inses"] == 1, out["source"]
        assert "<del>" + expect_in_model + "</del>" in out["model"]
        assert "<ins>" + expect_in_source + "</ins>" in out["source"]

    def test_a_whitespace_only_difference_is_reported_clean(self):
        """The PDF line-number case. `normalized` treats it as a real match, so the page
        must not present it as a discrepancy."""
        out = _run(
            "labeling required for safe and effective use",
            "labeling required for  safe and effective use",
        )
        assert out["clean"] is True

    def test_marks_appear_on_the_side_they_belong_to(self):
        """A token missing from the document belongs in the model pane and nowhere else;
        putting it in both would say the document contains something it does not."""
        out = _run("the sponsor shall submit", "the sponsor shall")
        assert out["dels"] >= 1
        assert out["inses"] == 0
        assert "<ins>" not in out["source"]

    def test_an_unlocated_quote_marks_everything_as_model_only(self):
        out = _run("some quoted text", "")
        assert out["clean"] is False
        assert out["dels"] >= 1
        assert out["inses"] == 0

    def test_an_empty_quote_does_not_crash(self):
        out = _run("", "the document says something")
        assert out["dels"] == 0


class TestDiffIsSafe:
    def test_markup_in_the_source_is_escaped_not_rendered(self):
        """The source text is a PDF extract and the quote is model output. Neither is
        trusted input, and both flow through the diff before reaching innerHTML."""
        payload = "<script>alert(1)</script> and more"
        out = _run(payload, payload)
        assert "<script>" not in out["model"]
        assert "&lt;script&gt;" in out["model"]

    def test_escaping_survives_a_diff_that_marks_the_dangerous_token(self):
        out = _run("<img onerror=x> here", "safe here")
        assert "<img" not in out["model"]
        assert "&lt;img" in out["model"]

    def test_quotes_and_ampersands_are_escaped(self):
        out = _run('a "quoted" & thing', 'a "quoted" & thing')
        assert "&quot;" in out["model"] and "&amp;" in out["model"]
