"""Tests for `bin/build-demo`.

The script had none, and it is the one piece of tooling whose output is *published*. Three
of its defects were the same shape as everything else this project has found -- a claim
that outran what was measured:

* the narration said "a hundred and sixty documents" when 152 have a usable text layer and
  are the only ones in the index;
* it said dense retrieval loses "by a factor of seventy five", which is the `fixed`
  chunking's margin, while the service it was filming serves `structural` at 43x;
* its quota guard matched the substring "quota", so a *per-minute* 429 -- whose body reads
  "you exceeded your current quota" -- was reported as a spent day, advising the operator
  to come back tomorrow over a limit that clears in seconds.

The narration and the probe are embedded Python heredocs, so they are extracted and run
directly. That needs no Chrome, no ffmpeg, no running service and no quota.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "build-demo"
SYSTEM_BASH = "/bin/bash"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists() or not Path(SYSTEM_BASH).exists(),
    reason="needs bin/build-demo and /bin/bash",
)


def _heredoc(name: str) -> str:
    """The body of a `cat > name <<'MARKER'` block, verbatim."""
    text = SCRIPT.read_text()
    marker = {"narrate.py": "NARRATE_PY", "probe_check.py": "PROBE_PY"}[name]
    m = re.search(rf"^cat > {re.escape(name)} <<'{marker}'\n(.*?)^{marker}$", text, re.S | re.M)
    assert m, f"could not find the {name} heredoc"
    return m.group(1)


def _run_embedded(name: str, *args, stdin=""):
    return subprocess.run(
        ["python3", "-c", _heredoc(name), *args], input=stdin, capture_output=True, text=True
    )


def _health(**over):
    base = {"docs": 152, "chunks": 13423, "chunking": "structural", "ready": True}
    base.update(over)
    return json.dumps(base)


REPORT = str(ROOT / "reports" / "failure_modes.json")


class TestTheScriptItself:
    def test_it_parses_under_the_system_bash(self):
        r = subprocess.run([SYSTEM_BASH, "-n", str(SCRIPT)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

    def test_help_exits_zero_and_needs_nothing(self):
        """`--help` used to fall through to "unknown flag" and exit 2, so the only way to
        read the usage was to open the file."""
        r = subprocess.run(
            [SYSTEM_BASH, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60
        )
        assert r.returncode == 0
        assert "--no-voice" in r.stdout

    def test_an_unknown_flag_still_fails_and_points_at_help(self):
        r = subprocess.run(
            [SYSTEM_BASH, str(SCRIPT), "--nope"], capture_output=True, text=True, timeout=60
        )
        assert r.returncode == 2
        assert "--help" in r.stderr

    def test_ffprobe_is_checked_because_subtitle_timing_depends_on_it(self):
        """ffprobe ships with ffmpeg but a minimal build can omit it, and its absence is
        silent: every cue gets a zero duration and the subtitles all appear at once."""
        assert re.search(r"for tool in [^\n]*\bffprobe\b", SCRIPT.read_text())

    def test_it_is_reachable_from_the_cli_and_the_makefile(self):
        assert "bin/build-demo" in (ROOT / "bin" / "agentrag").read_text()
        assert re.search(r"^demo:", (ROOT / "Makefile").read_text(), re.M)


class TestNarrationIsDerived:
    def test_the_document_count_comes_from_health(self):
        out = _run_embedded("narrate.py", _health(docs=152), REPORT).stdout
        assert "a hundred and fifty two public F D A guidance documents" in out
        assert "a hundred and sixty" not in out

    def test_a_different_corpus_size_changes_the_spoken_number(self):
        out = _run_embedded("narrate.py", _health(docs=207), REPORT).stdout
        assert "two hundred and seven public" in out

    def test_the_identifier_margin_follows_the_chunking_actually_served(self):
        """The defect: the narration quoted the best configuration's 75x while filming a
        service serving a different one."""
        structural = _run_embedded("narrate.py", _health(chunking="structural"), REPORT).stdout
        semantic = _run_embedded("narrate.py", _health(chunking="semantic"), REPORT).stdout
        assert "by a factor of forty three" in structural
        assert "by a factor of fifty six" in semantic

    def test_the_better_configuration_is_named_as_a_different_split_not_as_this_one(self):
        out = _run_embedded("narrate.py", _health(chunking="structural"), REPORT).stdout
        assert "as much as seventy five on a different way of splitting" in out

    def test_the_passage_count_is_rounded_down_so_the_claim_stays_true(self):
        out = _run_embedded("narrate.py", _health(chunks=13423), REPORT).stdout
        assert "over thirteen thousand passages" in out

    def test_rounding_to_nearest_would_overstate_the_corpus(self):
        """13,423 floors and rounds to the same 13, so the current corpus cannot tell the
        two apart. At 13,800 it can: "over fourteen thousand passages" would be false of
        13,800, and "over" is doing real work in that sentence."""
        out = _run_embedded("narrate.py", _health(chunks=13800), REPORT).stdout
        assert "over thirteen thousand passages" in out
        assert "fourteen thousand" not in out

    def test_an_unmeasured_chunking_fails_loudly_rather_than_quoting_another(self):
        r = _run_embedded("narrate.py", _health(chunking="exotic"), REPORT)
        assert r.returncode != 0
        assert "no identifier margin" in r.stderr
        assert "agentrag failures" in r.stderr

    def test_a_health_payload_missing_fields_fails_rather_than_guessing(self):
        r = _run_embedded("narrate.py", json.dumps({"chunks": 1, "chunking": "structural"}), REPORT)
        assert r.returncode != 0
        assert "did not report" in r.stderr

    def test_exactly_five_scenes_are_emitted(self):
        out = _run_embedded("narrate.py", _health(), REPORT).stdout.strip().splitlines()
        assert len(out) == 5
        assert [line.split("|")[0] for line in out] == [
            "01_empty",
            "02_answer",
            "03_cite",
            "04_refusal",
            "05_arch",
        ]

    def test_no_narration_line_contains_a_digit(self):
        """Everything spoken has to be words: `say` reads digits inconsistently across
        voices, and the subtitle is generated from the same string."""
        out = _run_embedded("narrate.py", _health(), REPORT).stdout
        for line in out.strip().splitlines():
            narration = line.split("|", 1)[1]
            assert not re.search(r"\d", narration), narration


class TestTheProbeGuards:
    @staticmethod
    def _probe(**payload):
        return _run_embedded("probe_check.py", stdin=json.dumps(payload))

    def test_a_healthy_probe_passes(self):
        r = self._probe(
            answered=True, citations=[1, 2, 3], verification={"verified": 2, "claimed": 3}
        )
        assert r.returncode == 0
        assert "2/3 located" in r.stdout

    def test_an_exhausted_day_stops_the_build(self):
        r = self._probe(answered=False, refusal_source="daily_quota", reason="quota exhausted")
        assert r.returncode != 0
        assert "QUOTA EXHAUSTED" in r.stderr

    def test_a_per_minute_rate_limit_is_not_reported_as_a_spent_day(self):
        """The substring bug. A per-minute 429's body contains the word "quota", so the
        old guard told the operator to wait for tomorrow's reset."""
        r = self._probe(
            answered=False,
            refusal_source="generation_error",
            reason="generation failed: http 429: You exceeded your current quota",
        )
        assert r.returncode != 0
        assert "RATE LIMITED" in r.stderr
        assert "QUOTA EXHAUSTED" not in r.stderr
        assert "requests per day" not in r.stderr

    def test_the_daily_case_is_decided_by_the_field_not_the_prose(self):
        """A reason mentioning the daily cap without the structured field must not be
        promoted to one; the field is what the upstream `quotaId` check produced."""
        r = self._probe(
            answered=False,
            refusal_source="generation_error",
            reason="generation failed: rate limit, retry in 4.4s",
        )
        assert "RATE LIMITED" in r.stderr

    def test_an_empty_context_refusal_says_re_running_will_not_help(self):
        r = self._probe(answered=False, refusal_source="no_context", reason="no usable context")
        assert r.returncode != 0
        assert "will\n  not help" in r.stderr or "not help" in r.stderr
        assert "agentrag status" in r.stderr

    def test_a_score_gate_refusal_says_no_quota_was_spent(self):
        r = self._probe(answered=False, refusal_source="score_gate", reason="below threshold")
        assert "no quota was spent" in r.stderr

    def test_an_unreported_source_is_labelled_rather_than_guessed(self):
        r = self._probe(answered=False, reason="model declined")
        assert "Source: unreported" in r.stderr
        assert "non-deterministic" in r.stderr

    def test_answering_without_citations_stops_the_build(self):
        r = self._probe(answered=True, citations=[], verification={"verified": 0, "claimed": 0})
        assert r.returncode != 0
        assert "cited nothing" in r.stderr

    def test_zero_located_quotes_stops_the_build(self):
        """Scene 3 narrates the verification panes as the thing worth looking at. Five red
        borders under that narration is the same audio/picture mismatch the quota guard
        exists to prevent."""
        r = self._probe(
            answered=True, citations=[1, 2, 3, 4, 5], verification={"verified": 0, "claimed": 5}
        )
        assert r.returncode != 0
        assert "NOT ONE quote" in r.stderr
        assert "0/5 located" in r.stderr

    def test_one_located_quote_is_enough_to_proceed(self):
        r = self._probe(
            answered=True, citations=[1, 2, 3, 4, 5], verification={"verified": 1, "claimed": 5}
        )
        assert r.returncode == 0

    def test_an_empty_response_body_is_reported_as_a_dead_service(self):
        r = _run_embedded("probe_check.py", stdin="")
        assert r.returncode != 0
        assert "returned nothing" in r.stderr


def _inline_heredoc(marker: str) -> str:
    """The body of a bare `python3 - <<'MARKER'` block, verbatim.

    Separate from `_heredoc` because these are piped into an interpreter rather than
    written to a file, so they have no name to key on -- only the marker.
    """
    text = SCRIPT.read_text()
    m = re.search(rf"<<'{re.escape(marker)}'\n(.*?)^{re.escape(marker)}$", text, re.S | re.M)
    assert m, f"could not find the {marker} heredoc"
    return m.group(1)


def _shell_field(name: str) -> str:
    """A top-level `NAME=value` assignment from the script."""
    m = re.search(rf"^{re.escape(name)}=(\S+)", SCRIPT.read_text(), re.M)
    assert m, f"{name} is not assigned at the top level"
    return m.group(1)


class TestTheSubtitleBandIsReserved:
    """The captions used to be drawn straight over the picture.

    At 720p that put them across the EVALUATE lane of the architecture diagram, covering
    `192 configurations`, `the finding` and the 43-75x result -- the single most important
    number in the video, obscured in the frame whose job is to show it. The band is the
    fix, and these tests pin the arithmetic that keeps it reserved.
    """

    def test_the_band_and_the_content_fill_the_frame_exactly(self):
        band = int(_shell_field("SUBTITLE_BAND"))
        assert band > 0
        # CONTENT_H is derived, so the two cannot drift apart into a gap or an overlap.
        assert "CONTENT_H=$((720 - SUBTITLE_BAND))" in SCRIPT.read_text()

    def test_frames_are_scaled_into_the_content_area_not_the_whole_frame(self):
        text = SCRIPT.read_text()
        assert "scale=1280:$CONTENT_H:force_original_aspect_ratio=decrease" in text
        # 1280:720 here would fill the frame and put the picture back under the captions.
        assert "scale=1280:720:force_original_aspect_ratio=decrease" not in text

    def test_frames_are_top_aligned_so_the_band_is_not_shared(self):
        """`(oh-ih)/2` would centre the picture and split the reserved height in two, so
        half the band would have picture in it and the captions would overlap again."""
        text = SCRIPT.read_text()
        assert "pad=1280:720:(ow-iw)/2:0:color=#0f1117" in text
        assert "pad=1280:720:(ow-iw)/2:(oh-ih)/2" not in text

    def test_two_caption_lines_fit_inside_the_band(self):
        band = int(_shell_field("SUBTITLE_BAND"))
        style = re.search(r"force_style=\\?\n?'([^']+)'", SCRIPT.read_text(), re.S)
        assert style, "the burn-in step has no force_style"
        opts = dict(kv.split("=", 1) for kv in style.group(1).replace("\\\n", "").split(","))
        size, margin = float(opts["FontSize"]), float(opts["MarginV"])
        # libass line advance is ~1.2x the point size; two lines plus the bottom margin.
        assert 2 * 1.2 * size + margin <= band, (
            f"two lines at FontSize={size} plus MarginV={margin} need "
            f"{2 * 1.2 * size + margin:.0f}px but the band is {band}px"
        )

    def test_the_captions_are_anchored_to_the_bottom(self):
        style = re.search(r"force_style=\\?\n?'([^']+)'", SCRIPT.read_text(), re.S)
        opts = dict(kv.split("=", 1) for kv in style.group(1).replace("\\\n", "").split(","))
        assert opts["Alignment"] == "2", "ASS alignment 2 is bottom-centre"


class TestTheSubtitleCanvasIsPinned:
    """ffmpeg renders an SRT through a 384x288 ASS canvas.

    Every style number is therefore scaled by 2.5 on the way to 720p unless the canvas is
    pinned -- which is why `MarginV=42` placed the captions 105 real pixels up, over the
    picture, rather than in the margin it names. A band measured in pixels can only be
    aimed at with style units that are also pixels.
    """

    def _pin(self, tmp_path, body):
        (tmp_path / "demo.ass").write_text(body)
        return subprocess.run(
            ["python3", "-c", _inline_heredoc("ASS_PIN")],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

    HEADER = "[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\n"

    def test_it_rewrites_the_canvas_to_the_video_size(self, tmp_path):
        r = self._pin(tmp_path, self.HEADER)
        assert r.returncode == 0, r.stderr
        out = (tmp_path / "demo.ass").read_text()
        assert "PlayResX: 1280" in out and "PlayResY: 720" in out
        assert "384" not in out and "288" not in out

    def test_it_refuses_rather_than_guessing_when_there_is_no_canvas(self, tmp_path):
        """Defaulting here would leave the style units an unknown multiple of a pixel, and
        the failure mode is captions over the content -- silent, and only visible by
        watching the finished file."""
        r = self._pin(tmp_path, "[Script Info]\nScriptType: v4.00+\n")
        assert r.returncode != 0
        assert "refusing to guess" in (r.stderr + r.stdout)

    def test_it_leaves_the_rest_of_the_file_alone(self, tmp_path):
        body = self.HEADER + "\n[Events]\nDialogue: 0,0:00:00.00,0:00:04.14,Default,,0,0,0,,hi\n"
        r = self._pin(tmp_path, body)
        assert r.returncode == 0
        assert (
            "Dialogue: 0,0:00:00.00,0:00:04.14,Default,,0,0,0,,hi"
            in (tmp_path / "demo.ass").read_text()
        )

    def test_the_burn_in_reads_the_pinned_ass_not_the_raw_srt(self):
        """Burning `demo.srt` would go back through the unpinned 384x288 canvas and undo
        the whole fix while still looking like it had been applied."""
        text = SCRIPT.read_text()
        assert "subtitles=demo.ass:force_style=" in text
        assert "subtitles=demo.srt" not in text


class TestTheCitationsFrameIsCropped:
    """Headless Chrome's `--screenshot` renders from the document origin and ignores the
    scroll the `#citations` deep link performs.

    Scene 3 came out 87% empty, with the verification panes just entering at the bottom
    edge, under narration describing them. Confirmed over CDP rather than guessed: the page
    reports `scrollY: 610` and `citeTopViewport: 0`, so the deep link is correct and the
    screenshot is what ignores it. The fix captures the whole page and crops to the offset
    the page publishes on `body[data-cite-top]`.
    """

    def _sed(self, dom):
        m = re.search(r"sed -n '(s/\.\*data-cite-top[^']+)'", SCRIPT.read_text())
        assert m, "cite_shot no longer parses data-cite-top with sed"
        r = subprocess.run(["sed", "-n", m.group(1)], input=dom, capture_output=True, text=True)
        return r.stdout.strip()

    def test_it_reads_the_offset_the_page_publishes(self):
        assert self._sed('<body class="x" data-cite-top="609">\n') == "609"

    def test_it_reads_the_offset_wherever_it_sits_in_the_tag(self):
        assert self._sed('<body data-cite-top="1204" class="y">\n') == "1204"

    def test_a_page_without_the_attribute_yields_nothing_so_the_guard_fires(self):
        """An empty parse has to stay empty. A default would crop a fixed offset against a
        page whose height depends on the answer -- which is the guess the deep link was
        added to remove."""
        assert self._sed('<body class="x">\n<div>no citations</div>\n') == ""

    def test_the_guard_refuses_an_unreported_offset(self):
        text = SCRIPT.read_text()
        assert "the page did not report data-cite-top" in text
        assert '[ -n "$top" ]' in text

    def test_the_canvas_is_checked_against_the_offset_rather_than_assumed(self):
        """The page grows with the answer, so a canvas that fits today's page is not a
        canvas that fits tomorrow's. Cropping past the bottom would yield a black frame."""
        text = SCRIPT.read_text()
        assert '[ "$((top + 720))" -le "$CITE_CANVAS" ]' in text
        assert int(_shell_field("CITE_CANVAS")) >= 720

    def test_the_crop_doubles_the_offset_for_the_device_scale_factor(self):
        """Frames are captured at `--force-device-scale-factor=2`, so a CSS offset is half
        a device pixel offset. Cropping at the CSS value would land at the wrong place by
        exactly the offset itself."""
        text = SCRIPT.read_text()
        assert "crop=2560:1440:0:$((top * 2))" in text

    def test_the_geometry_and_the_pixels_come_from_one_page_load(self):
        """Two loads would be two generation requests against a 20-per-day quota, and two
        different answers -- so the offset could describe a page that was never captured."""
        body = re.search(r"cite_shot \(\) \{(.*?)\n\}", SCRIPT.read_text(), re.S)
        assert body, "cite_shot is gone"
        assert body.group(1).count("$CHROME") == 1
        assert "--screenshot=" in body.group(1) and "--dump-dom" in body.group(1)

    def test_the_citations_scene_uses_the_cropping_capture(self):
        text = SCRIPT.read_text()
        assert re.search(r"^cite_shot 03_cite\.png ", text, re.M)
        assert not re.search(r"^shot 03_cite\.png ", text, re.M)
