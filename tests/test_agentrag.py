"""Structural tests for `bin/agentrag`.

Written because the script's `--no-build` path shipped broken and failed *silently*:
macOS ships bash 3.2, where `set -u` plus an empty array expansion (`"${build[@]}"` with
`build=()`) is a fatal "unbound variable". `agentrag start` exited before its health wait
and its status report, printing only "==> starting api + qdrant". The `--build` path was
fine, so a single manual test of the default passed — the same shape as every other defect
this project has found: the path nobody exercised.

These tests need no Docker, no colima and no network. They check the two things that can
drift silently: that the script parses and runs under the *system* bash, and that its help
text and its dispatch table still describe each other.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "bin" / "agentrag"

#: The system bash, not whatever is first on PATH. Homebrew's bash 5 would hide exactly
#: the incompatibility that broke this script.
SYSTEM_BASH = "/bin/bash"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists() or not Path(SYSTEM_BASH).exists(),
    reason="needs bin/agentrag and /bin/bash",
)


def _run(*args, timeout=60):
    return subprocess.run(
        [SYSTEM_BASH, str(SCRIPT), *args], capture_output=True, text=True, timeout=timeout
    )


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


class TestItParsesUnderTheSystemBash:
    def test_the_script_is_syntactically_valid(self):
        proc = subprocess.run(
            [SYSTEM_BASH, "-n", str(SCRIPT)], capture_output=True, text=True, timeout=60
        )
        assert proc.returncode == 0, proc.stderr

    def test_no_empty_array_expansion_survives(self):
        """The exact bash 3.2 trap that broke `--no-build`. `"${x[@]}"` on a possibly
        empty array is fatal under `set -u` here, and the failure is a one-line stderr
        message in the middle of otherwise normal-looking output."""
        offenders = [
            (i, line)
            for i, line in enumerate(_source().splitlines(), 1)
            # Comments are skipped: the comment explaining this very trap quotes the
            # unsafe form, and a comment cannot execute.
            if not line.lstrip().startswith("#")
            and re.search(r'"\$\{[A-Za-z_][A-Za-z0-9_]*\[@\]\}"', line)
        ]
        assert not offenders, f"unsafe array expansion for bash 3.2: {offenders}"

    def test_it_is_executable(self):
        assert SCRIPT.stat().st_mode & 0o111


class TestHelpAndDispatchAgree:
    """Either direction of drift is a real bug: an advertised command that does not
    dispatch is a broken promise, and a dispatched command that is never advertised is
    undiscoverable."""

    def _advertised(self) -> set[str]:
        out = _run().stdout
        return set(re.findall(r"agentrag ([a-z][a-z-]*)", out))

    def _dispatched(self) -> set[str]:
        source = _source()
        block = source[source.index('cmd="${1:-help}"') :]
        names: set[str] = set()
        for line in block.splitlines():
            match = re.match(r"\s{2}([a-z|_\-]+)\)\s", line)
            if match and match.group(1) != "*":
                names.update(part for part in match.group(1).split("|") if part)
        return names

    def test_bare_invocation_lists_commands_and_exits_zero(self):
        proc = _run()
        assert proc.returncode == 0, proc.stderr
        for expected in ("start", "stop", "status"):
            assert f"agentrag {expected}" in proc.stdout, expected

    def test_every_advertised_command_dispatches(self):
        missing = self._advertised() - self._dispatched()
        assert not missing, f"advertised in help but not dispatched: {sorted(missing)}"

    def test_the_three_headline_commands_are_both_advertised_and_dispatched(self):
        for name in ("start", "stop", "status"):
            assert name in self._advertised(), name
            assert name in self._dispatched(), name

    def test_help_is_reachable_by_every_documented_spelling(self):
        for args in ([], ["help"], ["--help"], ["-h"]):
            proc = _run(*args)
            assert proc.returncode == 0, (args, proc.stderr)
            assert "agentrag status" in proc.stdout, args

    def test_an_unknown_command_fails_loudly_rather_than_doing_nothing(self):
        proc = _run("frobnicate")
        assert proc.returncode == 2
        assert "unknown command" in proc.stderr
        # And it still shows what *is* available, so the failure is self-correcting.
        assert "agentrag status" in proc.stdout


class TestItNeverPrintsTheKey:
    def test_the_key_is_exported_but_never_echoed(self):
        """`start` passes GEMINI_API_KEY to compose as an environment variable so no
        secret enters an image layer. It must not reach the terminal either -- and `run`
        echoes every command it executes, so the key must never be a command argument."""
        source = _source()
        block = source[source.index("load_key() {") : source.index("api_json()")]
        assert "export GEMINI_API_KEY" in block
        for line in block.splitlines():
            if "GEMINI_API_KEY" in line and re.match(r"\s*(say|info|warn|printf|echo|run)\b", line):
                pytest.fail(f"the key could reach the terminal: {line.strip()}")

    def test_status_reports_key_presence_without_the_value(self):
        source = _source()
        assert "Gemini key present (not shown)" in source

    def test_no_run_invocation_takes_the_key_as_an_argument(self):
        for line in _source().splitlines():
            if line.strip().startswith("run ") and "GEMINI_API_KEY" in line:
                pytest.fail(f"`run` echoes its command; this would print the key: {line.strip()}")


class TestRepoRootResolution:
    def test_it_resolves_the_repo_through_a_symlink(self, tmp_path):
        """It is installed as a symlink on PATH, and `readlink -f` is not portable to
        BSD, so the resolution loop is hand-rolled and worth testing."""
        link = tmp_path / "agentrag-link"
        link.symlink_to(SCRIPT)
        proc = subprocess.run(
            [SYSTEM_BASH, str(link)], capture_output=True, text=True, timeout=60, cwd=str(tmp_path)
        )
        assert proc.returncode == 0, proc.stderr
        assert str(SCRIPT.parents[1]) in proc.stdout

    def test_it_does_not_depend_on_the_callers_directory(self, tmp_path):
        proc = subprocess.run(
            [SYSTEM_BASH, str(SCRIPT)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(tmp_path),
        )
        assert proc.returncode == 0
        assert str(SCRIPT.parents[1]) in proc.stdout
