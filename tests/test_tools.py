"""Unit tests for coding tools (no LLM required)."""

from __future__ import annotations

import os
import tempfile

import pytest

from rlm_code.tools import (
    bash,
    build_coding_tools,
    edit_file,
    glob_files,
    grep_files,
    read_file,
    set_working_directory,
    write_file,
)


@pytest.fixture(autouse=True)
def tmp_workdir(tmp_path):
    """Set working directory to a temp directory for every test."""
    set_working_directory(str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


class TestBash:
    def test_echo(self):
        result = bash("echo hello")
        assert "hello" in result
        assert "[exit code: 0]" in result

    def test_timeout(self):
        result = bash("sleep 10", timeout=1)
        assert "timed out" in result.lower()

    def test_stderr(self):
        result = bash("echo err >&2")
        assert "err" in result
        assert "[stderr]" in result

    def test_exit_code(self):
        result = bash("exit 42")
        assert "[exit code: 42]" in result


# ---------------------------------------------------------------------------
# read_file / write_file
# ---------------------------------------------------------------------------


class TestReadWrite:
    def test_read_write(self, tmp_workdir):
        write_file("test.txt", "line1\nline2\nline3\n")
        content = read_file("test.txt")
        assert "1\tline1" in content
        assert "2\tline2" in content
        assert "3\tline3" in content

    def test_read_not_found(self):
        result = read_file("nonexistent.txt")
        assert "[error]" in result
        assert "not found" in result.lower()

    def test_read_offset_limit(self, tmp_workdir):
        lines = "\n".join(f"line{i}" for i in range(20))
        write_file("big.txt", lines)
        content = read_file("big.txt", offset=5, limit=3)
        assert "line5" in content
        assert "line7" in content
        assert "line8" not in content

    def test_write_creates_dirs(self, tmp_workdir):
        write_file("a/b/c/file.txt", "nested")
        assert os.path.exists(os.path.join(str(tmp_workdir), "a", "b", "c", "file.txt"))
        content = read_file("a/b/c/file.txt")
        assert "nested" in content


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


class TestEdit:
    def test_edit(self, tmp_workdir):
        write_file("edit.txt", "hello world\nfoo bar\n")
        result = edit_file("edit.txt", "foo bar", "baz qux")
        assert "Replaced" in result
        content = read_file("edit.txt")
        assert "baz qux" in content
        assert "foo bar" not in content

    def test_edit_not_found_text(self, tmp_workdir):
        write_file("edit2.txt", "hello world\n")
        result = edit_file("edit2.txt", "nonexistent text", "replacement")
        assert "[error]" in result
        assert "not found" in result.lower()

    def test_edit_file_not_found(self):
        result = edit_file("missing.txt", "a", "b")
        assert "[error]" in result


# ---------------------------------------------------------------------------
# glob_files
# ---------------------------------------------------------------------------


class TestGlob:
    def test_glob(self, tmp_workdir):
        write_file("a.py", "")
        write_file("b.py", "")
        write_file("c.txt", "")
        result = glob_files("*.py")
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_glob_skips_git(self, tmp_workdir):
        os.makedirs(os.path.join(str(tmp_workdir), ".git"), exist_ok=True)
        write_file(".git/config", "gitconfig")
        write_file("real.py", "")
        result = glob_files("*")
        assert "real.py" in result
        assert ".git" not in result

    def test_glob_no_match(self, tmp_workdir):
        result = glob_files("*.xyz")
        assert "No files" in result

    def test_glob_subdir(self, tmp_workdir):
        write_file("sub/deep.py", "")
        result = glob_files("*.py")
        assert "deep.py" in result


# ---------------------------------------------------------------------------
# grep_files
# ---------------------------------------------------------------------------


class TestGrep:
    def test_grep(self, tmp_workdir):
        write_file("search.py", "def hello():\n    return 'world'\n")
        result = grep_files("def hello")
        assert "search.py" in result
        assert "def hello" in result

    def test_grep_no_match(self, tmp_workdir):
        write_file("empty.py", "pass\n")
        result = grep_files("nonexistent_pattern_xyz")
        assert "No matches" in result

    def test_grep_with_glob(self, tmp_workdir):
        write_file("a.py", "target\n")
        write_file("b.txt", "target\n")
        result = grep_files("target", glob="*.py")
        assert "a.py" in result
        assert "b.txt" not in result

    def test_grep_with_context(self, tmp_workdir):
        write_file("ctx.py", "line1\nline2\ntarget\nline4\nline5\n")
        result = grep_files("target", context=1)
        assert "line2" in result
        assert "line4" in result


# ---------------------------------------------------------------------------
# build_coding_tools
# ---------------------------------------------------------------------------


class TestBuildTools:
    def test_build_returns_all_tools(self, tmp_workdir):
        tools = build_coding_tools(str(tmp_workdir))
        expected = {"bash", "read_file", "write_file", "edit_file", "glob_files", "grep_files"}
        assert set(tools.keys()) == expected

    def test_tools_have_description(self, tmp_workdir):
        tools = build_coding_tools(str(tmp_workdir))
        for name, entry in tools.items():
            assert "tool" in entry, f"{name} missing 'tool' key"
            assert "description" in entry, f"{name} missing 'description' key"
            assert callable(entry["tool"]), f"{name} tool is not callable"
