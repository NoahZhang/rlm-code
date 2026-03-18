"""
Coding tools for the RLM Code CLI.

Provides bash, read_file, write_file, edit_file, glob_files, and grep_files
as custom tools for the RLM REPL environment.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess

_working_directory: str = os.getcwd()

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache"}


def set_working_directory(path: str) -> None:
    global _working_directory
    _working_directory = os.path.abspath(path)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_working_directory, path))


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def bash(command: str, timeout: int = 120) -> str:
    """Execute a shell command and return stdout, stderr, and exit code."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=_working_directory,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        parts: list[str] = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr}")
        parts.append(f"[exit code: {result.returncode}]")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"[error] Command timed out after {timeout}s"
    except Exception as e:
        return f"[error] {e}"


def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a file and return its content with line numbers."""
    resolved = _resolve_path(path)
    if not os.path.exists(resolved):
        return f"[error] File not found: {path}"
    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        selected = lines[offset : offset + limit]
        numbered = []
        for i, line in enumerate(selected, start=offset + 1):
            numbered.append(f"{i:>6}\t{line.rstrip()}")
        return "\n".join(numbered)
    except Exception as e:
        return f"[error] {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    resolved = _resolve_path(path)
    try:
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"[error] {e}"


def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace the first occurrence of old_text with new_text in a file."""
    resolved = _resolve_path(path)
    if not os.path.exists(resolved):
        return f"[error] File not found: {path}"
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            content = f.read()
        if old_text not in content:
            return f"[error] old_text not found in {path}"
        new_content = content.replace(old_text, new_text, 1)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(new_content)
        old_lines = old_text.count("\n") + 1
        new_lines = new_text.count("\n") + 1
        return f"Replaced {old_lines} lines with {new_lines} lines in {path}"
    except Exception as e:
        return f"[error] {e}"


def glob_files(pattern: str, path: str | None = None) -> str:
    """Find files matching a glob pattern. Skips .git, node_modules, __pycache__, etc."""
    base = _resolve_path(path) if path else _working_directory
    if not os.path.isdir(base):
        return f"[error] Directory not found: {base}"

    matches: list[str] = []
    limit = 200
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                rel = os.path.relpath(os.path.join(root, name), _working_directory)
                matches.append(rel)
                if len(matches) >= limit:
                    break
        if len(matches) >= limit:
            break

    if not matches:
        return f"No files matching '{pattern}'"
    result = "\n".join(sorted(matches))
    if len(matches) >= limit:
        result += f"\n... (truncated at {limit} results)"
    return result


def grep_files(
    pattern: str, path: str | None = None, glob: str | None = None, context: int = 0
) -> str:
    """Search file contents using grep -rnE. Returns up to 500 matching lines."""
    search_path = _resolve_path(path) if path else _working_directory
    cmd = ["grep", "-rnE"]
    if context > 0:
        cmd.extend(["-C", str(context)])
    if glob:
        cmd.extend(["--include", glob])
    cmd.extend([
        "--exclude-dir=.git",
        "--exclude-dir=node_modules",
        "--exclude-dir=__pycache__",
        "--exclude-dir=.venv",
    ])
    cmd.extend([pattern, search_path])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        output = result.stdout
        if not output:
            return f"No matches for pattern '{pattern}'"
        lines = output.splitlines()
        limit = 500
        if len(lines) > limit:
            lines = lines[:limit]
            lines.append(f"... (truncated at {limit} results)")
        # Make paths relative
        results = []
        for line in lines:
            results.append(line.replace(f"{_working_directory}/", ""))
        return "\n".join(results)
    except subprocess.TimeoutExpired:
        return "[error] grep timed out after 30s"
    except Exception as e:
        return f"[error] {e}"


# ---------------------------------------------------------------------------
# Build tools dict for RLM
# ---------------------------------------------------------------------------


def build_coding_tools(working_directory: str) -> dict[str, dict]:
    """Build the custom_tools dict in RLM format for the coding tools."""
    set_working_directory(working_directory)
    return {
        "bash": {
            "tool": bash,
            "description": "bash(command, timeout=120) — Execute a shell command. Returns stdout, stderr, and exit code.",
        },
        "read_file": {
            "tool": read_file,
            "description": "read_file(path, offset=0, limit=2000) — Read a file with line numbers.",
        },
        "write_file": {
            "tool": write_file,
            "description": "write_file(path, content) — Write content to a file, creating directories as needed.",
        },
        "edit_file": {
            "tool": edit_file,
            "description": "edit_file(path, old_text, new_text) — Replace first occurrence of old_text with new_text.",
        },
        "glob_files": {
            "tool": glob_files,
            "description": "glob_files(pattern, path=None) — Find files matching a glob pattern.",
        },
        "grep_files": {
            "tool": grep_files,
            "description": "grep_files(pattern, path=None, glob=None, context=0) — Search file contents with regex.",
        },
    }
