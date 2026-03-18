"""Core coding engine: code execution, parsing, and iteration loop."""

from __future__ import annotations

import ast
import io
import json
import re
import sys
import threading
import traceback
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field

from .llm import LLMClient
from .tools import build_coding_tools


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExecResult:
    stdout: str
    stderr: str


@dataclass
class CodeBlockResult:
    code: str
    stdout: str
    stderr: str


@dataclass
class Iteration:
    iteration: int
    response: str
    code_blocks: list[CodeBlockResult]


@dataclass
class EngineResult:
    answer: str
    iterations: list[Iteration]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```repl\s*\n(.*?)\n```", re.DOTALL)
_FINAL_RE = re.compile(r"^\s*FINAL\((.*?)\)", re.MULTILINE)
_FINAL_VAR_RE = re.compile(r"FINAL_VAR\((\w+)\)")


def parse_code_blocks(text: str) -> list[str]:
    """Extract code from ```repl``` fenced blocks."""
    return _CODE_BLOCK_RE.findall(text)


_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>functions\.(\w+):\d+<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[str]:
    """Fallback: convert raw tool-call markup to Python code strings."""
    results = []
    for name, args_json in _TOOL_CALL_RE.findall(text):
        try:
            args = json.loads(args_json)
        except (json.JSONDecodeError, ValueError):
            continue
        # Build Python call: tool_name(key=value, ...)
        parts = [f"{k}={repr(v)}" for k, v in args.items()]
        code = f"result = {name}({', '.join(parts)})\nprint(result)"
        results.append(code)
    return results


def parse_final_answer(text: str, executor: CodeExecutor) -> str | None:
    """Check for FINAL_VAR(...) or FINAL(...) in text. Returns answer or None."""
    # Check if FINAL_VAR was called during code execution
    if executor.last_final_answer is not None:
        return executor.last_final_answer

    # Check text for FINAL_VAR(var_name)
    m = _FINAL_VAR_RE.search(text)
    if m:
        var_name = m.group(1)
        if var_name in executor.namespace:
            return str(executor.namespace[var_name])

    # Check text for FINAL(...)
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# Code executor
# ---------------------------------------------------------------------------


_BLOCKED_BUILTINS = {"exit", "quit", "eval", "exec", "input", "compile", "globals", "locals"}


class CodeExecutor:
    """Persistent Python execution environment with coding tools injected."""

    def __init__(
        self,
        working_directory: str,
    ) -> None:
        self.last_final_answer: str | None = None
        self.namespace: dict = {}
        self._lock = threading.Lock()
        self._init_namespace(working_directory)

    def _init_namespace(
        self,
        working_directory: str,
    ) -> None:
        # Safe builtins — remove dangerous ones
        safe_builtins = {k: v for k, v in __builtins__.items() if k not in _BLOCKED_BUILTINS} if isinstance(__builtins__, dict) else {k: getattr(__builtins__, k) for k in dir(__builtins__) if k not in _BLOCKED_BUILTINS}
        self.namespace["__builtins__"] = safe_builtins

        # Inject coding tools
        tools = build_coding_tools(working_directory)
        for name, entry in tools.items():
            self.namespace[name] = entry["tool"]

        # Inject FINAL_VAR and SHOW_VARS helpers
        self.namespace["FINAL_VAR"] = self._final_var
        self.namespace["SHOW_VARS"] = self._show_vars

    def _final_var(self, var_name: str) -> str:
        """Record a variable's value as the final answer."""
        if var_name in self.namespace:
            self.last_final_answer = str(self.namespace[var_name])
            return self.last_final_answer
        return f"[error] Variable '{var_name}' not found"

    def _show_vars(self) -> str:
        """Return a summary of user-defined variables in the namespace."""
        skip = {"__builtins__", "FINAL_VAR", "SHOW_VARS",
                "bash", "read_file", "write_file", "edit_file",
                "glob_files", "grep_files"}
        items = []
        for k, v in self.namespace.items():
            if k.startswith("_") or k in skip or callable(v):
                continue
            rep = repr(v)
            if len(rep) > 200:
                rep = rep[:200] + "..."
            items.append(f"  {k} = {rep}")
        if not items:
            return "(no user variables)"
        return "\n".join(items)

    def execute(self, code: str) -> ExecResult:
        """Execute code in the persistent namespace, capturing stdout/stderr.

        Import statements are executed first so they land in globals.
        If the last statement is a bare expression its value is auto-printed
        (like a REPL / Jupyter cell).
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        self.last_final_answer = None

        with self._lock:
            try:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    tree = ast.parse(code)

                    # --- separate imports ---
                    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
                    rest = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]

                    if imports:
                        import_tree = ast.Module(body=imports, type_ignores=[])
                        exec(compile(import_tree, "<repl>", "exec"), self.namespace, self.namespace)

                    # --- auto-eval last expression ---
                    if rest and isinstance(rest[-1], ast.Expr):
                        last_expr = rest.pop()
                        if rest:
                            mod = ast.Module(body=rest, type_ignores=[])
                            exec(compile(mod, "<repl>", "exec"), self.namespace, self.namespace)
                        result = eval(
                            compile(ast.Expression(last_expr.value), "<repl>", "eval"),
                            self.namespace,
                            self.namespace,
                        )
                        if result is not None:
                            stdout_buf.write(repr(result) + "\n")
                    elif rest:
                        mod = ast.Module(body=rest, type_ignores=[])
                        exec(compile(mod, "<repl>", "exec"), self.namespace, self.namespace)
            except Exception:
                stderr_buf.write(traceback.format_exc())

        return ExecResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
        )


# ---------------------------------------------------------------------------
# Coding engine
# ---------------------------------------------------------------------------

_MAX_OUTPUT_LEN = 20000


class CodingEngine:
    """Iterative coding engine: LLM generates code, executor runs it, loop until FINAL."""

    def __init__(
        self,
        llm: LLMClient,
        system_prompt: str,
        working_directory: str,
        max_iterations: int = 30,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self._executor = CodeExecutor(working_directory)
        self._message_history: list[dict] = []

    def _llm_call(
        self,
        on_token: Callable[[str], None] | None = None,
        extra_messages: list[dict] | None = None,
    ) -> str:
        """Call the LLM, using streaming when on_token is provided and supported.

        *extra_messages* are appended for this call only (not persisted).
        """
        messages = self._message_history
        if extra_messages:
            messages = self._message_history + extra_messages
        if on_token and hasattr(self.llm, "stream_completion"):
            return self.llm.stream_completion(messages, on_token=on_token)
        return self.llm.completion(messages)

    def run(
        self,
        user_prompt: str,
        on_iteration: Callable[[Iteration], None] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> EngineResult:
        """Run the coding loop for a user prompt. Returns EngineResult."""
        # Initialize message history on first run
        if not self._message_history:
            self._message_history.append({"role": "system", "content": self.system_prompt})

        self._message_history.append({"role": "user", "content": user_prompt})
        iterations: list[Iteration] = []

        for i in range(1, self.max_iterations + 1):
            # Stage-aware hint (not persisted into history)
            if i == 1:
                hint = [{"role": "user", "content": "You have not interacted with the REPL yet. Start by understanding the problem. Do not jump to a final answer prematurely."}]
            else:
                hint = [{"role": "user", "content": "Above is your previous REPL interaction history. Continue working towards solving the problem."}]
            response = self._llm_call(on_token, extra_messages=hint)
            self._message_history.append({"role": "assistant", "content": response})

            code_blocks = parse_code_blocks(response)
            if not code_blocks:
                code_blocks = parse_tool_calls(response)

            # No code blocks → check for FINAL or return as plain-text answer
            if not code_blocks:
                iteration = Iteration(
                    iteration=i,
                    response=response,
                    code_blocks=[],
                )
                iterations.append(iteration)
                if on_iteration:
                    on_iteration(iteration)
                answer = parse_final_answer(response, self._executor)
                if answer is not None:
                    return EngineResult(answer=answer, iterations=iterations)
                # Pure conversational reply — no need to keep iterating
                return EngineResult(answer=response, iterations=iterations)

            block_results: list[CodeBlockResult] = []

            for code in code_blocks:
                result = self._executor.execute(code)
                block_results.append(CodeBlockResult(
                    code=code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                ))

                # Format output and feed back as user message
                output = self._format_output(code, result)
                self._message_history.append({"role": "user", "content": output})

            iteration = Iteration(
                iteration=i,
                response=response,
                code_blocks=block_results,
            )
            iterations.append(iteration)
            if on_iteration:
                on_iteration(iteration)

            # Check for final answer
            answer = parse_final_answer(response, self._executor)
            if answer is not None:
                return EngineResult(answer=answer, iterations=iterations)

        # Exceeded max iterations — ask LLM to wrap up
        self._message_history.append({
            "role": "user",
            "content": "You have reached the maximum number of iterations. Please provide your final answer now using FINAL(your answer here).",
        })
        response = self._llm_call(on_token)
        self._message_history.append({"role": "assistant", "content": response})

        iterations.append(Iteration(
            iteration=self.max_iterations + 1,
            response=response,
            code_blocks=[],
        ))

        answer = parse_final_answer(response, self._executor)
        if answer is not None:
            return EngineResult(answer=answer, iterations=iterations)

        # Fallback: use the last response as the answer
        return EngineResult(answer=response, iterations=iterations)

    def _format_output(self, code: str, result: ExecResult) -> str:
        parts = [f"Code:\n```python\n{code}\n```\nOutput:"]
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n"
            output += f"[stderr]\n{result.stderr}"
        if not output:
            output = "(no output)"
        if len(output) > _MAX_OUTPUT_LEN:
            output = output[:_MAX_OUTPUT_LEN] + "\n... (truncated)"
        parts.append(output)
        return "\n".join(parts)
