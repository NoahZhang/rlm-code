"""Unit tests for engine components (no LLM calls required)."""

from __future__ import annotations

import os

import pytest

from rlm_code.engine import (
    CodeExecutor,
    CodingEngine,
    EngineResult,
    Iteration,
    parse_code_blocks,
    parse_final_answer,
    parse_tool_calls,
)


# ---------------------------------------------------------------------------
# parse_code_blocks
# ---------------------------------------------------------------------------


class TestParseCodeBlocks:
    def test_single_block(self):
        text = "Some text\n```repl\nprint('hello')\n```\nMore text"
        blocks = parse_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0] == "print('hello')"

    def test_multiple_blocks(self):
        text = "```repl\nx = 1\n```\ntext\n```repl\ny = 2\n```"
        blocks = parse_code_blocks(text)
        assert len(blocks) == 2
        assert blocks[0] == "x = 1"
        assert blocks[1] == "y = 2"

    def test_no_blocks(self):
        text = "Just some text with no code blocks"
        blocks = parse_code_blocks(text)
        assert blocks == []


# ---------------------------------------------------------------------------
# parse_tool_calls
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    def test_single_tool_call(self):
        text = '<|tool_calls_section_begin|><|tool_call_begin|>functions.glob_files:0<|tool_call_argument_begin|>{"pattern": "**/*.py"}<|tool_call_end|><|tool_calls_section_end|>'
        blocks = parse_tool_calls(text)
        assert len(blocks) == 1
        assert "glob_files" in blocks[0]
        assert "**/*.py" in blocks[0]

    def test_multiple_tool_calls(self):
        text = (
            '<|tool_calls_section_begin|>'
            '<|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>{"path": "main.py"}<|tool_call_end|>'
            '<|tool_call_begin|>functions.grep_files:1<|tool_call_argument_begin|>{"pattern": "TODO", "glob": "*.py"}<|tool_call_end|>'
            '<|tool_calls_section_end|>'
        )
        blocks = parse_tool_calls(text)
        assert len(blocks) == 2

    def test_no_tool_calls(self):
        text = "Just some normal text"
        blocks = parse_tool_calls(text)
        assert blocks == []

    def test_invalid_json_skipped(self):
        text = '<|tool_call_begin|>functions.bash:0<|tool_call_argument_begin|>{bad json}<|tool_call_end|>'
        blocks = parse_tool_calls(text)
        assert blocks == []

    def test_non_repl_blocks_ignored(self):
        text = "```python\nprint('hello')\n```\n```repl\nx = 1\n```"
        blocks = parse_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0] == "x = 1"

    def test_multiline_code(self):
        text = "```repl\nfor i in range(3):\n    print(i)\n```"
        blocks = parse_code_blocks(text)
        assert len(blocks) == 1
        assert "for i in range(3):" in blocks[0]
        assert "    print(i)" in blocks[0]


# ---------------------------------------------------------------------------
# parse_final_answer
# ---------------------------------------------------------------------------


class TestParseFinalAnswer:
    def test_final_simple(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        text = "I'm done.\nFINAL(The task is complete.)"
        result = parse_final_answer(text, executor)
        assert result == "The task is complete."

    def test_final_single_line_capture(self, tmp_path):
        """Non-greedy regex should capture only up to the first closing paren."""
        executor = CodeExecutor(str(tmp_path))
        text = "FINAL(The answer is 42)"
        result = parse_final_answer(text, executor)
        assert result == "The answer is 42"

    def test_final_no_over_match(self, tmp_path):
        """Greedy+DOTALL used to match across lines — verify it no longer does."""
        executor = CodeExecutor(str(tmp_path))
        text = "FINAL(answer)\nsome text with )"
        result = parse_final_answer(text, executor)
        assert result == "answer"

    def test_final_var_from_namespace(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        executor.namespace["my_result"] = "computed value"
        text = "FINAL_VAR(my_result)"
        result = parse_final_answer(text, executor)
        assert result == "computed value"

    def test_final_var_from_execution(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        executor.namespace["answer"] = 42
        executor.execute("FINAL_VAR('answer')")
        result = parse_final_answer("some response", executor)
        assert result == "42"

    def test_no_final(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        text = "I need to keep working on this."
        result = parse_final_answer(text, executor)
        assert result is None


# ---------------------------------------------------------------------------
# CodeExecutor
# ---------------------------------------------------------------------------


class TestCodeExecutor:
    def test_print_output(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        result = executor.execute("print('hello world')")
        assert result.stdout.strip() == "hello world"
        assert result.stderr == ""

    def test_exception_stderr(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        result = executor.execute("1/0")
        assert result.stdout == ""
        assert "ZeroDivisionError" in result.stderr

    def test_persistent_namespace(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        executor.execute("x = 42")
        result = executor.execute("print(x)")
        assert "42" in result.stdout

    def test_tools_available(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        # All 6 tools should be in namespace
        for name in ["bash", "read_file", "write_file", "edit_file", "glob_files", "grep_files"]:
            assert name in executor.namespace
            assert callable(executor.namespace[name])

    def test_show_vars(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        executor.execute("x = 10")
        executor.execute("name = 'test'")
        result = executor.execute("print(SHOW_VARS())")
        assert "x = 10" in result.stdout
        assert "name = 'test'" in result.stdout

    def test_final_var(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        executor.execute("answer = 'the result'")
        executor.execute("FINAL_VAR('answer')")
        assert executor.last_final_answer == "the result"

    def test_blocked_builtins(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        result = executor.execute("exit()")
        assert result.stderr  # Should error because exit is blocked

    def test_blocked_eval_exec(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        for name in ("eval", "exec", "input", "compile", "globals", "locals"):
            result = executor.execute(f"{name}('1')")
            assert result.stderr, f"{name} should be blocked"

    # -- auto-eval (REPL-like behaviour) --

    def test_auto_eval_expression(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        result = executor.execute("2 + 2")
        assert "4" in result.stdout

    def test_auto_eval_after_statements(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        result = executor.execute("x = 5\nx * 2")
        assert "10" in result.stdout

    def test_auto_eval_assignment_no_output(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        result = executor.execute("x = 5")
        assert result.stdout == ""

    def test_auto_eval_print_no_duplicate(self, tmp_path):
        """print() returns None — auto-eval should NOT output 'None'."""
        executor = CodeExecutor(str(tmp_path))
        result = executor.execute("print('hi')")
        assert result.stdout.strip() == "hi"

    # -- import separation --

    def test_import_separation(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        result = executor.execute("import math\nmath.sqrt(16)")
        assert "4.0" in result.stdout

    def test_write_and_read_file(self, tmp_path):
        executor = CodeExecutor(str(tmp_path))
        result = executor.execute("result = write_file('test.txt', 'hello')\nprint(result)")
        assert "Successfully wrote" in result.stdout
        result = executor.execute("content = read_file('test.txt')\nprint(content)")
        assert "hello" in result.stdout


# ---------------------------------------------------------------------------
# CodingEngine (with mock LLM)
# ---------------------------------------------------------------------------


class MockLLM:
    """Mock LLM that returns pre-configured responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def completion(self, messages: list[dict]) -> str:
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
        else:
            response = "FINAL(fallback answer)"
        self._call_count += 1
        return response


class MockStreamingLLM(MockLLM):
    """Mock LLM with stream_completion that yields tokens character by character."""

    def stream_completion(
        self, messages: list[dict], on_token=None,
    ) -> str:
        full = self.completion(messages)
        if on_token:
            for ch in full:
                on_token(ch)
        return full


class TestCodingEngine:
    def test_immediate_final(self, tmp_path):
        llm = MockLLM(["I know the answer.\nFINAL(42)"])
        engine = CodingEngine(
            llm=llm,
            system_prompt="You are a test assistant.",
            working_directory=str(tmp_path),
            max_iterations=5,
        )
        result = engine.run("What is the answer?")
        assert result.answer == "42"
        assert len(result.iterations) == 1

    def test_code_then_final(self, tmp_path):
        llm = MockLLM([
            "Let me compute.\n```repl\nx = 2 + 2\nprint(x)\n```",
            "The answer is 4.\nFINAL(The result is 4.)",
        ])
        engine = CodingEngine(
            llm=llm,
            system_prompt="You are a test assistant.",
            working_directory=str(tmp_path),
            max_iterations=5,
        )
        result = engine.run("What is 2+2?")
        assert result.answer == "The result is 4."
        assert len(result.iterations) == 2
        assert len(result.iterations[0].code_blocks) == 1
        assert "4" in result.iterations[0].code_blocks[0].stdout

    def test_max_iterations_exceeded(self, tmp_path):
        # LLM never returns FINAL
        llm = MockLLM([
            "```repl\nprint('working')\n```",
            "```repl\nprint('still working')\n```",
            "```repl\nprint('more work')\n```",
            "FINAL(gave up)",
        ])
        engine = CodingEngine(
            llm=llm,
            system_prompt="You are a test assistant.",
            working_directory=str(tmp_path),
            max_iterations=3,
        )
        result = engine.run("Do something")
        assert result.answer == "gave up"

    def test_persistent_across_runs(self, tmp_path):
        llm = MockLLM([
            "```repl\nx = 100\nprint('set x')\n```",
            "FINAL(done setting x)",
            "```repl\nprint(f'x is {x}')\n```",
            "FINAL(x was preserved)",
        ])
        engine = CodingEngine(
            llm=llm,
            system_prompt="You are a test assistant.",
            working_directory=str(tmp_path),
            max_iterations=5,
        )
        result1 = engine.run("set x = 100")
        assert result1.answer == "done setting x"

        result2 = engine.run("what is x?")
        assert result2.answer == "x was preserved"
        # Verify x is still in executor namespace
        assert engine._executor.namespace.get("x") == 100

    def test_no_code_blocks_returns_immediately(self, tmp_path):
        """When LLM replies without code blocks or FINAL, return the reply in 1 iteration."""
        llm = MockLLM(["你好！有什么可以帮你的？"])
        engine = CodingEngine(
            llm=llm,
            system_prompt="You are a test assistant.",
            working_directory=str(tmp_path),
            max_iterations=10,
        )
        result = engine.run("你好")
        assert result.answer == "你好！有什么可以帮你的？"
        assert len(result.iterations) == 1
        assert llm._call_count == 1

    def test_tool_call_markup_fallback(self, tmp_path):
        """When LLM outputs raw tool-call markup, engine should convert and execute it."""
        llm = MockLLM([
            '<|tool_calls_section_begin|><|tool_call_begin|>functions.bash:0<|tool_call_argument_begin|>{"command": "echo hello"}<|tool_call_end|><|tool_calls_section_end|>',
            "FINAL(done)",
        ])
        engine = CodingEngine(
            llm=llm,
            system_prompt="test",
            working_directory=str(tmp_path),
            max_iterations=5,
        )
        result = engine.run("say hello")
        assert result.answer == "done"
        # First iteration should have executed the bash command
        assert len(result.iterations[0].code_blocks) == 1
        assert "hello" in result.iterations[0].code_blocks[0].stdout

    def test_multiple_code_blocks_in_one_response(self, tmp_path):
        llm = MockLLM([
            "```repl\na = 1\nprint(a)\n```\nNow another:\n```repl\nb = 2\nprint(b)\n```",
            "FINAL(done)",
        ])
        engine = CodingEngine(
            llm=llm,
            system_prompt="You are a test assistant.",
            working_directory=str(tmp_path),
            max_iterations=5,
        )
        result = engine.run("do it")
        assert result.answer == "done"
        assert len(result.iterations[0].code_blocks) == 2
        assert "1" in result.iterations[0].code_blocks[0].stdout
        assert "2" in result.iterations[0].code_blocks[1].stdout


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


class TestStreamingEngine:
    def test_streaming_callback_receives_tokens(self, tmp_path):
        """on_token should be called with every character of the LLM response."""
        llm = MockStreamingLLM(["Hello!\nFINAL(hi)"])
        engine = CodingEngine(
            llm=llm,
            system_prompt="test",
            working_directory=str(tmp_path),
            max_iterations=5,
        )
        tokens: list[str] = []
        result = engine.run("hi", on_token=lambda t: tokens.append(t))
        assert result.answer == "hi"
        assert "".join(tokens) == "Hello!\nFINAL(hi)"

    def test_streaming_with_code_and_final(self, tmp_path):
        """Streaming should work across multiple iterations with code blocks."""
        llm = MockStreamingLLM([
            "Let me compute.\n```repl\nprint(1+1)\n```",
            "FINAL(2)",
        ])
        engine = CodingEngine(
            llm=llm,
            system_prompt="test",
            working_directory=str(tmp_path),
            max_iterations=5,
        )
        tokens: list[str] = []
        iterations: list[Iteration] = []
        result = engine.run(
            "compute",
            on_iteration=lambda it: iterations.append(it),
            on_token=lambda t: tokens.append(t),
        )
        assert result.answer == "2"
        assert len(iterations) == 2
        # Tokens should contain characters from both LLM calls
        full = "".join(tokens)
        assert "Let me compute" in full
        assert "FINAL(2)" in full

    def test_non_streaming_llm_fallback(self, tmp_path):
        """When LLM has no stream_completion, on_token is ignored gracefully."""
        llm = MockLLM(["FINAL(works)"])
        engine = CodingEngine(
            llm=llm,
            system_prompt="test",
            working_directory=str(tmp_path),
            max_iterations=5,
        )
        tokens: list[str] = []
        result = engine.run("hi", on_token=lambda t: tokens.append(t))
        assert result.answer == "works"
        # MockLLM has no stream_completion, so no tokens should be received
        assert tokens == []


# ---------------------------------------------------------------------------
# Iteration stage hints
# ---------------------------------------------------------------------------


class RecordingLLM:
    """LLM that records the messages it receives for each call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.recorded_messages: list[list[dict]] = []

    def completion(self, messages: list[dict]) -> str:
        self.recorded_messages.append(list(messages))
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
        else:
            response = "FINAL(fallback)"
        self._call_count += 1
        return response


class TestIterationHints:
    def test_first_iteration_hint(self, tmp_path):
        llm = RecordingLLM(["FINAL(done)"])
        engine = CodingEngine(
            llm=llm,
            system_prompt="sys",
            working_directory=str(tmp_path),
            max_iterations=5,
        )
        engine.run("hello")
        # First call should have a hint about not having interacted yet
        msgs = llm.recorded_messages[0]
        last_content = msgs[-1]["content"]
        assert "not interacted" in last_content.lower() or "have not" in last_content.lower()

    def test_subsequent_iteration_hint(self, tmp_path):
        llm = RecordingLLM([
            "```repl\nprint('hi')\n```",
            "FINAL(done)",
        ])
        engine = CodingEngine(
            llm=llm,
            system_prompt="sys",
            working_directory=str(tmp_path),
            max_iterations=5,
        )
        engine.run("do something")
        # Second call should have a hint about previous interaction history
        msgs = llm.recorded_messages[1]
        last_content = msgs[-1]["content"]
        assert "previous" in last_content.lower() or "history" in last_content.lower()
