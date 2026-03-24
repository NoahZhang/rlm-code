"""Textual TUI application for RLM Code."""

from __future__ import annotations

import sys

from dotenv import load_dotenv
from rich.syntax import Syntax
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.css.query import NoMatches
from textual.widgets import Footer, Header, Input, Markdown, Static

from .config import CLIConfig, parse_args
from .engine import CodingEngine, EngineResult, Iteration
from .llm import LLMClient
from .prompt import build_system_prompt
from .tools import build_coding_tools


# ---------------------------------------------------------------------------
# Custom widgets
# ---------------------------------------------------------------------------


class MessageWidget(Static):
    """A single message in the conversation."""


# ---------------------------------------------------------------------------
# TUI App
# ---------------------------------------------------------------------------

APP_CSS = """
Screen {
    background: #1a1b26;
}

#conversation {
    height: 1fr;
    overflow-y: auto;
    padding: 1 2;
    background: #1a1b26;
}

.user-msg {
    color: #7aa2f7;
    margin-bottom: 1;
}

.iteration-header {
    color: #bb9af7;
    text-style: bold;
    margin-top: 1;
}

.llm-text {
    color: #c0caf5;
    margin-bottom: 0;
}

MarkdownH1 {
    color: #bb9af7;
    margin: 1 0 0 0;
}

MarkdownH2 {
    color: #7aa2f7;
    margin: 1 0 0 0;
}

MarkdownH3 {
    color: #7dcfff;
    margin: 0;
}

MarkdownFence {
    margin: 0 2;
    margin-bottom: 1;
}

MarkdownBlockQuote {
    border-left: thick #7aa2f7;
    padding: 0 0 0 1;
}

.code-block {
    margin: 0 2;
    margin-bottom: 0;
}

.stdout-block {
    color: #565f89;
    margin: 0 2;
    margin-bottom: 1;
}

.final-answer {
    color: #9ece6a;
    text-style: bold;
    margin-top: 1;
    margin-bottom: 1;
}

.error-msg {
    color: #f7768e;
    margin-bottom: 1;
}

.status-msg {
    color: #565f89;
    text-style: italic;
    margin-bottom: 1;
}

#input-area {
    dock: bottom;
    height: auto;
    max-height: 5;
    padding: 0 2;
    background: #24283b;
}

#prompt-input {
    background: #24283b;
    color: #c0caf5;
    border: none;
}

#prompt-input:focus {
    border: none;
}

Header {
    background: #24283b;
    color: #7aa2f7;
}

Footer {
    background: #24283b;
    color: #565f89;
}
"""


class CodingApp(App):
    """RLM Code interactive TUI."""

    CSS = APP_CSS
    TITLE = "RLM Code"
    BINDINGS = [
        ("ctrl+c", "interrupt", "Interrupt"),
        ("ctrl+d", "quit", "Quit"),
    ]

    def __init__(self, config: CLIConfig) -> None:
        super().__init__()
        self.config = config
        self._engine: CodingEngine | None = None
        self._running = False
        self._input_history: list[str] = []
        self._history_index = -1
        self._streaming_widget: MessageWidget | None = None
        self._streaming_text: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield ScrollableContainer(id="conversation")
        yield Input(
            placeholder="Enter your request...",
            id="prompt-input",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"{self.config.model} — {self.config.cwd}"
        self._init_engine()
        self._append_status(f"Ready. Backend: {self.config.backend}, Model: {self.config.model}")
        self._append_status(f"Working directory: {self.config.cwd}")
        self.query_one("#prompt-input", Input).focus()

    def _init_engine(self) -> None:
        llm = LLMClient(
            backend=self.config.backend,
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        sub_llm = LLMClient(
            backend=self.config.backend,
            model=self.config.sub_model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        tools = build_coding_tools(self.config.cwd)
        system_prompt = build_system_prompt(tools)
        self._engine = CodingEngine(
            llm=llm,
            system_prompt=system_prompt,
            working_directory=self.config.cwd,
            max_iterations=self.config.max_iterations,
            sub_llm=sub_llm,
        )

    # -- UI helpers ----------------------------------------------------------

    def _append_widget(self, widget: Static) -> None:
        container = self.query_one("#conversation", ScrollableContainer)
        container.mount(widget)
        widget.scroll_visible()

    def _append_user(self, text: str) -> None:
        self._append_widget(MessageWidget(f"> {text}", classes="user-msg"))

    def _append_status(self, text: str) -> None:
        self._append_widget(MessageWidget(text, classes="status-msg"))

    def _append_llm_text(self, text: str) -> None:
        if text.strip():
            self._append_widget(Markdown(text, classes="llm-text"))

    def _append_code(self, code: str) -> None:
        syntax = Syntax(code, "python", theme="one-dark", line_numbers=False)
        self._append_widget(MessageWidget(syntax, classes="code-block"))

    def _append_stdout(self, text: str) -> None:
        if text.strip():
            # Truncate very long output
            lines = text.splitlines()
            if len(lines) > 50:
                text = "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more lines)"
            self._append_widget(MessageWidget(text, classes="stdout-block"))

    def _append_final(self, text: str) -> None:
        self._append_widget(MessageWidget(f"✓ {text}", classes="final-answer"))

    def _append_error(self, text: str) -> None:
        self._append_widget(MessageWidget(text, classes="error-msg"))

    def _begin_streaming(self) -> None:
        """Mount an empty widget to show streaming LLM output."""
        self._streaming_text = ""
        self._streaming_widget = MessageWidget("", classes="llm-text")
        self._append_widget(self._streaming_widget)

    def _append_token(self, token: str) -> None:
        """Append a token to the streaming preview widget."""
        if self._streaming_widget is None:
            self._begin_streaming()
        self._streaming_text += token
        self._streaming_widget.update(self._streaming_text)
        self._streaming_widget.scroll_visible()

    def _end_streaming(self) -> None:
        """Remove the streaming preview widget and reset state."""
        if self._streaming_widget is not None:
            self._streaming_widget.remove()
            self._streaming_widget = None
            self._streaming_text = ""

    def _set_input_enabled(self, enabled: bool) -> None:
        try:
            inp = self.query_one("#prompt-input", Input)
            inp.disabled = not enabled
            if enabled:
                inp.placeholder = "Enter your request..."
                inp.focus()
            else:
                inp.placeholder = "Running..."
        except NoMatches:
            pass

    # -- Render engine result ------------------------------------------------

    def _render_iteration(self, entry: Iteration) -> None:
        """Render a single Iteration into the conversation."""
        self._append_widget(
            MessageWidget(f"◆ Iteration {entry.iteration}", classes="iteration-header")
        )

        # LLM response text (strip code blocks since we render them separately)
        self._render_response_text(entry.response)

        # Code blocks and their results
        for block in entry.code_blocks:
            if block.code.strip():
                self._append_code(block.code)
            if block.stdout:
                self._append_stdout(block.stdout)
            if block.stderr:
                self._append_stdout(f"[stderr] {block.stderr}")

    def _render_response_text(self, response: str) -> None:
        """Extract and render text portions of the LLM response (outside code blocks)."""
        lines = response.split("\n")
        text_buf: list[str] = []
        in_code = False
        for line in lines:
            if line.strip().startswith("```repl"):
                if text_buf:
                    self._append_llm_text("\n".join(text_buf))
                    text_buf = []
                in_code = True
            elif line.strip() == "```" and in_code:
                in_code = False
            elif not in_code:
                # Skip FINAL() lines in display
                stripped = line.strip()
                if not stripped.startswith("FINAL(") and not stripped.startswith("FINAL_VAR("):
                    text_buf.append(line)
        if text_buf:
            joined = "\n".join(text_buf).strip()
            if joined:
                self._append_llm_text(joined)

    # -- Event handlers ------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        if text.lower() in ("quit", "exit"):
            self.exit()
            return

        self._input_history.append(text)
        self._history_index = -1
        self._append_user(text)
        self._run_completion(text)

    def on_key(self, event) -> None:
        if event.key == "up" and self._input_history:
            if self._history_index == -1:
                self._history_index = len(self._input_history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            try:
                inp = self.query_one("#prompt-input", Input)
                inp.value = self._input_history[self._history_index]
            except NoMatches:
                pass
        elif event.key == "down" and self._input_history:
            if self._history_index >= 0:
                self._history_index += 1
                if self._history_index >= len(self._input_history):
                    self._history_index = -1
            try:
                inp = self.query_one("#prompt-input", Input)
                inp.value = (
                    self._input_history[self._history_index]
                    if self._history_index >= 0
                    else ""
                )
            except NoMatches:
                pass

    @work(thread=True)
    def _run_completion(self, prompt: str) -> None:
        self.call_from_thread(self._set_input_enabled, False)
        self.call_from_thread(self._append_status, "Thinking...")

        def on_token(token: str) -> None:
            self.call_from_thread(self._append_token, token)

        def on_iteration(iteration: Iteration) -> None:
            self.call_from_thread(self._end_streaming)
            self.call_from_thread(self._render_iteration, iteration)

        try:
            result = self._engine.run(
                prompt, on_iteration=on_iteration, on_token=on_token,
            )
            self.call_from_thread(self._end_streaming)
            # Skip final banner when the answer is just the raw response
            # (conversational reply, already rendered by on_iteration)
            last_response = result.iterations[-1].response if result.iterations else None
            if result.answer != last_response:
                self.call_from_thread(self._append_final, result.answer)
        except KeyboardInterrupt:
            self.call_from_thread(self._end_streaming)
            self.call_from_thread(self._append_error, "Interrupted.")
        except Exception as e:
            self.call_from_thread(self._end_streaming)
            self.call_from_thread(self._append_error, f"Error: {e}")
        finally:
            self.call_from_thread(self._set_input_enabled, True)

    def action_interrupt(self) -> None:
        if self._running:
            self._append_error("Interrupt requested — the current operation may take a moment to stop.")

    def action_quit(self) -> None:
        self.exit()


# ---------------------------------------------------------------------------
# One-shot mode (no TUI)
# ---------------------------------------------------------------------------


def run_oneshot(config: CLIConfig) -> None:
    """Run a single prompt with stdout output (no TUI)."""
    llm = LLMClient(
        backend=config.backend,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )
    sub_llm = LLMClient(
        backend=config.backend,
        model=config.sub_model,
        api_key=config.api_key,
        base_url=config.base_url,
    )
    tools = build_coding_tools(config.cwd)
    system_prompt = build_system_prompt(tools)
    engine = CodingEngine(
        llm=llm,
        system_prompt=system_prompt,
        working_directory=config.cwd,
        max_iterations=config.max_iterations,
        sub_llm=sub_llm,
    )

    result = engine.run(config.prompt)

    # Print iterations for verbose output
    for entry in result.iterations:
        print(f"\n◆ Iteration {entry.iteration}")
        print(entry.response)
        for block in entry.code_blocks:
            if block.stdout:
                print(block.stdout)
            if block.stderr:
                print(f"[stderr] {block.stderr}")

    print(f"\n{'='*60}")
    print(f"Final answer:\n{result.answer}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    load_dotenv()
    config = parse_args()

    if config.prompt:
        run_oneshot(config)
    else:
        app = CodingApp(config)
        app.run()


if __name__ == "__main__":
    main()
