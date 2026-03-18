# RLM Code

An interactive coding CLI powered by LLMs. Think Claude Code / Cursor — the LLM writes Python code in `` ```repl``` `` blocks to call filesystem tools, inspect results, iterate, and converge on an answer.

## How it works

RLM Code provides 6 filesystem tools (`bash`, `read_file`, `write_file`, `edit_file`, `glob_files`, `grep_files`) in a persistent Python execution environment. The LLM doesn't call tools via a structured function-call API — instead it writes Python code that invokes the tool functions directly, reads `print()` output, and decides what to do next. This loop continues until the LLM emits `FINAL(...)` with a summary.

```
User prompt
    ↓
System prompt (role + tools + examples + guidelines + termination protocol)
    ↓
┌─ Iteration loop (up to 30 iterations) ────────────────────┐
│  1. Stage hint injected (not persisted in history)         │
│       ↓                                                    │
│  2. LLM generates text + ```repl``` code blocks           │
│       ↓                                                    │
│  3. CodeExecutor runs code (AST-parsed, REPL-style)       │
│       ↓                                                    │
│  4. stdout/stderr fed back as user message                 │
│       ↓                                                    │
│  5. Check for FINAL(...) → done, else continue loop       │
└────────────────────────────────────────────────────────────┘
    ↓
Final answer returned to user
```

### Key design decisions

| Decision | Why |
|----------|-----|
| Python code blocks instead of JSON function-calling | LLM can compose tool calls, use conditionals, string processing, loops |
| Execution results fed back as user messages | Natural conversation flow — LLM writes code, "user" reports results |
| Persistent namespace across iterations | Variables survive across rounds, enabling incremental computation |
| Stage hints not persisted in history | Guide LLM behavior without polluting context |
| Streaming preview + final Markdown rendering | Stream with plain text (fast), final render with Markdown (polished) |
| Tool-call markup fallback parsing | Gracefully handles fine-tuned models that emit raw function-call tokens |

## Installation

```bash
# From the project directory
pip install -e .
```

Requires Python >= 3.11.

## Configuration

Set API keys, model, and base URL via environment variables or a `.env` file (auto-loaded via python-dotenv).

See `.env.example` for a template.

## Usage

### Interactive TUI mode (default)

```bash
rlm-code
```

Launches a Textual TUI with a Tokyo Night color scheme. Type requests in the input bar, watch the LLM reason and call tools in real-time with streaming output.

```
┌─ RLM Code ─────────────────── gpt-5 ─── ~/Projects ─┐
│                                                        │
│  > fix the off-by-one in parser.py                     │
│                                                        │
│  ◆ Iteration 1                                         │
│    Let me read the file first.                         │
│    ┌─ repl ─────────────────────────┐                  │
│    │ content = read_file("parser.py")│                  │
│    │ print(content)                  │                  │
│    └────────────────────────────────┘                  │
│    stdout: [file contents...]                          │
│                                                        │
│  ◆ Iteration 2                                         │
│    Found the bug on line 42. Fixing...                 │
│    ┌─ repl ─────────────────────────┐                  │
│    │ edit_file("parser.py", ...)     │                  │
│    └────────────────────────────────┘                  │
│    stdout: Replaced 1 lines with 1 lines in parser.py │
│                                                        │
│  ✓ Fixed the off-by-one error in parser.py line 42.    │
│                                                        │
│  > _                                                   │
└────────────────────────────────────────────────────────┘
```

Keyboard shortcuts:
- **Enter**: Submit request
- **Up/Down**: Input history
- **Ctrl+C**: Interrupt current operation
- **Ctrl+D** or type `quit`/`exit`: Exit

### One-shot mode

Pass a prompt as a positional argument to skip the TUI and print results to stdout:

```bash
rlm-code "list all Python files in this project"
rlm-code "explain what src/utils.py does"
rlm-code "add type hints to the parse() function in parser.py"
```

### CLI options

```
rlm-code [prompt] [options]

positional arguments:
  prompt                  One-shot prompt (omit for interactive TUI mode)

options:
  --backend {openai,anthropic}
                          LLM backend (default: openai, env: RLM_BACKEND)
  --model MODEL           Model name (default: gpt-5 / claude-sonnet-4-6, env: RLM_MODEL)
  --base-url URL          API base URL (env: RLM_BASE_URL). For OpenAI-compatible providers.
  --api-key API_KEY       API key (default: from env vars)
  --max-iterations N      Max iterations per request (default: 30)
  --cwd PATH              Working directory for tools (default: current dir)
```

Priority: CLI args > environment variables > defaults.

### Examples

```bash
# Use Anthropic backend
rlm-code --backend anthropic --model claude-sonnet-4-20250514

# Use an OpenAI-compatible provider (e.g. local vLLM, OpenRouter, etc.)
rlm-code --base-url http://localhost:8000/v1 --model my-local-model

# Or configure entirely via .env
#   RLM_BASE_URL=https://openrouter.ai/api/v1
#   RLM_MODEL=deepseek/deepseek-chat
#   OPENAI_API_KEY=sk-or-...
rlm-code

# Work on a specific directory
rlm-code --cwd ~/Projects/myapp

# Quick one-shot with a custom model
rlm-code --model gpt-4o-mini "what does this project do?"
```

## Tools

The LLM has access to 6 filesystem tools (in `rlm_code/tools.py`):

| Tool | Signature | Description |
|------|-----------|-------------|
| `bash` | `(command, timeout=120)` | Execute a shell command. Returns stdout + stderr + exit code. |
| `read_file` | `(path, offset=0, limit=2000)` | Read a file with line numbers. Supports pagination via offset/limit. |
| `write_file` | `(path, content)` | Write content to a file. Creates parent directories automatically. |
| `edit_file` | `(path, old_text, new_text)` | Replace first exact occurrence of `old_text` with `new_text`. |
| `glob_files` | `(pattern, path=None)` | Find files matching a glob pattern. Skips `.git`, `node_modules`, `__pycache__`, etc. Max 200 results. |
| `grep_files` | `(pattern, path=None, glob=None, context=0)` | Search file contents with regex (`grep -rnE`). Max 500 results. 30s timeout. |

All paths are resolved relative to the working directory (`--cwd`). The tools run in the user's filesystem — **there is no sandbox**. The LLM can read, write, and execute anything you can.

## Project structure

```
rlm_code/
├── __init__.py       # exports main()
├── __main__.py       # python -m rlm_code entry point
├── config.py         # CLIConfig dataclass + argparse
├── engine.py         # CodingEngine: iteration loop, CodeExecutor, parsing
├── llm.py            # LLMClient: OpenAI + Anthropic backends (sync + streaming)
├── prompt.py         # System prompt template + build_system_prompt()
├── tools.py          # 6 tool functions + build_coding_tools()
└── tui.py            # Textual TUI app + one-shot mode + main()

tests/
├── test_engine.py    # 44 tests: engine loop, executor, parsing, streaming, hints
└── test_tools.py     # 21 tests: all tools (no LLM, no network)
```

## Architecture

### Engine (`engine.py`)

- **CodingEngine**: The core iteration loop. Calls the LLM with stage-aware hints (iteration 1: "understand the problem first"; subsequent: "continue working"), parses `` ```repl``` `` code blocks, executes them via `CodeExecutor`, feeds output back as user messages, and loops until `FINAL(...)` or max iterations. When max iterations is exceeded, a forced wrap-up prompt is sent.
- **CodeExecutor**: Persistent Python namespace. Code is AST-parsed: imports separated and executed first; if the last statement is a bare expression its value is auto-printed (REPL/Jupyter-style). Execution protected by `threading.Lock`. Dangerous builtins (`eval`, `exec`, `input`, `compile`, `globals`, `locals`, `exit`, `quit`) are blocked. Output truncated at 20,000 characters.
- **Parsing**: `parse_code_blocks()` extracts `` ```repl``` `` fences. `parse_tool_calls()` is a fallback that converts raw function-calling markup from fine-tuned models into executable Python. `parse_final_answer()` detects `FINAL(...)` and `FINAL_VAR(...)` completion signals.

### LLM Client (`llm.py`)

- Unified `LLMClient` with `completion()` (sync) and `stream_completion()` (streaming with `on_token` callback).
- **OpenAI backend**: Supports `base_url` for compatible providers (vLLM, OpenRouter, etc.). SDKs are lazily imported on first call.
- **Anthropic backend**: Automatically extracts the system message from the message list and passes it separately (as required by the Anthropic API).

### TUI (`tui.py`)

- Built with [Textual](https://textual.textualize.io/) (full-terminal UI framework).
- **Tokyo Night** color scheme: `#1a1b26` background, `#7aa2f7` user text, `#bb9af7` iteration headers, `#9ece6a` final answers, `#f7768e` errors, `#565f89` status.
- **Streaming**: LLM tokens stream to a plain-text `MessageWidget` in real-time via `call_from_thread()`. When each iteration completes, the streaming preview is removed and replaced with a properly rendered `Markdown` widget + syntax-highlighted code blocks.
- Engine `run()` executes in a background thread (`@work(thread=True)`).

## Testing

```bash
pytest tests/ -v
```

65 tests total. All use `tmp_path` fixtures — no LLM calls, no network, no side effects on your filesystem. Engine tests use `MockLLM` / `MockStreamingLLM` / `RecordingLLM` classes.

## License

MIT
