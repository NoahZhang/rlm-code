"""CLI configuration: argument parsing and environment variable handling."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


@dataclass
class CLIConfig:
    backend: str
    model: str
    sub_model: str
    api_key: str | None
    base_url: str | None
    max_iterations: int
    cwd: str
    prompt: str | None  # None = interactive TUI mode


def parse_args(argv: list[str] | None = None) -> CLIConfig:
    parser = argparse.ArgumentParser(
        prog="rlm-code",
        description="Interactive coding CLI",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="One-shot prompt (omit for interactive TUI mode)",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("RLM_BACKEND", "openai"),
        choices=["openai", "anthropic"],
        help="LLM backend (default: openai, env: RLM_BACKEND)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("RLM_MODEL"),
        help="Model name (default: gpt-4o for openai, claude-sonnet-4-20250514 for anthropic, env: RLM_MODEL)",
    )
    parser.add_argument(
        "--sub-model",
        default=os.environ.get("RLM_SUB_MODEL"),
        help="Sub-model for llm_query tool (default: same as --model, env: RLM_SUB_MODEL)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RLM_BASE_URL"),
        help="Base URL for the API (env: RLM_BASE_URL). Useful for OpenAI-compatible providers.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: from OPENAI_API_KEY or ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Max iterations per request (default: 30)",
    )
    parser.add_argument(
        "--cwd",
        default=os.getcwd(),
        help="Working directory (default: current directory)",
    )
    args = parser.parse_args(argv)

    # Resolve model defaults based on backend
    if args.model is None:
        args.model = "gpt-4o" if args.backend == "openai" else "claude-sonnet-4-20250514"
    if args.sub_model is None:
        args.sub_model = args.model

    # Resolve API key: CLI arg > env var
    api_key = args.api_key
    if api_key is None:
        if args.backend == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

    return CLIConfig(
        backend=args.backend,
        model=args.model,
        sub_model=args.sub_model,
        api_key=api_key,
        base_url=args.base_url,
        max_iterations=args.max_iterations,
        cwd=os.path.abspath(args.cwd),
        prompt=args.prompt,
    )
