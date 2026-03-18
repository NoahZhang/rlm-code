"""LLM client abstraction for OpenAI and Anthropic backends."""

from __future__ import annotations

from collections.abc import Callable


class LLMClient:
    """Unified LLM client supporting OpenAI and Anthropic APIs."""

    def __init__(
        self,
        backend: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def completion(self, messages: list[dict]) -> str:
        """Send messages to the LLM and return the response text."""
        if self.backend == "anthropic":
            return self._anthropic_completion(messages)
        return self._openai_completion(messages)

    def stream_completion(
        self,
        messages: list[dict],
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Send messages to the LLM and return the response text, streaming tokens via on_token."""
        if self.backend == "anthropic":
            return self._anthropic_stream(messages, on_token)
        return self._openai_stream(messages, on_token)

    def _openai_completion(self, messages: list[dict]) -> str:
        import openai

        kwargs: dict = {"model": self.model}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = openai.OpenAI(**{k: v for k, v in kwargs.items() if k != "model"})
        response = client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content or ""

    def _anthropic_completion(self, messages: list[dict]) -> str:
        import anthropic

        kwargs: dict = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        client = anthropic.Anthropic(**kwargs)

        # Extract system message from the message list
        system = ""
        filtered: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered.append(msg)

        create_kwargs: dict = {
            "model": self.model,
            "max_tokens": 32768,
            "messages": filtered,
        }
        if system:
            create_kwargs["system"] = system

        response = client.messages.create(**create_kwargs)
        return response.content[0].text

    def _openai_stream(
        self, messages: list[dict], on_token: Callable[[str], None] | None,
    ) -> str:
        import openai

        kwargs: dict = {"model": self.model}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = openai.OpenAI(**{k: v for k, v in kwargs.items() if k != "model"})

        chunks: list[str] = []
        stream = client.chat.completions.create(
            model=self.model, messages=messages, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                chunks.append(delta)
                if on_token:
                    on_token(delta)
        return "".join(chunks)

    def _anthropic_stream(
        self, messages: list[dict], on_token: Callable[[str], None] | None,
    ) -> str:
        import anthropic

        kwargs: dict = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        client = anthropic.Anthropic(**kwargs)

        system = ""
        filtered: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered.append(msg)

        create_kwargs: dict = {
            "model": self.model,
            "max_tokens": 32768,
            "messages": filtered,
        }
        if system:
            create_kwargs["system"] = system

        chunks: list[str] = []
        with client.messages.stream(**create_kwargs) as stream:
            for text in stream.text_stream:
                chunks.append(text)
                if on_token:
                    on_token(text)
        return "".join(chunks)
