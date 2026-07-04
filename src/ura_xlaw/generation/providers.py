"""LLM provider strategies used by the QA generation service."""

from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    """Synchronous text-generation strategy."""

    def generate(self, prompt: str, model: str) -> str: ...


class AsyncLLMProvider(Protocol):
    """Asynchronous text-generation strategy."""

    async def generate(self, prompt: str, model: str) -> str: ...

    async def close(self) -> None: ...


class OpenAIProvider:
    def __init__(self, api_key: str):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("Install the 'openai' package to use OpenAI.") from exc
        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, model: str) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return response.choices[0].message.content or ""


class AsyncOpenAIProvider:
    def __init__(self, api_key: str):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("Install the 'openai' package to use OpenAI.") from exc
        self._client = AsyncOpenAI(api_key=api_key)

    async def generate(self, prompt: str, model: str) -> str:
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    async def close(self) -> None:
        await self._client.close()


class GeminiProvider:
    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "Install the 'google-generativeai' package to use Gemini."
            ) from exc
        genai.configure(api_key=api_key)
        self._genai = genai

    def generate(self, prompt: str, model: str) -> str:
        client = self._genai.GenerativeModel(model)
        response = client.generate_content(
            prompt,
            generation_config=self._genai.GenerationConfig(
                response_mime_type="application/json"
            ),
        )
        return response.text


def create_provider(name: str, api_key: str) -> LLMProvider:
    providers = {"openai": OpenAIProvider, "gemini": GeminiProvider}
    try:
        return providers[name.lower()](api_key)
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {name}") from exc
