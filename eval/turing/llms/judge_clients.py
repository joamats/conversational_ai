"""Client adapters for multi-model Turing judges.

Claude uses httpx to call the Vertex AI rawPredict endpoint.
Gemini uses the google.genai SDK.
All adapters expose the same client.chat.completions.create() interface
as the OpenAI SDK, so judge_absolute() and judge_comparative() in
llm_judge.py work unchanged.
"""
from __future__ import annotations

import logging
from typing import Any

import google.auth
import google.auth.transport.requests
import httpx

logger = logging.getLogger(__name__)

_ANTHROPIC_LOCATION = "us-east5"
_REQUEST_TIMEOUT = 180.0


# ---------------------------------------------------------------------------
# Minimal response objects matching the OpenAI SDK shape
# ---------------------------------------------------------------------------

class _TokenDetails:
    __slots__ = ("reasoning_tokens", "cached_tokens")
    def __init__(self, reasoning_tokens: int = 0, cached_tokens: int = 0):
        self.reasoning_tokens = reasoning_tokens
        self.cached_tokens = cached_tokens


class _Usage:
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens",
                 "completion_tokens_details", "prompt_tokens_details")
    def __init__(self, input_tokens: int = 0, output_tokens: int = 0,
                 cache_read_input_tokens: int = 0):
        self.prompt_tokens = input_tokens
        self.completion_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens
        self.completion_tokens_details = _TokenDetails(reasoning_tokens=0)
        self.prompt_tokens_details = _TokenDetails(cached_tokens=cache_read_input_tokens)


class _Msg:
    __slots__ = ("content",)
    def __init__(self, content: str):
        self.content = content


class _Choice:
    __slots__ = ("message",)
    def __init__(self, content: str):
        self.message = _Msg(content)


class _Resp:
    __slots__ = ("choices", "usage")
    def __init__(self, content: str, usage: _Usage | None = None):
        self.choices = [_Choice(content)]
        self.usage = usage or _Usage()


def _get_project_and_token() -> tuple[str, str]:
    """Get the GCP project and a fresh access token from default credentials."""
    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return project, credentials.token


# ---------------------------------------------------------------------------
# Anthropic on Vertex AI  (rawPredict endpoint, us-east5)
# ---------------------------------------------------------------------------

class _AnthropicVertexCompletions:
    def __init__(self, location: str):
        self._location = location

    def create(self, *, model: str, messages: list[dict], **kwargs: Any) -> _Resp:
        project, token = _get_project_and_token()

        system_parts: list[str] = []
        api_messages: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(m["content"])
            else:
                api_messages.append({"role": m["role"], "content": m["content"]})

        url = (
            f"https://{self._location}-aiplatform.googleapis.com/v1/"
            f"projects/{project}/locations/{self._location}/"
            f"publishers/anthropic/models/{model}:rawPredict"
        )
        body: dict[str, Any] = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": api_messages,
            "max_tokens": 8192,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)

        resp = httpx.post(
            url, json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]
        usage_data = data.get("usage", {})
        usage = _Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cache_read_input_tokens=usage_data.get("cache_read_input_tokens", 0),
        )
        return _Resp(text, usage=usage)


class _AnthropicVertexChat:
    def __init__(self, location: str):
        self.completions = _AnthropicVertexCompletions(location)


class AnthropicVertexJudge:
    """OpenAI-interface wrapper around Claude on Vertex AI (rawPredict)."""

    def __init__(self, location: str = _ANTHROPIC_LOCATION):
        self.chat = _AnthropicVertexChat(location)


# ---------------------------------------------------------------------------
# Gemini on Vertex AI  (google.genai SDK)
# ---------------------------------------------------------------------------

class _GeminiGenAICompletions:
    def __init__(self):
        from google import genai
        self._client = genai.Client(vertexai=True)

    def create(self, *, model: str, messages: list[dict], **kwargs: Any) -> _Resp:
        from google.genai import types

        system_parts: list[str] = []
        contents: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(m["content"])
            else:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append(
                    {"role": role, "parts": [{"text": m["content"]}]}
                )

        if not contents:
            contents.append({"role": "user", "parts": [{"text": "Begin."}]})

        config = types.GenerateContentConfig(
            system_instruction="\n\n".join(system_parts) if system_parts else None,
            max_output_tokens=8192,
        )

        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        return _Resp(response.text or "")


class _GeminiGenAIChat:
    def __init__(self):
        self.completions = _GeminiGenAICompletions()


class GeminiVertexJudge:
    """OpenAI-interface wrapper around Gemini on Vertex AI (google.genai SDK)."""

    def __init__(self):
        self.chat = _GeminiGenAIChat()


# ---------------------------------------------------------------------------
# Judge configurations and factory
# ---------------------------------------------------------------------------

JUDGE_CONFIGS: list[dict[str, str]] = [
    {"name": "gpt-5.4", "model": "gpt-5.4-2026-03-05", "client_type": "openai", "reasoning_effort": "xhigh"},
    {"name": "claude-opus-4.6", "model": "claude-opus-4-6", "client_type": "anthropic_vertex"},
    # {"name": "gemini-3.1", "model": "gemini-3.1-pro-preview", "client_type": "gemini_vertex"},
]


def create_judge_client(client_type: str) -> Any:
    """Create a judge client matching the OpenAI chat.completions.create() interface."""
    if client_type == "openai":
        from openai import OpenAI

        from eval._utils import (
            get_openai_api_key,
        )

        api_key = get_openai_api_key()
        return OpenAI(api_key=api_key, base_url="https://us.api.openai.com/v1/")
    if client_type == "anthropic_vertex":
        return AnthropicVertexJudge()
    if client_type == "gemini_vertex":
        return GeminiVertexJudge()
    raise ValueError(f"Unknown client type: {client_type}")
