"""Unified triage doctor agent for all patient simulator experiments.

Conducts a structured clinical interview and produces a structured
urgency assessment.
"""
from __future__ import annotations

import json
import logging
import time

import google.auth
import google.auth.transport.requests
import httpx

from doctor.prompt import (
    INTERVIEW_SYSTEM_PROMPT,
    build_assessment_prompt,
)

logger = logging.getLogger(__name__)

DEFAULT_URGENCY_LEVELS: dict[str, str] = {
    "self-care": "Can be managed at home",
    "non-urgent": "Follow up with PCP in the coming days/weeks",
    "urgent": "Needs urgent care or emergency visit now",
}


_ANTHROPIC_LOCATION = "us-east5"
_REQUEST_TIMEOUT = 180.0


def _get_gcp_project_and_token() -> tuple[str, str]:
    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return project, credentials.token


def _infer_provider(model: str) -> str:
    if model.startswith("claude"):
        return "claude"
    if model.startswith("gemini"):
        return "gemini"
    return "openai"


class TriageDoctorAgent:
    """Doctor agent with configurable urgency levels and structured output.

    Two-phase approach:
    1. Interview turns — normal text completions. Lean system prompt
       maximises prompt-cache hits.
    2. Assessment — one ``assess()`` call with structured output
       (``response_format`` for OpenAI, ``response_schema`` for Gemini).
    """

    def __init__(self, config: dict) -> None:
        self._model = config.get("model", "gpt-5.4-mini-2026-03-17")
        self._reasoning_effort = config.get("reasoning_effort", "high")
        self._max_turns: int = int(config.get("max_turns", 5))
        self._assessment_repeats: int = int(config.get("assessment_repeats", 1))
        self._allow_early_assess: bool = config.get("allow_early_assess", True)
        self._urgency_levels: dict[str, str] = config.get(
            "urgency_levels", DEFAULT_URGENCY_LEVELS
        )
        self._provider = _infer_provider(self._model)

        system_prompt_override = config.get("system_prompt")
        if system_prompt_override:
            self._system_prompt = system_prompt_override.replace(
                "{max_turns}", str(self._max_turns)
            )
        else:
            self._system_prompt = INTERVIEW_SYSTEM_PROMPT.format(
                max_turns=self._max_turns
            )

        if not self._allow_early_assess:
            idx = self._system_prompt.find("\nEARLY ASSESSMENT")
            if idx != -1:
                self._system_prompt = self._system_prompt[:idx]

        urgency_enum = list(self._urgency_levels.keys())
        self._assessment_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "triage_assessment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "urgency": {"type": "string", "enum": urgency_enum},
                        "urgency_explanation": {"type": "string"},
                    },
                    "required": ["urgency", "urgency_explanation"],
                    "additionalProperties": False,
                },
            },
        }
        self._assessment_prompt = build_assessment_prompt(self._urgency_levels)

        if self._provider == "openai":
            from openai import OpenAI

            from eval._utils import (
                get_openai_api_key,
            )

            self._client = OpenAI(
                api_key=get_openai_api_key(),
                base_url="https://us.api.openai.com/v1/",
            )
        elif self._provider == "claude":
            self._client = None
        else:
            from google import genai  # type: ignore[import]

            self._client = genai.Client(vertexai=True)

        self._messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
        ]
        self._transcript: list[dict] = []
        self._turn_count: int = 0
        self._token_usage: list[dict] = []

    # ── Interview ─────────────────────────────────────────────────────────

    def start(self) -> str:
        """Generate the doctor's opening message."""
        self._turn_count = 1
        response_text = self._call_model()
        self._messages.append({"role": "assistant", "content": response_text})
        self._transcript.append({"role": "doctor", "content": response_text})
        return response_text

    _ASSESS_TOKEN = "<<ASSESS>>"

    def respond(self, patient_msg: str) -> tuple[str, bool]:
        """Process a patient message and return ``(doctor_text, is_final)``.

        Returns ``("", True)`` when the interview is over — either because
        ``turn_count`` exceeded ``max_turns`` or the doctor produced
        ``<<ASSESS>>``.  The caller should then invoke ``assess_multi()``.
        """
        self._turn_count += 1

        self._messages.append({"role": "user", "content": patient_msg})
        self._transcript.append({"role": "patient", "content": patient_msg})

        if self._turn_count > self._max_turns:
            return "", True

        response_text = self._call_model()

        if self._allow_early_assess and self._ASSESS_TOKEN in response_text and self._turn_count >= 2:
            return "", True

        self._messages.append({"role": "assistant", "content": response_text})
        self._transcript.append({"role": "doctor", "content": response_text})

        return response_text, False

    # ── Assessment ────────────────────────────────────────────────────────

    def assess(self, seed: int = 42, temperature: float | None = None) -> dict:
        """Structured-output call to extract the urgency assessment.

        Uses ``response_format`` (OpenAI) or ``response_schema`` (Gemini)
        to guarantee valid JSON with a constrained urgency enum.
        Returns ``{"urgency": str, "urgency_explanation": str}``.
        """
        assess_messages = [
            msg for msg in self._messages if msg["role"] != "system"
        ] + [
            {"role": "user", "content": self._assessment_prompt},
        ]

        if self._provider == "openai":
            return self._assess_openai(assess_messages, seed=seed)
        if self._provider == "claude":
            return self._assess_claude(assess_messages, temperature=temperature or 0.0)
        return self._assess_gemini(assess_messages, seed=seed)

    def assess_multi(self, n: int | None = None) -> list[dict]:
        """Run N independent assessment calls with different seeds.

        For Claude (no seed support), uses temperatures 0.0–0.9 instead.
        Returns a list of N ``{"urgency": str, "urgency_explanation": str}`` dicts.
        """
        n = n or self._assessment_repeats
        if self._provider == "claude":
            return [self.assess(temperature=i / 10) for i in range(n)]
        return [self.assess(seed=42 + i) for i in range(n)]

    def _assess_openai(self, messages: list[dict], seed: int = 42) -> dict:
        t0 = time.perf_counter()
        response = self._client.with_options(timeout=120.0).chat.completions.create(
            model=self._model,
            messages=messages,
            reasoning_effort=self._reasoning_effort,
            response_format=self._assessment_response_format,
            seed=seed,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._track_openai_usage(response, latency_ms)
        return json.loads(response.choices[0].message.content)

    def _assess_gemini(self, messages: list[dict], seed: int = 42) -> dict:
        from google.genai import types  # type: ignore[import]

        system_instruction, contents = self._messages_to_gemini(messages)
        urgency_enum = list(self._urgency_levels.keys())
        response_schema = {
            "type": "object",
            "properties": {
                "urgency": {"type": "string", "enum": urgency_enum},
                "urgency_explanation": {"type": "string"},
            },
            "required": ["urgency", "urgency_explanation"],
        }
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0,
            seed=seed,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        t0 = time.perf_counter()
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._track_gemini_usage(response, latency_ms)
        return json.loads(response.text)

    def record_patient_message(self, msg: str) -> None:
        """Record a patient message without generating a doctor response.

        Use when the patient ends the conversation early.
        """
        self._transcript.append({"role": "patient", "content": msg})
        self._messages.append({"role": "user", "content": msg})

    # ── Accessors ─────────────────────────────────────────────────────────

    def get_transcript(self) -> list[dict]:
        """Return ``[{role, content}]`` for the interview (no assessment)."""
        return list(self._transcript)

    @property
    def token_usage(self) -> list[dict]:
        return list(self._token_usage)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _call_model(self) -> str:
        if self._provider == "openai":
            return self._call_openai()
        if self._provider == "claude":
            return self._call_claude()
        return self._call_gemini()

    def _call_openai(self) -> str:
        t0 = time.perf_counter()
        response = self._client.with_options(timeout=120.0).chat.completions.create(
            model=self._model,
            messages=self._messages,
            reasoning_effort=self._reasoning_effort,
            seed=42,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._track_openai_usage(response, latency_ms)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from doctor model")
        return content

    def _claude_raw_predict(self, messages: list[dict], **kwargs) -> dict:
        project, token = _get_gcp_project_and_token()
        system_parts: list[str] = []
        api_messages: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(m["content"])
            else:
                api_messages.append({"role": m["role"], "content": m["content"]})

        if not api_messages:
            api_messages.append({"role": "user", "content": "Begin."})

        url = (
            f"https://{_ANTHROPIC_LOCATION}-aiplatform.googleapis.com/v1/"
            f"projects/{project}/locations/{_ANTHROPIC_LOCATION}/"
            f"publishers/anthropic/models/{self._model}:rawPredict"
        )
        body: dict = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": api_messages,
            "max_tokens": 8192,
            **kwargs,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)

        resp = httpx.post(
            url, json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def _call_claude(self) -> str:
        t0 = time.perf_counter()
        data = self._claude_raw_predict(self._messages)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        text = data["content"][0]["text"]
        self._track_claude_usage(data, latency_ms)
        if not text:
            raise ValueError("Empty response from doctor model")
        return text

    def _assess_claude(self, messages: list[dict], temperature: float = 0.0) -> dict:
        urgency_enum = list(self._urgency_levels.keys())
        tool = {
            "name": "triage_assessment",
            "description": "Record the triage urgency assessment.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "urgency": {"type": "string", "enum": urgency_enum},
                    "urgency_explanation": {"type": "string"},
                },
                "required": ["urgency", "urgency_explanation"],
            },
        }
        t0 = time.perf_counter()
        data = self._claude_raw_predict(
            messages,
            tools=[tool],
            tool_choice={"type": "tool", "name": "triage_assessment"},
            temperature=temperature,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._track_claude_usage(data, latency_ms)
        for block in data["content"]:
            if block["type"] == "tool_use":
                return block["input"]
        raise ValueError("Claude did not return tool_use for assessment")

    def _track_claude_usage(self, data: dict, latency_ms: float) -> None:
        usage = data.get("usage", {})
        self._token_usage.append({
            "model_id": self._model,
            "node": "doctor",
            "token_usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                "reasoning_tokens": 0,
                "cached_tokens": usage.get("cache_read_input_tokens", 0),
            },
            "latency_ms": latency_ms,
        })

    def _call_gemini(self) -> str:
        from google.genai import types  # type: ignore[import]

        system_instruction, contents = self._messages_to_gemini(self._messages)
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Begin."}]}]
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0,
            seed=42,
        )
        t0 = time.perf_counter()
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._track_gemini_usage(response, latency_ms)
        return response.text or ""

    def _track_gemini_usage(self, response, latency_ms: float) -> None:
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
        self._token_usage.append({
            "model_id": self._model,
            "node": "doctor",
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "reasoning_tokens": 0,
                "cached_tokens": cached_tokens,
            },
            "latency_ms": latency_ms,
        })

    def _track_openai_usage(self, response, latency_ms: float) -> None:
        usage = response.usage
        reasoning_tokens = getattr(
            getattr(usage, "completion_tokens_details", None),
            "reasoning_tokens", 0,
        ) or 0
        cached_tokens = getattr(
            getattr(usage, "prompt_tokens_details", None),
            "cached_tokens", 0,
        ) or 0
        self._token_usage.append({
            "model_id": self._model,
            "node": "doctor",
            "token_usage": {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "reasoning_tokens": reasoning_tokens,
                "cached_tokens": cached_tokens,
            },
            "latency_ms": latency_ms,
        })

    @staticmethod
    def _messages_to_gemini(
        messages: list[dict],
    ) -> tuple[str | None, list[dict]]:
        """Convert OpenAI-format messages to Gemini's format."""
        system_instruction: str | None = None
        contents: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(
                    {"role": role, "parts": [{"text": msg["content"]}]}
                )
        return system_instruction, contents
