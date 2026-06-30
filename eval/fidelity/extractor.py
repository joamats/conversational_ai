"""Two-phase LLM pipeline for clinical fidelity evaluation.

Phase 1: extract_claims()  — atomic claim extraction from vignette + transcript.
Phase 2: classify_claims() — score each patient claim against vignette ground truth.
"""
from __future__ import annotations

import json
import logging
import time

from eval._utils import (
    format_transcript,
    parse_json_response,
)
from eval.turing.llms.judge_clients import (
    create_judge_client,
)
from eval.fidelity.prompts import (
    EXTRACT_CLAIMS_PROMPT,
    CLASSIFY_CLAIMS_PROMPT,
)

logger = logging.getLogger(__name__)


def _call_json(
    client, messages: list[dict], model: str, reasoning_effort: str,
    node_name: str = "judge", client_type: str = "openai",
) -> tuple[dict, float, dict]:
    create_kwargs: dict = {"model": model, "messages": messages}
    if client_type == "openai":
        create_kwargs["reasoning_effort"] = reasoning_effort
        create_kwargs["response_format"] = {"type": "json_object"}
        create_kwargs["seed"] = 42

    t0 = time.perf_counter()
    if client_type == "openai":
        response = client.with_options(timeout=180.0).chat.completions.create(**create_kwargs)
    else:
        response = client.chat.completions.create(**create_kwargs)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from model")

    parsed = parse_json_response(content)
    if not parsed:
        logger.error("Failed to parse JSON: %s", content[:500])
        raise ValueError(f"Could not parse JSON from response: {content[:200]}")

    usage = response.usage
    reasoning_tokens = getattr(
        getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0
    ) or 0
    cached_tokens = getattr(
        getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0
    ) or 0
    token_usage = {
        "model_id": model,
        "node": node_name,
        "token_usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
        },
        "latency_ms": latency_ms,
    }

    return parsed, latency_ms, token_usage


def extract_claims(
    vignette: str,
    transcript: list[dict],
    judge_config: dict,
) -> dict:
    """Phase 1: Extract atomic claims from vignette and conversation.

    Returns:
        {
            "vignette_claims": [str, ...],
            "conversation_patient_claims": [str, ...],
            "extraction_latency_ms": float,
        }
    """
    conversation_text = format_transcript(transcript)
    prompt = EXTRACT_CLAIMS_PROMPT.format(
        vignette=vignette,
        conversation=conversation_text,
    )

    model = judge_config.get("model", "gpt-5.4-2026-03-05")
    reasoning_effort = judge_config.get("reasoning_effort", "medium")
    client_type = judge_config.get("client_type", "openai")
    client = create_judge_client(client_type)

    messages = [{"role": "user", "content": prompt}]
    parsed, latency_ms, tok = _call_json(client, messages, model, reasoning_effort, "judge_extract", client_type)

    vignette_claims = parsed.get("vignette_claims", [])
    conversation_claims = parsed.get("conversation_patient_claims", [])

    logger.info(
        "Extracted %d vignette claims, %d patient claims (%.0fms)",
        len(vignette_claims), len(conversation_claims), latency_ms,
    )

    return {
        "vignette_claims": vignette_claims,
        "conversation_patient_claims": conversation_claims,
        "extraction_latency_ms": latency_ms,
        "judge_token_usage": [tok],
    }


def classify_claims(
    vignette_claims: list[str],
    conversation_patient_claims: list[str],
    judge_config: dict,
) -> dict:
    """Phase 2: Classify each patient claim against vignette ground truth.

    Returns:
        {
            "classified_claims": [{claim, category, score, label, rationale, matched_vignette_claim}, ...],
            "vignette_claims_not_mentioned": [{claim}, ...],
            "summary": str,
            "classification_latency_ms": float,
        }
    """
    vc_block = "\n".join(f"- {c}" for c in vignette_claims)
    cc_block = "\n".join(f"- {c}" for c in conversation_patient_claims)

    prompt = CLASSIFY_CLAIMS_PROMPT.format(
        vignette_claims=vc_block,
        conversation_patient_claims=cc_block,
    )

    model = judge_config.get("model", "gpt-5.4-2026-03-05")
    reasoning_effort = judge_config.get("reasoning_effort", "medium")
    client_type = judge_config.get("client_type", "openai")
    client = create_judge_client(client_type)

    messages = [{"role": "user", "content": prompt}]
    parsed, latency_ms, tok = _call_json(client, messages, model, reasoning_effort, "judge_classify", client_type)

    classified = parsed.get("classified_claims", [])
    not_mentioned = parsed.get("vignette_claims_not_mentioned", [])
    summary = parsed.get("summary", "")

    n_supported = sum(1 for c in classified if c.get("category") == "supported")
    n_unsupported = sum(1 for c in classified if c.get("category") == "unsupported")

    logger.info(
        "Classified %d claims: %d supported, %d unsupported, %d vignette gaps (%.0fms)",
        len(classified), n_supported, n_unsupported, len(not_mentioned), latency_ms,
    )

    return {
        "classified_claims": classified,
        "vignette_claims_not_mentioned": not_mentioned,
        "summary": summary,
        "classification_latency_ms": latency_ms,
        "judge_token_usage": [tok],
    }


def evaluate_conversation(
    vignette: str,
    transcript: list[dict],
    judge_config: dict,
) -> dict:
    """Run the full two-phase fidelity evaluation on one conversation.

    Returns combined extraction + classification results.
    """
    extraction = extract_claims(vignette, transcript, judge_config)

    if not extraction["conversation_patient_claims"]:
        logger.warning("No patient claims extracted — skipping classification.")
        return {
            **extraction,
            "classified_claims": [],
            "vignette_claims_not_mentioned": [
                {"claim": c} for c in extraction["vignette_claims"]
            ],
            "summary": "No patient claims were found in the conversation.",
            "classification_latency_ms": 0,
        }

    classification = classify_claims(
        extraction["vignette_claims"],
        extraction["conversation_patient_claims"],
        judge_config,
    )

    combined = {**extraction, **classification}
    combined["judge_token_usage"] = (
        extraction.get("judge_token_usage", []) + classification.get("judge_token_usage", [])
    )
    return combined
