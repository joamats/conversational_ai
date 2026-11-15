"""Experiment 2 additions to the clinical-fidelity pipeline (``eval/fidelity``).

Each source vignette's claims are extracted once, and every conversation of that case is classified against the
same list, so the five persona conversations of a case share one coverage denominator. The patient-claim
extraction and classification steps are those of ``eval.fidelity.extractor``.

Contents: ``EXTRACT_VIGNETTE_CLAIMS_PROMPT``, ``extract_vignette_claims`` and ``evaluate_conversation`` (the
``eval.fidelity.extractor`` function with an optional ``vignette_claims`` argument).
"""
from __future__ import annotations

import logging

from eval.fidelity.extractor import _call_json, classify_claims, extract_claims
from eval.turing.llms.judge_clients import create_judge_client

logger = logging.getLogger(__name__)


EXTRACT_VIGNETTE_CLAIMS_PROMPT = """\
You are a clinical information extraction specialist.

You will be given:
- PATIENT VIGNETTE: A structured patient profile (ground truth).

Your task is to enumerate the factual claims from the vignette.

Produce a JSON object with exactly one key:

1. "vignette_claims"
   A flat list of all distinct, atomic factual claims derivable from the vignette.
   Each entry should be a single self-contained statement (e.g. "Patient is 84 years old",
   "Patient has atrial fibrillation", "Patient takes atorvastatin 20mg daily",
   "Patient denies alcohol use"). Cover all clinically meaningful content: demographics,
   social history, medical history, current visit details, medications, allergies,
   symptoms, and diagnosis.

The list must be deduplicated: if the same fact is stated more than once, include it only
once using the clearest and most complete formulation. Two claims cover the same topic if
they assert the same underlying fact — minor wording differences do not make them distinct
entries.

Output valid JSON only. No preamble, no explanation outside the JSON.

---

PATIENT VIGNETTE:
{vignette}
"""


def extract_vignette_claims(vignette: str, judge_config: dict) -> dict:
    """Extract atomic claims from a vignette alone.

    Used when several conversations share one vignette, so they can all be
    scored against the same claim list (a common coverage denominator).

    Returns:
        {
            "vignette_claims": [str, ...],
            "extraction_latency_ms": float,
            "judge_token_usage": [dict],
        }
    """
    prompt = EXTRACT_VIGNETTE_CLAIMS_PROMPT.format(vignette=vignette)

    model = judge_config.get("model", "gpt-5.4-2026-03-05")
    reasoning_effort = judge_config.get("reasoning_effort", "medium")
    client_type = judge_config.get("client_type", "openai")
    client = create_judge_client(client_type)

    messages = [{"role": "user", "content": prompt}]
    parsed, latency_ms, tok = _call_json(
        client, messages, model, reasoning_effort, "judge_extract_vignette", client_type,
    )

    vignette_claims = parsed.get("vignette_claims", [])
    logger.info("Extracted %d vignette-only claims (%.0fms)", len(vignette_claims), latency_ms)

    return {
        "vignette_claims": vignette_claims,
        "extraction_latency_ms": latency_ms,
        "judge_token_usage": [tok],
    }


def evaluate_conversation(
    vignette: str,
    transcript: list[dict],
    judge_config: dict,
    vignette_claims: list[str] | None = None,
) -> dict:
    """Run the full two-phase fidelity evaluation on one conversation.

    If ``vignette_claims`` is given (e.g. from ``extract_vignette_claims``,
    shared across conversations of the same vignette), the patient claims are
    classified against it instead of the vignette claims extracted alongside
    this conversation.

    Returns combined extraction + classification results.
    """
    extraction = extract_claims(vignette, transcript, judge_config)
    if vignette_claims is not None:
        extraction["vignette_claims"] = vignette_claims

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
