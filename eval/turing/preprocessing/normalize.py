"""Normalize simulated conversations to turing evaluation format.

Takes raw simulated conversations (simulated_raw.jsonl) and produces
simulated.jsonl matching the format used by the turing test.

Normalization steps:
  1. Remove boilerplate intro messages
  2. Redact any leaked patient names
  3. Truncate so the last turn is by the patient
  4. Format as turing JSONL: {id, transcript, metadata}
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_RAW_PATH = DATA_DIR / "simulated_raw.jsonl"
DEFAULT_OUTPUT_PATH = DATA_DIR / "simulated.jsonl"

_NAME_PATTERN = re.compile(
    r"(?:Hi|Hello|Dear),?\s+([A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15})?)\s*[:\n,!.]"
)


def _redact_names(text: str) -> str:
    """Replace greeting-style names with [PATIENT_NAME]."""
    return _NAME_PATTERN.sub(
        lambda m: m.group(0).replace(m.group(1), "[PATIENT_NAME]"), text
    )


def normalize_conversation(raw: dict) -> dict | None:
    """Normalize a single simulated conversation to turing format.

    Returns None if the conversation is too short.
    """
    transcript = raw.get("transcript", [])
    if not transcript or len(transcript) < 2:
        return None

    # Drop doctor greeting + patient eligibility response
    transcript = transcript[2:]

    # Drop EHR boilerplate: "Your record shows...", and retry
    # messages ("I'm sorry, I didn't catch/understand...") + patient
    # responses.
    while len(transcript) >= 2 and transcript[0]["role"] == "doctor":
        content = transcript[0]["content"]
        if ("Your record shows" in content
                or "I'm sorry, I didn't" in content):
            transcript = transcript[2:]
        else:
            break

    if len(transcript) < 2:
        return None

    # Truncate at PHA report (doctor message with "## What to do" or "## Why we recommend")
    truncated_transcript = []
    for msg in transcript:
        if msg["role"] == "doctor" and re.search(
            r"##\s*(What to do|Why we recommend|When to seek)", msg["content"]
        ):
            break
        truncated_transcript.append(msg)
    transcript = truncated_transcript

    # Truncate so the last turn is by the patient
    while transcript and transcript[-1]["role"] != "patient":
        transcript.pop()

    if len(transcript) < 2:
        return None

    # Normalize doctor messages: greeting + name redaction
    normalized_transcript = []
    for msg in transcript:
        normalized_msg = dict(msg)
        if msg["role"] == "doctor":
            content = msg["content"]
            # Normalize greeting to "Hi, [PATIENT_NAME]:"
            content = re.sub(
                r"^Hi,?\s*[^:\n]+:\s*\n*",
                "Hi, [PATIENT_NAME]:\n\n",
                content,
            )
            content = re.sub(r"^Hi,?\s*\n+", "Hi, [PATIENT_NAME]:\n\n", content)
            normalized_msg["content"] = content
        normalized_transcript.append(normalized_msg)

    session_id = raw["session_id"]
    return {
        "id": session_id,
        "transcript": normalized_transcript,
        "metadata": {
            "source_session_id": session_id,
            "age": raw.get("age"),
            "gender": raw.get("gender"),
            "turn_count": raw.get("turn_count"),
            "urgency_status": raw.get("urgency_status", ""),
            "pha_urgency_label": raw.get("pha_urgency_label", ""),
            "urgency_reasoning": raw.get("urgency_reasoning", ""),
            "care_recommendations": raw.get("care_recommendations", ""),
            "when_to_escalate": raw.get("when_to_escalate", ""),
            "most_probable_diagnosis": raw.get("most_probable_diagnosis", ""),
            "labs_tests_needed": raw.get("labs_tests_needed", ""),
            "sim_params": raw.get("sim_params"),
            "source": "simulated",
        },
    }


def run_normalization(
    raw_path: str | Path = DEFAULT_RAW_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> list[dict]:
    """Normalize all simulated conversations to turing format.

    Returns list of normalized conversation dicts.
    """
    raw_path = Path(raw_path)
    output_path = Path(output_path)

    records: list[dict] = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise FileNotFoundError(f"No records in {raw_path}")

    logger.info("Normalizing %d simulated conversations", len(records))

    results: list[dict] = []
    for record in tqdm(records, desc="Normalizing", unit="case"):
        if record.get("error"):
            logger.debug("Skipping %s: %s", record["session_id"], record["error"])
            continue
        normalized = normalize_conversation(record)
        if normalized:
            results.append(normalized)
        else:
            logger.debug(
                "Skipping %s: too few turns after normalization",
                record["session_id"],
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(
        "Normalization complete: %d/%d usable -> %s",
        len(results),
        len(records),
        output_path,
    )
    return results
