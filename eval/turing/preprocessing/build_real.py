"""Build real.jsonl from real_clean.csv.

Converts preprocessed real conversations into turing evaluation format:
{id, transcript, metadata}.

Handles:
  - Role normalization (Agent → doctor, Patient → patient)
  - Greeting/boilerplate removal
  - Name redaction
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

_PATIENT_SIM_DIR = Path(__file__).parents[3]
DEFAULT_CSV_PATH = _PATIENT_SIM_DIR / "data" / "real" / "real_clean.csv"
TURING_DATA_DIR = _PATIENT_SIM_DIR / "eval" / "turing" / "data"
DEFAULT_REAL_OUTPUT = TURING_DATA_DIR / "real.jsonl"


def build_real_jsonl(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    output_path: str | Path = DEFAULT_REAL_OUTPUT,
) -> list[dict]:
    """Build real.jsonl from real_clean.csv.

    Returns list of conversation dicts with complete metadata.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")

    if not records:
        raise FileNotFoundError(f"No records in {csv_path}")

    logger.info("Building real.jsonl from %d records in %s", len(records), csv_path.name)

    results: list[dict] = []
    for record in tqdm(records, desc="Building real.jsonl", unit="case"):
        conversation = record.get("conversation", "[]")
        if isinstance(conversation, str):
            messages = json.loads(conversation)
        else:
            messages = conversation or []

        if not messages:
            logger.debug("Skipping %s: no messages", record.get("session_id"))
            continue

        transcript = []
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if role == "agent":
                content = re.sub(
                    r"^Hi,?\s*[^:\n]+:\s*\n*",
                    "Hi, [PATIENT_NAME]:\n\n",
                    content,
                )
                content = re.sub(r"^Hi,?\s*\n+", "Hi, [PATIENT_NAME]:\n\n", content)
                transcript.append({"role": "doctor", "content": content})
            elif role == "patient":
                transcript.append({"role": "patient", "content": content})

        while transcript and transcript[0]["role"] == "patient":
            transcript.pop(0)

        # Drop doctor greeting + patient eligibility response
        transcript = transcript[2:]

        # Drop EHR boilerplate: "Your record shows...", and retry
        # messages ("I'm sorry, I didn't catch/understand...") + patient
        # responses. Strip pairs from the top until we're past them.
        while len(transcript) >= 2 and transcript[0]["role"] == "doctor":
            content = transcript[0]["content"]
            if ("Your record shows" in content
                    or "I'm sorry, I didn't" in content):
                transcript = transcript[2:]
            else:
                break

        if len(transcript) < 3:
            logger.debug("Skipping %s: too short", record.get("session_id"))
            continue

        session_id = str(record.get("session_id", ""))

        def _str(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return ""
            return str(val)

        metadata = {
            "session_id": session_id,
            "batch": _str(record.get("batch")),
            "patient_id": _str(record.get("patient_id")),
            "age": record.get("age"),
            "gender": _str(record.get("gender")),
            "location": _str(record.get("location")),
            "urgency_status": _str(record.get("pha_triage")),
            "pha_urgency_label": _str(record.get("pha_triage")),
            "urgency_reasoning": _str(record.get("urgency_reasoning")),
            "care_recommendations": _str(record.get("care_recommendations")),
            "when_to_escalate": _str(record.get("when_to_escalate")),
            "most_probable_diagnosis": _str(record.get("most_probable_diagnosis")),
            "labs_tests_needed": _str(record.get("labs_tests_needed")),
            "labeller_urgency": _str(record.get("labeller_triage")),
        }

        results.append({
            "id": session_id,
            "transcript": transcript,
            "metadata": metadata,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(
        "Built real.jsonl: %d conversations -> %s",
        len(results),
        output_path,
    )
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
    results = build_real_jsonl(csv_path=csv_path)
    print(f"\nBuilt {len(results)} real conversations")
    print(f"Output: {DEFAULT_REAL_OUTPUT}")
