"""LLM-based parameter extraction from real conversations.

Reads real_clean.csv and uses an LLM to extract simulation parameters
(vignette, behavioral params, EHR data) from each conversation.

Anti-leakage guarantees:
  - Vignette is built ONLY from patient-stated facts
  - No annotations, labels, or PHA outputs are sent to the LLM
  - EHR data is extracted directly from source columns (no LLM interpretation)

Parameter levels are loaded from src/param_schema.json at import time
so the extraction prompt always stays in sync with the simulator.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)

_PATIENT_SIM_DIR = Path(__file__).parents[3]
DEFAULT_CSV_PATH = _PATIENT_SIM_DIR / "data" / "real" / "real_clean.csv"
TURING_DATA_DIR = _PATIENT_SIM_DIR / "eval" / "turing" / "data"
DEFAULT_PARAMS_OUTPUT = TURING_DATA_DIR / "extracted_params.jsonl"

# ────────────────────────────────────────────────────────────────────────────
# Schema / param definitions
# ────────────────────────────────────────────────────────────────────────────

_SCHEMA_PATH = _PATIENT_SIM_DIR / "src" / "param_schema.json"
_SCHEMA: dict = json.loads(_SCHEMA_PATH.read_text())

_ALL_PARAMS: dict[str, dict] = {}
for _group in _SCHEMA["groups"]:
    for _pname, _pdef in _group["params"].items():
        _ALL_PARAMS[_pname] = _pdef

_CATEGORY_PARAMS: list[str] = [
    "anxiety", "frustration", "patience", "trust", "embarrassment",
    "recall_level", "oversharing_tendency",
    "health_literacy", "verbosity", "communication_style",
    "english_proficiency", "profanity_level",
    "typo_rate", "abbreviation_tendency", "punctuation_style", "capitalisation",
]
_TEXT_LIST_PARAMS: list[str] = ["topic_sensitivity"]
_TEXT_OPTIONAL_PARAMS: list[str] = ["mother_tongue", "cultural_context", "socioeconomic_context"]

_FIXED_SIM_PARAMS: dict = {
    "gives_up": False,
    "stops_when_goal_met": False,
    "max_turns": None,
}


def _build_param_definitions() -> str:
    """Build PARAMETER DEFINITIONS block from param_schema.json."""
    lines: list[str] = []

    for pname in _CATEGORY_PARAMS:
        pdef = _ALL_PARAMS[pname]
        label = pdef["label"]
        categories = pdef["categories"]
        options = " | ".join(
            f'"{key}" ({cdef["description"].split(".")[0].lower()})'
            for key, cdef in categories.items()
        )
        lines.append(f'{pname} [{label}]: {options}')

    for pname in _TEXT_LIST_PARAMS:
        pdef = _ALL_PARAMS[pname]
        label = pdef["label"]
        desc = pdef["description"].split(".")[0] + "."
        lines.append(f'{pname} [{label}]: list of strings — {desc}')

    for pname in _TEXT_OPTIONAL_PARAMS:
        pdef = _ALL_PARAMS[pname]
        label = pdef["label"]
        desc = pdef["description"].split(".")[0] + "."
        lines.append(f'{pname} [{label}]: string or null — {desc}')

    return "\n".join(lines)


def _build_json_template() -> str:
    """Build the expected JSON output template from schema defaults."""
    template: dict = {
        "vignette": "Structured clinical vignette (see VIGNETTE FORMAT below)",
        "patient_goal": "What the patient wanted from this conversation (1 sentence)",
    }
    for pname in _CATEGORY_PARAMS:
        template[pname] = _ALL_PARAMS[pname]["default"]
    for pname in _TEXT_LIST_PARAMS:
        template[pname] = _ALL_PARAMS[pname]["default"]
    for pname in _TEXT_OPTIONAL_PARAMS:
        template[pname] = _ALL_PARAMS[pname]["default"]
    return json.dumps(template)


EXTRACTION_SYSTEM_PROMPT = """\
You are an expert at analyzing real doctor-patient conversations to extract \
behavioral parameters that characterize the patient.

You will receive a real conversation from a mobile health app between a patient \
and an AI doctor, along with some EHR context. Analyze the PATIENT's messages \
to determine their communication style, emotional state, and clinical presentation.

VIGNETTE FORMAT:
Build a structured clinical vignette that captures everything the patient \
said or confirmed/denied during the conversation. The conversation is the \
primary source — it represents what the patient knows and has said before. \
EHR data is secondary context to help fill in demographics and conditions \
the doctor asked about.

The vignette must be explicit about yes/no facts. For every question the \
doctor asked and the patient answered, include the answer clearly:
- If the patient confirmed something: state it as fact ("takes levothyroxine daily")
- If the patient denied something: state the denial ("not currently being treated \
for Hashimoto's", "denies nausea")
- If the doctor mentioned a condition from the EHR and the patient responded, \
include the patient's response — this is critical for screening questions

Structure:
"[age]-year-old [gender]. [Chief complaint with timeline]. [Confirmed symptoms]. \
[Denied symptoms]. [Medical history the patient confirmed or denied]. \
[Medications confirmed or denied]. [Any other facts the patient stated]."

Keep it 3-8 sentences. Be explicit — a simulated patient will use this \
vignette as their only source of knowledge.

Example: "43-year-old female. Acute stomach pain for 5 hours, rated 8/10, \
constant, diffuse but worst in the middle abdomen. Associated anorexia and \
mild loose stool. Denies nausea, vomiting, and fever. Has a history of \
Hashimoto's thyroiditis but is not currently being treated for it. Had a \
laparoscopic appendectomy for acute appendicitis. No allergies."

RULES FOR BEHAVIORAL PARAMS:
1. Analyze the patient's ACTUAL writing patterns for communication params
2. For verbosity, judge by the MEDIAN patient message length, not the longest
3. Do NOT infer diagnoses beyond what the patient said or confirmed

WRITING STYLE RULES — observe the patient's ACTUAL text, character by character:
- typo_rate: if ANY message has a typo, misspelling, or wrong word → "yes"
- abbreviation_tendency: if the patient uses ANY shorthand (idk, bc, rn, w/, \
  pls, thx, lol, gonna, kinda, etc.) → "yes"
- punctuation_style: if the patient MOSTLY omits periods and commas → "none". \
  Only set "standard" if they consistently use proper punctuation throughout.
- capitalisation: if the patient mostly writes in lowercase (including starting \
  sentences without capitals) → "all_lower". Only "standard" if they \
  consistently capitalize sentence starts.
- profanity_level: if any swearing or crude language appears → "yes"

EMOTIONAL STATE RULES — match the clinical context, not just explicit words:
- anxiety: set "anxious" if the chief complaint involves something scary \
  (chest pain, breathing difficulty, bleeding, neurological symptoms, cancer \
  worry) or if the patient asks worried questions. Most patients seeking care \
  have SOME anxiety — only use "neutral" for routine/mundane visits.
- embarrassment: set "embarrassed" if the topic involves bodily functions, \
  sexual health, mental health, substance use, or anything socially sensitive.
- frustration: set "yes" if the patient shows ANY impatience, curtness, or \
  complaint about the process.
- trust: set "sceptical" if the patient pushes back, questions advice, or \
  seeks second opinions. Use "trustful" only for cooperative patients.
- patience: set "impatient" if the patient gives very short answers or seems \
  to want a quick resolution.

CONTEXT RULES:
- mother_tongue: null unless the patient's writing shows clear L1 interference \
or they mention their native language.
- cultural_context: null unless the patient explicitly references cultural \
factors or norms.
- socioeconomic_context: null unless the patient mentions financial constraints, \
insurance concerns, work pressures, or similar.
- topic_sensitivity: empty list unless the patient visibly deflects or refuses \
to discuss a specific topic.

PARAMETER DEFINITIONS:

{param_definitions}

Respond with EXACTLY this JSON (no markdown, no extra text):
{json_template}
""".format(
    param_definitions=_build_param_definitions(),
    json_template=_build_json_template(),
)

_BOILERPLATE_PHRASES = [
    "experimental AI feature",
    "you'll tell me about your symptoms",
    "It takes about 5 minutes",
    "let's get started",
]


def _format_conversation(messages: list[dict]) -> str:
    """Format conversation for the extraction prompt, skipping boilerplate intro."""
    parts = []
    skip_next_ack = False
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        if role == "AGENT" and any(p in content for p in _BOILERPLATE_PHRASES):
            skip_next_ack = True
            continue
        if skip_next_ack and role == "PATIENT" and content.strip().lower() in ("ok", "okay"):
            skip_next_ack = False
            continue
        skip_next_ack = False
        label = "DOCTOR" if role == "AGENT" else "PATIENT"
        parts.append(f"{label}: {content}")
    return "\n\n".join(parts)


def _build_ehr_data(record: dict) -> dict:
    """Extract EHR data directly from CSV columns (no LLM needed)."""
    age = record.get("age", "")
    gender = str(record.get("gender", "")).lower()
    demographics = f"{age}-year-old {gender}" if age else gender

    return {
        "demographics": demographics,
        "social_history": "",
        "chronic_conditions": record.get("chronic_conditions", "") or "",
        "last_encounter_notes": record.get("last_encounter", "") or "",
        "labs": record.get("recent_labs", "") or "",
        "active_meds": "",
        "allergies": "",
        "previous_dx": record.get("previous_dx", "") or "",
        "immunizations": "",
    }


def extract_single(
    record: dict,
    client: OpenAI,
    model: str,
    reasoning_effort: str = "medium",
) -> dict:
    """Extract simulation parameters from a single real conversation."""
    messages = record.get("conversation", [])
    if not messages:
        return {"session_id": record["session_id"], "error": "no messages"}

    conversation_text = _format_conversation(messages)
    ehr_data = _build_ehr_data(record)

    user_parts = [
        "Analyze this conversation and extract patient parameters.",
        "",
        "CONVERSATION:",
        conversation_text,
    ]

    ehr_lines = []
    for field, value in ehr_data.items():
        if value and str(value).strip() and str(value).strip().upper() != "N/A":
            ehr_lines.append(f"  {field}: {value}")
    if ehr_lines:
        user_parts.extend(["", "EHR DATA:", *ehr_lines])

    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_parts)},
        ],
    }
    if reasoning_effort != "none":
        kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["temperature"] = 0

    t0 = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    latency = time.perf_counter() - t0

    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])
        else:
            return {
                "session_id": record["session_id"],
                "error": f"Parse error: {raw[:200]}",
            }

    ehr_data = _build_ehr_data(record)

    sim_params: dict = {}
    for pname in _CATEGORY_PARAMS:
        default = _ALL_PARAMS[pname]["default"]
        value = parsed.get(pname, default)
        valid_keys = set(_ALL_PARAMS[pname]["categories"].keys())
        sim_params[pname] = value if value in valid_keys else default
    for pname in _TEXT_LIST_PARAMS:
        sim_params[pname] = parsed.get(pname, _ALL_PARAMS[pname]["default"]) or []
    for pname in _TEXT_OPTIONAL_PARAMS:
        sim_params[pname] = parsed.get(pname, _ALL_PARAMS[pname]["default"])
    sim_params.update(_FIXED_SIM_PARAMS)

    return {
        "session_id": record["session_id"],
        "vignette": parsed.get("vignette", ""),
        "patient_goal": parsed.get("patient_goal", ""),
        "sim_params": sim_params,
        "ehr_data": ehr_data,
        "age": record.get("age"),
        "gender": record.get("gender"),
        "extraction_latency_s": round(latency, 2),
    }


def _load_records(csv_path: Path) -> list[dict]:
    """Load records from CSV, parsing the conversation column from JSON."""
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    for r in records:
        conv = r.get("conversation", "[]")
        if isinstance(conv, str):
            try:
                r["conversation"] = json.loads(conv)
            except (json.JSONDecodeError, TypeError):
                r["conversation"] = []
    return records


def run_extraction(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    output_path: str | Path = DEFAULT_PARAMS_OUTPUT,
    model: str = "gpt-5.5-2026-04-23",
    reasoning_effort: str = "medium",
    max_workers: int = 4,
) -> list[dict]:
    """Run parameter extraction on all cases in real_clean.csv.

    Returns list of extracted parameter dicts, also written to output_path.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    records = _load_records(csv_path)

    if not records:
        raise FileNotFoundError(f"No records in {csv_path}")

    logger.info("Extracting params from %d conversations with %s", len(records), model)
    from eval._utils import get_openai_api_key
    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key, base_url="https://us.api.openai.com/v1/")

    results: list[dict] = []

    if max_workers <= 1:
        for record in tqdm(records, desc="Extracting params", unit="case"):
            result = extract_single(record, client, model, reasoning_effort)
            results.append(result)
            logger.debug(
                "Extracted: %s (%.1fs)",
                result["session_id"],
                result.get("extraction_latency_s", 0),
            )
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for record in records:
                f = executor.submit(
                    extract_single, record, client, model, reasoning_effort
                )
                futures[f] = record["session_id"]
            with tqdm(total=len(futures), desc="Extracting params", unit="case") as pbar:
                for f in as_completed(futures):
                    result = f.result()
                    results.append(result)
                    pbar.update(1)
                    logger.debug(
                        "Extracted: %s (%.1fs)",
                        result["session_id"],
                        result.get("extraction_latency_s", 0),
                    )

    results.sort(key=lambda r: r["session_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ok = sum(1 for r in results if "error" not in r)
    logger.info(
        "Extraction complete: %d/%d successful -> %s", n_ok, len(results), output_path
    )
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
    results = run_extraction(csv_path=csv_path)
    n_ok = sum(1 for r in results if "error" not in r)
    print(f"\nExtracted params: {n_ok}/{len(results)} successful")
    print(f"Output: {DEFAULT_PARAMS_OUTPUT}")
