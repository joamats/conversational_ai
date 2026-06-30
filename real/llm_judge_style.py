"""LLM-as-a-judge for patient communication style in real conversations.

Evaluates each conversation along multiple dimensions grouped into:
  - Goal: what the patient is trying to achieve
  - Symptom category: respiratory, cardiovascular, GI, etc.
  - Sensitive topics: deflection, ignored questions, sensitive content
  - Recall: memory precision (low / medium / high)
  - Feelings: anxious, frustrated, impatient, sceptical, embarrassed
  - Oversharing: provides more info than asked
  - Communication: grammar issues, typos, capitalisation, punctuation,
    health literacy (1-3)

Data source: BigQuery parquet cache (real/data/all_sessions.parquet),
filtered to conversations with ≥2 patient messages.

Results are saved to real/data/llm_judge_style_results.json.

Usage:
    # Full run (all eligible conversations):
    python -m real.llm_judge_style

    # Test run (single conversation, prints result to stdout):
    python -m real.llm_judge_style --test
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from eval._utils import (
    get_openai_api_key,
)

logger = logging.getLogger(__name__)

PARQUET_PATH = Path(__file__).parent / "data" / "all_sessions.parquet"
FILTERED_CSV_PATH = Path(__file__).parent / "data" / "judge_input.csv"
OUTPUT_PATH = Path(__file__).parent / "data" / "llm_judge_style_results.json"

SYSTEM_PROMPT = """\
You are an expert linguist and health communication researcher. You will \
analyze the PATIENT's messages in a doctor-patient conversation to assess \
their communication style along several dimensions.

The DOCTOR is an AI agent — ignore the doctor's style entirely.
Focus ONLY on the PATIENT's messages.

Respond with EXACTLY this JSON (no markdown, no extra text):
{
  "goal": {
    "value": "brief description of what the patient is trying to achieve",
    "evidence": "brief quote or observation supporting this"
  },
  "symptom_category": {
    "value": "exactly one of: respiratory | cardiovascular | gastrointestinal | genitourinary | musculoskeletal | neurological | constitutional | dermatological | mental_health | ent | endocrine | other_mixed | unrelated | unevaluable",
    "evidence": "brief quote or observation of the chief complaint"
  },
  "sensitive_topic": {
    "present": true/false,
    "evidence": "brief description of any deflection, ignored question, or sensitive topic — or empty string if none"
  },
  "oversharing": {
    "value": true/false,
    "evidence": "brief quote showing unsolicited information, or empty string"
  },
  "recall": {
    "level": "low" | "medium" | "high",
    "evidence": "brief quote or observation"
  },
  "feelings": {
    "anxious": {"value": true/false, "evidence": "brief quote or observation"},
    "frustrated": {"value": true/false, "evidence": "brief quote or observation"},
    "impatient": {"value": true/false, "evidence": "brief quote or observation"},
    "sceptical": {"value": true/false, "evidence": "brief quote or observation"},
    "embarrassed": {"value": true/false, "evidence": "brief quote or observation"}
  },
  "communication": {
    "grammar_issues": {"value": true/false, "evidence": "brief quote or observation"},
    "typos": {"value": true/false, "evidence": "brief quote showing the typo, or empty string"},
    "non_standard_capitalisation": {"value": true/false, "evidence": "brief quote, or empty string"},
    "non_standard_punctuation": {"value": true/false, "evidence": "brief quote, or empty string"},
    "health_literacy": {"value": 1 | 2 | 3, "evidence": "brief quote or observation"}
  }
}

Dimension definitions:

GOAL:
  Summarise in one sentence what the patient is seeking from this \
conversation (e.g. "assess severity of headache", "get advice about \
medication side effects", "confirm whether ER visit is needed").

SYMPTOM CATEGORY ASSIGNMENT:
Based on the chief complaint and clinical presentation, assign exactly \
one symptom category from:
  - respiratory: cough, shortness of breath, wheezing, chest tightness \
with breathing
  - cardiovascular: chest pain, palpitations, oedema, syncope, claudication
  - gastrointestinal: abdominal pain, nausea, vomiting, diarrhoea, \
constipation, reflux
  - genitourinary: dysuria, haematuria, urinary frequency, \
vaginal/penile discharge, flank pain
  - musculoskeletal: joint pain, back pain, stiffness, swelling, limited \
range of motion
  - neurological: headache, dizziness, numbness, tingling, weakness, \
seizures, vision changes
  - constitutional: fever, fatigue, weight loss, night sweats, malaise
  - dermatological: rash, itching, skin lesions, wounds, burns, hair loss
  - mental_health: anxiety, depression, insomnia, suicidal ideation, \
substance use disorder
  - ent: ear pain, sore throat, nasal congestion, hearing loss, tinnitus, \
dysphagia
  - endocrine: polyuria, polydipsia, heat/cold intolerance, tremor, \
weight changes with metabolic features
  - other_mixed: symptoms span multiple categories or do not fit any \
single one
  - unrelated: not a health concern (e.g. jailbreak attempt, testing, \
off-topic)
  - unevaluable: insufficient patient data to determine (e.g too few turns)

SENSITIVE TOPIC:
  Is there any sign of deflection, an ignored question from the doctor, \
or a potentially sensitive topic the patient avoids or hesitates around? \
Mark true if any of these are present, false otherwise. If true, write \
the sensitive topics in evidence. \
IMPORTANT: Do NOT flag a session as having a sensitive topic simply \
because the patient stopped responding or abandoned the conversation. \
The patient's last message should not be treated as evidence of a \
sensitive topic — only consider earlier messages where the patient \
actively deflects, avoids, or shows hesitation around a specific topic.

OVERSHARING:
  Does the patient volunteer information beyond what the doctor asked? \
Mark true if the patient provides unsolicited details (e.g. personal \
stories, family history not requested, lifestyle details, emotional \
context, extra details, additional fact) that go beyond the specific \
question posed. Mark false if the patient only answers what was asked.

RECALL (how precisely the patient remembers clinical details):
  low    = vague about timing, medications, dosages; memory gaps apparent
  medium = remembers key facts but not all details; normal recall
  high   = remembers vitals, dosages, medication names without mistakes; \
precise and detailed

FEELINGS — for each, answer true or false:
  anxious      = any sign of worry, nervousness, or fear
  frustrated   = any sign of frustration or annoyance (with symptoms, \
situation, or the agent)
  impatient    = signs of impatience (short answers demanding speed, \
"hurry up", "how long", etc.) — if neutral, answer false
  sceptical    = any sign of scepticism, doubt, or lack of trust in the \
agent or its advice
  embarrassed  = any sign of embarrassment, hesitation, or discomfort \
discussing their issue

COMMUNICATION:
  grammar_issues = true if the patient's writing has noticeable grammar \
problems (imperfect sentences, ideas not well connected, fragments); \
false if grammar is fine
  typos          = true if typos or misspellings are present; false \
otherwise
  non_standard_capitalisation = true if the patient uses non-standard \
capitalisation — e.g. all lowercase, ALL CAPS, no capitalisation at \
sentence start, random capitalisation; false if capitalisation follows \
standard written English conventions
  non_standard_punctuation = true if the patient uses non-standard \
punctuation — e.g. no punctuation at all, missing sentence-final \
periods (except for the last sentence per turn), \
excessive punctuation (!!!), missing commas; false if \
punctuation follows standard written English conventions
  health_literacy = rate 1, 2, or 3 using the scale below:

    1 — Basic / functional literacy
      Information navigation: vague or inappropriate sources (e.g. "my \
son said", "someone on the bus said")
      Comprehension: partial understanding, often asks for clarification
      Appraisal: expresses uncertainty but does not evaluate
      Application: applies advice incorrectly or rigidly
      Vocabulary: body-part and everyday words (belly, chest, pain, ache, \
lump, poorly, sick, tablets, breathing, red)

    2 — Communicative / interactive literacy
      Information navigation: refers to at least one appropriate source \
(doctor, leaflet, etc.)
      Comprehension: generally understands, can restate most information
      Appraisal: shows some comparison or scepticism
      Application: applies advice appropriately to their situation
      Vocabulary: common clinic words (infection, swelling, blood \
pressure, inhaler, dose, treatment, symptoms, diagnosis, side effects, \
glucose, medication)

    3 — Critical literacy
      Information navigation: references multiple appropriate sources, \
uses them strategically (e.g. PubMed)
      Comprehension: accurately paraphrases and integrates information
      Appraisal: critically evaluates credibility and relevance
      Application: adapts advice flexibly and explains how it fits their \
context
      Vocabulary: analytical and advanced medical terms (airway, \
inflammation, prognosis, contraindication, efficacy, complication, \
comorbidity, adherence, recurrence, differential, causality)
"""

VALID_CATEGORIES = {
    "respiratory", "cardiovascular", "gastrointestinal", "genitourinary",
    "musculoskeletal", "neurological", "constitutional", "dermatological",
    "mental_health", "ent", "endocrine", "other_mixed", "unrelated",
    "unevaluable",
}

_MSG_PATTERN = re.compile(r'(\d{4}-\d{2}-\d{2}[T ]\S+)\|\|\|\[(\w+)\]:\s*')


def _parse_raw_chat_history(chat_history_str: str) -> list[dict]:
    """Convert raw BigQuery chat_history to [{role, content}, ...]."""
    chat_text = str(chat_history_str)
    matches = list(_MSG_PATTERN.finditer(chat_text))
    messages = []
    for i, m in enumerate(matches):
        sender = m.group(2).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(chat_text)
        text = chat_text[start:end].strip()
        if not text:
            continue
        role = "Patient" if sender == "patient" else "Agent"
        messages.append({"role": role, "content": text})
    return messages


def _count_patient_messages(chat_history_str: str) -> int:
    chat_text = str(chat_history_str)
    return len([m for m in _MSG_PATTERN.finditer(chat_text)
                if m.group(2).lower() == "patient"])


def _format_conversation(conversation: list[dict]) -> str:
    parts = []
    for turn in conversation:
        role = turn["role"].upper()
        parts.append(f"{role}: {turn['content']}")
    return "\n\n".join(parts)


def _parse_json_response(raw: str) -> dict:
    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end], strict=False)
        return {}


def judge_conversation(
    session_id: str,
    conversation: list[dict],
    client: OpenAI,
    model: str = "gpt-5.4-2026-03-05",
) -> dict:
    transcript_text = _format_conversation(conversation)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        reasoning_effort="high",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analyze the PATIENT's communication style in this "
                    f"conversation:\n\n{transcript_text}"
                ),
            },
        ],
    )
    latency = time.perf_counter() - t0

    raw = response.choices[0].message.content.strip()
    parsed = _parse_json_response(raw)

    # Extract structured fields
    goal_obj = parsed.get("goal", {})
    if isinstance(goal_obj, str):
        goal_val, goal_ev = goal_obj, ""
    else:
        goal_val = goal_obj.get("value", "")
        goal_ev = goal_obj.get("evidence", "")

    cat_obj = parsed.get("symptom_category", {})
    if isinstance(cat_obj, str):
        cat_val, cat_ev = cat_obj, ""
    else:
        cat_val = cat_obj.get("value", None)
        cat_ev = cat_obj.get("evidence", "")
    if cat_val not in VALID_CATEGORIES:
        cat_val = None

    sensitive = parsed.get("sensitive_topic", {})
    sensitive_present = sensitive.get("present", None)
    sensitive_evidence = sensitive.get("evidence", "")

    oversharing_raw = parsed.get("oversharing", {})
    if isinstance(oversharing_raw, dict):
        oversharing = oversharing_raw.get("value", None)
        oversharing_ev = oversharing_raw.get("evidence", "")
    elif isinstance(oversharing_raw, bool):
        oversharing, oversharing_ev = oversharing_raw, ""
    else:
        oversharing, oversharing_ev = None, ""
    if not isinstance(oversharing, bool):
        oversharing = None

    recall = parsed.get("recall", {})
    recall_level = recall.get("level", None)
    if recall_level not in ("low", "medium", "high"):
        recall_level = None
    recall_evidence = recall.get("evidence", "")

    feelings = parsed.get("feelings", {})
    feeling_fields = ["anxious", "frustrated", "impatient", "sceptical", "embarrassed"]
    feelings_out = {}
    for f in feeling_fields:
        raw_f = feelings.get(f, None)
        if isinstance(raw_f, dict):
            val = raw_f.get("value", None)
            ev = raw_f.get("evidence", "")
        elif isinstance(raw_f, bool):
            val, ev = raw_f, ""
        else:
            val, ev = None, ""
        feelings_out[f] = (bool(val) if isinstance(val, bool) else None, ev)

    comm = parsed.get("communication", {})

    grammar_raw = comm.get("grammar_issues", None)
    if isinstance(grammar_raw, dict):
        grammar_issues = grammar_raw.get("value", None)
        grammar_ev = grammar_raw.get("evidence", "")
    elif isinstance(grammar_raw, bool):
        grammar_issues, grammar_ev = grammar_raw, ""
    else:
        grammar_issues, grammar_ev = None, ""
    if not isinstance(grammar_issues, bool):
        grammar_issues = None

    typos_raw = comm.get("typos", None)
    if isinstance(typos_raw, dict):
        typos = typos_raw.get("value", None)
        typos_ev = typos_raw.get("evidence", "")
    elif isinstance(typos_raw, bool):
        typos, typos_ev = typos_raw, ""
    else:
        typos, typos_ev = None, ""
    if not isinstance(typos, bool):
        typos = None

    nsc_raw = comm.get("non_standard_capitalisation", None)
    if isinstance(nsc_raw, dict):
        non_standard_cap = nsc_raw.get("value", None)
        nsc_ev = nsc_raw.get("evidence", "")
    elif isinstance(nsc_raw, bool):
        non_standard_cap, nsc_ev = nsc_raw, ""
    else:
        non_standard_cap, nsc_ev = None, ""
    if not isinstance(non_standard_cap, bool):
        non_standard_cap = None

    nsp_raw = comm.get("non_standard_punctuation", None)
    if isinstance(nsp_raw, dict):
        non_standard_punct = nsp_raw.get("value", None)
        nsp_ev = nsp_raw.get("evidence", "")
    elif isinstance(nsp_raw, bool):
        non_standard_punct, nsp_ev = nsp_raw, ""
    else:
        non_standard_punct, nsp_ev = None, ""
    if not isinstance(non_standard_punct, bool):
        non_standard_punct = None

    hl_raw = comm.get("health_literacy", None)
    if isinstance(hl_raw, dict):
        health_literacy = hl_raw.get("value", None)
        hl_ev = hl_raw.get("evidence", "")
    elif isinstance(hl_raw, int):
        health_literacy, hl_ev = hl_raw, ""
    else:
        health_literacy, hl_ev = None, ""
    if not (isinstance(health_literacy, int) and health_literacy in (1, 2, 3)):
        health_literacy = None

    result = {
        "session_id": session_id,
        "goal": goal_val,
        "goal_evidence": goal_ev,
        "symptom_category": cat_val,
        "symptom_category_evidence": cat_ev,
        "sensitive_topic": sensitive_present,
        "sensitive_topic_evidence": sensitive_evidence,
        "oversharing": oversharing,
        "oversharing_evidence": oversharing_ev,
        "recall": recall_level,
        "recall_evidence": recall_evidence,
    }
    for f in feeling_fields:
        val, ev = feelings_out[f]
        result[f"feeling_{f}"] = val
        result[f"feeling_{f}_evidence"] = ev
    result.update({
        "grammar_issues": grammar_issues,
        "grammar_issues_evidence": grammar_ev,
        "typos": typos,
        "typos_evidence": typos_ev,
        "non_standard_capitalisation": non_standard_cap,
        "non_standard_capitalisation_evidence": nsc_ev,
        "non_standard_punctuation": non_standard_punct,
        "non_standard_punctuation_evidence": nsp_ev,
        "health_literacy": health_literacy,
        "health_literacy_evidence": hl_ev,
        "latency_s": round(latency, 2),
    })
    return result


def load_and_filter(
    parquet_path: Path = PARQUET_PATH,
    csv_path: Path = FILTERED_CSV_PATH,
    min_patient_msgs: int = 2,
) -> pd.DataFrame:
    """Load BigQuery cache, filter to ≥min_patient_msgs, save CSV."""
    df = pd.read_parquet(parquet_path)
    tqdm.pandas(desc="Counting patient messages")
    df["n_patient_messages"] = df["chat_history"].progress_apply(_count_patient_messages)
    df_filtered = df[df["n_patient_messages"] >= min_patient_msgs].copy()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(csv_path, index=False)
    logger.info(
        "Filtered %d → %d sessions (≥%d patient msgs), saved to %s",
        len(df), len(df_filtered), min_patient_msgs, csv_path,
    )
    return df_filtered


def run(
    model: str = "gpt-5.4-2026-03-05",
    max_workers: int = 16,
    test: bool = False,
) -> list[dict]:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_filter()
    logger.info("Loaded %d conversations for judging", len(df))

    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key, base_url="https://us.api.openai.com/v1/")

    items = []
    for _, row in df.iterrows():
        conversation = _parse_raw_chat_history(row["chat_history"])
        items.append((row["session_id"], conversation))

    if test:
        session_id, conv = items[0]
        print("=" * 80)
        print(f"SESSION: {session_id}  ({len(conv)} messages)")
        print("=" * 80)
        print(_format_conversation(conv))
        print("\n" + "=" * 80)
        print("JUDGE OUTPUT")
        print("=" * 80)
        result = judge_conversation(session_id, conv, client, model)
        print(json.dumps(result, indent=2))

        test_path = OUTPUT_PATH.parent / "llm_judge_test_result.json"
        with open(test_path, "w") as f:
            json.dump(
                {"model": model, "n_conversations": 1, "results": [result]},
                f,
                indent=2,
            )
        print(f"\nSaved to {test_path}")
        return [result]

    results: list[dict] = []
    if max_workers <= 1:
        for session_id, conv in tqdm(items, desc="Judging style"):
            result = judge_conversation(session_id, conv, client, model)
            results.append(result)
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for session_id, conv in items:
                f = executor.submit(
                    judge_conversation, session_id, conv, client, model
                )
                futures[f] = session_id
            with tqdm(total=len(futures), desc="Judging style") as pbar:
                for f in as_completed(futures):
                    try:
                        result = f.result()
                        results.append(result)
                    except Exception as e:
                        sid = futures[f]
                        logger.error("Failed for %s: %s", sid, e)
                        results.append({"session_id": sid, "error": str(e)})
                    pbar.update(1)

    results.sort(key=lambda r: r["session_id"])

    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            {"model": model, "n_conversations": len(results), "results": results},
            f,
            indent=2,
        )
    logger.info("Saved %d results to %s", len(results), OUTPUT_PATH)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run on a single conversation and print result")
    parser.add_argument("--model", default="gpt-5.4-2026-03-05")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    run(model=args.model, max_workers=args.workers, test=args.test)
