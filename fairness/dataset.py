"""Data loading and job generation for the persona compound fairness experiment.

Loads cases from preprocessed datasets (HV, AV), reads intersectional
persona definitions from personas.json, and produces a job manifest for
all case x persona x repetition combinations.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_PARAMS: dict[str, str] = {
    "verbosity": "brief",
    "recall_level": "normal",
    "oversharing_tendency": "no",
    "anxiety": "neutral",
    "embarrassment": "neutral",
    "frustration": "no",
    "trust": "trustful",
    "patience": "neutral",
    "english_proficiency": "native",
    "health_literacy": "medium",
}

DEFAULT_URGENCY_LEVELS: dict[str, str] = {
    "self-care": "Can be managed at home",
    "non-urgent": "Follow up with PCP in the coming days/weeks",
    "urgent": "Needs urgent care or emergency visit now",
}

GOLD_LABEL_MAP: dict[str, str] = {
    "Self-care": "self-care",
    "Follow up with PCP": "non-urgent",
    "Urgent care / emergency": "urgent",
}

PERSONA_BASELINE_PARAMS: dict[str, str] = {
    **BASELINE_PARAMS,
    "communication_style": "formal",
    "profanity_level": "no",
    "typo_rate": "no",
    "punctuation_style": "standard",
    "capitalisation": "standard",
}

_HV_EHR_FIELDS = [
    "chief_complaint",
    "history_of_present_illness",
    "review_of_systems",
    "chronic_conditions",
    "current_medications",
    "allergies",
    "immunizations",
    "recent_labs",
    "last_encounter_notes",
    "social_history",
    "family_history",
]


def build_triage_ord(urgency_levels: dict[str, str]) -> dict[str, int]:
    """Derive ordinal mapping from config urgency_levels key order (least -> most urgent)."""
    return {label: i for i, label in enumerate(urgency_levels)}

AV_TRIAGE_MAP: dict[str, str] = {
    "pursue self-care and monitoring": "self-care",
    "seek non-urgent follow-up care": "non-urgent",
    "seek immediate medical care": "urgent",
}


_AV_EHR_FIELDS = {
    "chief_symptom": "chief_complaint",
    "chronic_conditions": "chronic_conditions",
    "last_encounter_notes": "last_encounter_notes",
    "labs": "recent_labs",
    "previous_dx": "history_of_present_illness",
}


# ---------------------------------------------------------------------------
# Dataset loaders — each returns list[dict] with common schema
# ---------------------------------------------------------------------------

def _parse_prior_conversation(raw: str) -> list[dict] | None:
    """Parse the HV conversation column into a list of role/content dicts."""
    if not raw or not raw.strip():
        return None
    messages: list[dict] = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.startswith("Patient:"):
            messages.append({"role": "patient", "content": line[8:].strip()})
        elif line.startswith("Doctor:"):
            messages.append({"role": "doctor", "content": line[7:].strip()})
    return messages if messages else None


def _build_hv_ehr_json(row: dict) -> dict:
    """Build an EHR JSON dict from HV CSV columns."""
    ehr: dict = {}
    demographics = f"{row.get('age', '')}-year-old {row.get('gender', '')}"
    ehr["demographics"] = demographics
    for field in _HV_EHR_FIELDS:
        ehr[field] = row.get(field, "") or ""
    return ehr


def load_hv_cases(csv_path: str | Path, urgency_levels: dict[str, str] | None = None) -> list[dict]:
    """Load HV cases from the clean CSV."""
    triage_ord = build_triage_ord(urgency_levels or DEFAULT_URGENCY_LEVELS)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["patient_vignette"])
    df = df.fillna("")

    cases: list[dict] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        gold_label = GOLD_LABEL_MAP.get(row_dict["triage_label"])
        if gold_label is None or gold_label not in triage_ord:
            continue

        cases.append({
            "case_id": str(row_dict["encounter_id"]),
            "dataset_name": "hv",
            "patient_vignette": row_dict["patient_vignette"],
            "ehr_json": _build_hv_ehr_json(row_dict),
            "prior_conversation": _parse_prior_conversation(row_dict.get("conversation", "")),
            "gold_triage": gold_label,
            "gold_triage_ord": triage_ord[gold_label],
            "gender": row_dict.get("gender", ""),
            "age": row_dict.get("age", ""),
            "race": "",
        })
    return cases


def _parse_av_conversation(raw: str) -> list[dict] | None:
    """Parse AV conversation column (uses 'Agent:' instead of 'Doctor:')."""
    if not raw or not raw.strip():
        return None
    messages: list[dict] = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.startswith("Patient:"):
            messages.append({"role": "patient", "content": line[8:].strip()})
        elif line.startswith("Agent:"):
            messages.append({"role": "doctor", "content": line[6:].strip()})
    return messages if messages else None


def _build_av_ehr_json(row: dict) -> dict:
    """Build EHR JSON from AV CSV columns."""
    ehr: dict = {}
    demographics = f"{row.get('age', '')}-year-old {row.get('sex', '')}"
    ehr["demographics"] = demographics
    for av_col, ehr_field in _AV_EHR_FIELDS.items():
        ehr[ehr_field] = row.get(av_col, "") or ""
    for field in [
        "review_of_systems", "current_medications", "allergies",
        "immunizations", "social_history", "family_history",
    ]:
        ehr[field] = ""
    return ehr


def load_av_cases(csv_path: str | Path, urgency_levels: dict[str, str] | None = None) -> list[dict]:
    """Load AV cases with triage mapping and EHR extraction."""
    triage_ord = build_triage_ord(urgency_levels or DEFAULT_URGENCY_LEVELS)

    df = pd.read_csv(csv_path)
    df = df.fillna("")

    cases: list[dict] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        gold_label = AV_TRIAGE_MAP.get(row_dict.get("grader_triage", ""))
        if gold_label is None or gold_label not in triage_ord:
            continue

        cases.append({
            "case_id": f"av_{row_dict['case']}",
            "dataset_name": "av",
            "patient_vignette": row_dict.get("patient_summary", ""),
            "ehr_json": _build_av_ehr_json(row_dict),
            "prior_conversation": _parse_av_conversation(
                row_dict.get("conversation", "")
            ),
            "gold_triage": gold_label,
            "gold_triage_ord": triage_ord[gold_label],
            "gender": row_dict.get("sex", ""),
            "age": row_dict.get("age", ""),
            "race": "",
        })
    return cases


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_LOADERS = {
    "hv": load_hv_cases,
    "av": load_av_cases,
}

_logger = logging.getLogger(__name__)


def load_all_cases(datasets_config: list[dict]) -> list[dict]:
    """Load and combine cases from all configured datasets.

    Each dataset entry must have ``type``, ``path``, and ``urgency_levels``.
    Errors in one dataset are logged and skipped so the rest still run.
    """
    all_cases: list[dict] = []
    for ds in datasets_config:
        try:
            loader = _LOADERS[ds["type"]]
            urgency_levels = ds.get("urgency_levels")
            cases = loader(ds["path"], urgency_levels=urgency_levels)
            for c in cases:
                c["urgency_levels"] = urgency_levels
            all_cases.extend(cases)
            _logger.info("Loaded %d cases from %s", len(cases), ds["name"])
        except Exception:
            _logger.exception("Failed to load dataset %s — skipping", ds.get("name", ds.get("type")))
    return all_cases


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------

def load_personas(
    personas_path: str | Path,
    include_baseline: bool = True,
) -> list[dict]:
    """Load persona definitions from personas.json as a conditions list."""
    with open(personas_path) as f:
        data = json.load(f)
    conditions: list[dict] = []
    if include_baseline:
        conditions.append({"name": "baseline", "params": {}})
    for persona in data["personas"]:
        conditions.append({
            "name": persona["name"],
            "params": persona["params"],
        })
    return conditions


def build_persona_params(condition: dict) -> dict:
    """Merge condition overrides on top of persona baseline params."""
    return {**PERSONA_BASELINE_PARAMS, **condition.get("params", {})}


# ---------------------------------------------------------------------------
# Job manifest
# ---------------------------------------------------------------------------

def build_jobs(
    cases: list[dict],
    conditions: list[dict],
    config: dict,
) -> list[dict]:
    """Generate the full job manifest: cases x personas x repetitions."""
    from src.nodes import (
        DEFAULT_MODEL_MAP,
    )

    n_reps = config["eval"].get("n_repetitions", 3)
    seed = config["eval"].get("seed", 42)
    model_map = config.get("patient_simulator", {}).get(
        "model_map", DEFAULT_MODEL_MAP
    )

    jobs: list[dict] = []
    for case in cases:
        for cond in conditions:
            sim_params = build_persona_params(cond)
            for rep in range(n_reps):
                jobs.append({
                    "job_id": str(uuid.uuid4()),
                    "case_id": case["case_id"],
                    "dataset_name": case["dataset_name"],
                    "patient_vignette": case["patient_vignette"],
                    "ehr_json": case.get("ehr_json"),
                    "prior_conversation": case.get("prior_conversation"),
                    "gold_triage": case["gold_triage"],
                    "gold_triage_ord": case["gold_triage_ord"],
                    "urgency_levels": case.get("urgency_levels"),
                    "condition_name": cond["name"],
                    "sim_params": sim_params,
                    "model_map": model_map,
                    "repetition": rep,
                    "seed": seed + rep,
                    "gender": case.get("gender", ""),
                    "age": case.get("age", ""),
                    "race": case.get("race", ""),
                })
    return jobs


def build_dataset(config: dict) -> dict:
    """Build the full multi-dataset manifest."""
    from collections import Counter

    include_baseline = config["eval"].get("include_baseline", True)
    conditions = load_personas(
        config["eval"]["personas_path"], include_baseline=include_baseline
    )
    cases = load_all_cases(config["eval"]["datasets"])
    jobs = build_jobs(cases, conditions, config)

    ds_counts = Counter(c["dataset_name"] for c in cases)

    return {
        "config": config,
        "cases": cases,
        "jobs": jobs,
        "stats": {
            "n_cases": len(cases),
            "n_conditions": len(conditions),
            "n_repetitions": config["eval"].get("n_repetitions", 3),
            "total_jobs": len(jobs),
            "per_dataset": dict(ds_counts),
        },
    }
