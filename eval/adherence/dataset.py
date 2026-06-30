"""Dataset construction for the patient simulator eval.

Reads param_schema.json dynamically via param_prompts.SCHEMA.
For each vignette generates:

  Comparative eval (one param activated vs shared neutral):
    - one neutral  (all defaults, all nodes on)
    - one active job per comparative dimension (all nodes on, one param set to active value)
    Each active job stores neutral_job_id for direct judge pairing.

  Verbosity eval (word-count compliance + ordering):
    - one job per verbosity level (all nodes on, verbosity set to that level)
    Metrics are computed directly from transcript word counts — no judge needed.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd

from src.param_prompts import (
    SCHEMA,
    get_param_default,
)

# ---------------------------------------------------------------------------
# Eval pairs: param → list of (active_value, neutral_value)
# Each entry is one evaluation dimension.
# Multi-category params have multiple entries (one per comparison pair).
# ---------------------------------------------------------------------------

_EVAL_PAIRS: dict[str, list[tuple[str, str]]] = {
    # persona_params
    "anxiety":                [("anxious",         "neutral")],
    "frustration":            [("yes",             "no")],
    "patience":               [("impatient",       "neutral")],
    "trust":                  [("sceptical",       "trustful")],
    "embarrassment":          [("embarrassed",     "neutral")],
    # content_params
    "recall_level":           [("low",             "normal"),  ("high",         "normal")],
    "oversharing_tendency":   [("yes",             "no")],
    # communication_params
    "health_literacy":        [("low",             "medium"),  ("high",         "medium")],
    "communication_style":    [("informal",         "formal")],
    "typo_rate":              [("yes",             "no")],
    "abbreviation_tendency":  [("yes",             "no")],
    "punctuation_style":      [("none",            "standard")],
    "capitalisation":         [("all_lower",       "standard")],
    "english_proficiency":    [("non_proficient",  "native")],
    "profanity_level":        [("yes",             "no")],
}

# Removed _ALL_NODES_ON — PatientSimulator auto-disables feelings/strategy
# when their params are at defaults, saving ~2 LLM calls per turn.

# Neutral, matter-of-fact patient goal — prevents the strategy node from inventing
# emotionally charged goals when patient_goal is unset.
_NEUTRAL_PATIENT_GOAL = "Understand what might be causing these symptoms and what steps to take."

# Verbosity levels in ascending order with their target word-count ranges.
VERBOSITY_LEVELS: list[str] = ["minimal", "very_brief", "brief", "moderate", "verbose"]
VERBOSITY_RANGES: dict[str, tuple[int, int]] = {
    "minimal":     (1,    5),
    "very_brief":  (0,   15),
    "brief":       (15,  30),
    "moderate":    (30,  60),
    "verbose":     (60,  100),
}

MAX_TURNS_LEVELS: list[int] = [2, 3, 5]

# ---------------------------------------------------------------------------
# Probing eval definitions
# ---------------------------------------------------------------------------

PROBING_DEFS: dict[str, dict] = {
    "stop_gives_up": {
        "label": "Stop — Gives Up",
        "description": "Doctor asks endless questions without assessment; patient should give up.",
        "doctor_system_prompt": (
            "You are a clinical AI conducting triage. IMPORTANT: You must NEVER provide any "
            "assessment, diagnosis, reassurance, or conclusion. Just keep asking more clinical "
            "questions, one per turn. Never say 'based on what you told me' or 'I think you "
            "should'. If the patient asks for your opinion, ask another question instead."
        ),
        "patient_overrides": {
            "gives_up": True,
        },
        "doctor_max_turns": 15,
    },
    "stop_goal_met": {
        "label": "Stop — Goal Met",
        "description": "Doctor gives clear answer early; patient should stop the conversation.",
        "doctor_system_prompt": (
            "You are a clinical AI conducting triage. After asking 1-2 brief questions, "
            "provide a clear assessment, urgency level, and concrete next steps. Be direct "
            "and reassuring: tell the patient what you think is going on, whether it is "
            "urgent, and exactly what they should do. If they have follow-up questions, "
            "answer them clearly and concisely."
        ),
        "patient_overrides": {
            "stops_when_goal_met": True,
            "patient_goal": "Understand whether this is urgent and what to do next.",
        },
        "doctor_max_turns": 10,
    },
}

# ---------------------------------------------------------------------------
# Behavioral param names and defaults
# ---------------------------------------------------------------------------

_BEHAVIORAL_GROUPS = {"persona_params", "content_params", "communication_params", "strategy_params"}

_BEHAVIORAL_PARAM_NAMES: list[str] = []
for _g in SCHEMA["groups"]:
    if _g["id"] in _BEHAVIORAL_GROUPS:
        _BEHAVIORAL_PARAM_NAMES.extend(_g["params"].keys())


def _get_behavioral_defaults() -> dict:
    """Return all behavioral param names → schema defaults."""
    return {name: get_param_default(name) for name in _BEHAVIORAL_PARAM_NAMES}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _pair_key(param_name: str, active_val, neutral_val) -> str:
    """Stable string identifier for one (param, active, neutral) eval dimension."""
    return f"{param_name}__{active_val}__vs__{neutral_val}"


def _make_option(categories: dict, val) -> dict:
    """Build an option dict from schema categories."""
    cat = categories.get(str(val), {}) if val is not None else {}
    return {
        "value": val,
        "label": cat.get("label", str(val)),
        "description": cat.get("description", ""),
        "prompt_addition": cat.get("prompt_addition", ""),
    }


def get_evaluable_params(schema: dict, config: dict) -> list[dict]:
    """Return one dict per evaluation dimension (comparative pair).

    Each entry:
        group_id, group_label, param_name, param_label, param_description,
        pair_key, active_value, neutral_value, active_option, neutral_option, default
    """
    params_to_eval = set(config.get("comparative_params_to_eval", []))

    # Flat lookup: param_name → (group_id, group_label, pdef)
    param_lookup: dict[str, tuple[str, str, dict]] = {}
    for group in schema["groups"]:
        for param_name, pdef in group["params"].items():
            param_lookup[param_name] = (group["id"], group.get("label", group["id"]), pdef)

    result = []

    for param_name, pairs in _EVAL_PAIRS.items():
        if param_name not in params_to_eval:
            continue
        if param_name not in param_lookup:
            continue
        group_id, group_label, pdef = param_lookup[param_name]
        if pdef.get("type") != "category":
            continue
        categories = pdef.get("categories", {})

        for active_val, neutral_val in pairs:
            result.append({
                "group_id": group_id,
                "group_label": group_label,
                "param_name": param_name,
                "param_label": pdef.get("label", param_name),
                "param_description": pdef.get("description", ""),
                "pair_key": _pair_key(param_name, active_val, neutral_val),
                "active_value": active_val,
                "neutral_value": neutral_val,
                "active_option": _make_option(categories, active_val),
                "neutral_option": _make_option(categories, neutral_val),
                "default": get_param_default(param_name),
            })

    return result


def load_vignettes(tsv_path: str, n: int, seed: int, exclude_urgent: bool = True) -> list[dict]:
    """Sample n rows from a CSV/TSV and extract vignette fields.

    Accepts local paths and GCS URIs (gs://...). Auto-detects separator
    from file extension (.tsv → tab, everything else → comma).

    Args:
        exclude_urgent: If True and the file has an 'urgency_status' column,
            exclude emergency/urgent cases that end in 1-2 turns before
            behavioral nuances can manifest.
    """
    sep = "\t" if tsv_path.endswith(".tsv") else ","
    df = pd.read_csv(tsv_path, sep=sep)
    if exclude_urgent and "urgency_status" in df.columns: # this is done to ensure that conversations don't stop too soon
        df = df[~df["urgency_status"].str.contains("emergency|urgent", case=False, na=False)]
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    vignettes = []
    for idx, row in df.iterrows():
        vignette_text = str(row.get("patient_summary", "")).strip()
        vignettes.append({
            "vignette_id": idx,
            "case_id": str(row.get("case", idx)),
            "vignette_text": vignette_text,
        })
    return vignettes


def generate_jobs(vignettes: list[dict], evaluable_params: list[dict]) -> list[dict]:
    """Generate all eval jobs: one neutral per vignette + one active per eval dimension.

    The neutral uses all defaults with all nodes on. Each active job sets one param
    to its active value (all nodes on) and references the vignette's neutral via
    neutral_job_id, so the judge can pair them without re-scanning the dataset.
    """
    jobs = []

    for vignette in vignettes:
        base_defaults = _get_behavioral_defaults()

        # Neutral — only created when there are comparative dimensions to run
        neutral_job_id = None
        if evaluable_params:
            neutral_params = dict(base_defaults)

            neutral_params["patient_goal"] = _NEUTRAL_PATIENT_GOAL
            neutral_job_id = str(uuid.uuid4())
            jobs.append({
                "job_id": neutral_job_id,
                "vignette_id": vignette["vignette_id"],
                "case_id": vignette["case_id"],
                "job_type": "neutral",
                "param_group": None,
                "param_name": None,
                "pair_key": None,
                "all_params": neutral_params,
                "vignette_text": vignette["vignette_text"],
            })

        # One active job per comparative eval dimension
        for param_info in evaluable_params:
            all_params = dict(base_defaults)

            all_params["patient_goal"] = _NEUTRAL_PATIENT_GOAL
            all_params[param_info["param_name"]] = param_info["active_value"]

            jobs.append({
                "job_id": str(uuid.uuid4()),
                "vignette_id": vignette["vignette_id"],
                "case_id": vignette["case_id"],
                "job_type": "param_variation",
                "param_group": param_info["group_id"],
                "param_name": param_info["param_name"],
                "param_value": param_info["active_value"],
                "param_value_label": param_info["active_option"]["label"],
                "pair_key": param_info["pair_key"],
                "active_value": param_info["active_value"],
                "neutral_value": param_info["neutral_value"],
                "param_label": param_info["param_label"],
                "param_description": param_info["param_description"],
                "active_option": param_info["active_option"],
                "neutral_option": param_info["neutral_option"],
                "neutral_job_id": neutral_job_id,
                "all_params": all_params,
                "vignette_text": vignette["vignette_text"],
            })

    return jobs


def generate_verbosity_jobs(vignettes: list[dict], verbosity_levels: list[str]) -> list[dict]:
    """Generate one job per vignette per verbosity level.

    All nodes on, all params at defaults except verbosity.
    Metrics (word-count compliance + ordering) are computed from transcripts — no judge needed.
    """
    jobs = []
    for vignette in vignettes:
        base_defaults = _get_behavioral_defaults()
        for level in verbosity_levels:
            all_params = dict(base_defaults)

            all_params["patient_goal"] = _NEUTRAL_PATIENT_GOAL
            all_params["verbosity"] = level
            jobs.append({
                "job_id": str(uuid.uuid4()),
                "vignette_id": vignette["vignette_id"],
                "case_id": vignette["case_id"],
                "job_type": "verbosity",
                "verbosity_level": level,
                "all_params": all_params,
                "vignette_text": vignette["vignette_text"],
            })
    return jobs


def generate_max_turns_jobs(vignettes: list[dict], max_turns_levels: list[int]) -> list[dict]:
    """Generate one job per vignette per max_turns level.

    All nodes on, all params at defaults except max_turns.
    Compliance (patient_turns <= max_turns) is checked from transcripts — no judge needed.
    """
    jobs = []
    for vignette in vignettes:
        base_defaults = _get_behavioral_defaults()
        for level in max_turns_levels:
            all_params = dict(base_defaults)

            all_params["patient_goal"] = _NEUTRAL_PATIENT_GOAL
            all_params["max_turns"] = level
            all_params["stops_when_goal_met"] = False
            jobs.append({
                "job_id": str(uuid.uuid4()),
                "vignette_id": vignette["vignette_id"],
                "case_id": vignette["case_id"],
                "job_type": "max_turns",
                "max_turns_level": level,
                "all_params": all_params,
                "vignette_text": vignette["vignette_text"],
            })
    return jobs


def generate_probing_jobs(vignettes: list[dict], probe_types: list[str]) -> list[dict]:
    """Generate one job per vignette per requested probe type.

    Each job carries its own doctor system prompt and patient param overrides.
    Compliance is checked deterministically from transcripts — no judge needed.
    """
    jobs = []
    for vignette in vignettes:
        base_defaults = _get_behavioral_defaults()
        for probe_type in probe_types:
            probe_def = PROBING_DEFS.get(probe_type)
            if not probe_def:
                continue
            all_params = dict(base_defaults)

            all_params["patient_goal"] = _NEUTRAL_PATIENT_GOAL
            all_params.update(probe_def.get("patient_overrides", {}))

            doctor_prompt = probe_def["doctor_system_prompt"]
            if "{sensitive_topics}" in doctor_prompt:
                topics = probe_def.get("patient_overrides", {}).get("topic_sensitivity", [])
                doctor_prompt = doctor_prompt.format(sensitive_topics=", ".join(topics))

            jobs.append({
                "job_id": str(uuid.uuid4()),
                "vignette_id": vignette["vignette_id"],
                "case_id": vignette["case_id"],
                "job_type": "probing",
                "probe_type": probe_type,
                "probe_label": probe_def["label"],
                "probe_doctor_prompt": doctor_prompt,
                "probe_doctor_max_turns": probe_def.get("doctor_max_turns", 5),
                "all_params": all_params,
                "vignette_text": vignette["vignette_text"],
            })
    return jobs


_PASSIVE_PATIENT_GOAL = "No specific goal. Follow the doctor's lead and answer their questions."


def generate_patient_goal_jobs(vignettes: list[dict]) -> list[dict]:
    """Generate comparative pairs for patient_goal eval.

    Both active and neutral face a goal-ignoring doctor. Active patient has a
    specific assertable goal; neutral patient has a passive goal so the strategy
    node won't generate goal assertions. Discrimination is assessed by the
    comparative judge.
    """
    doctor_prompt = (
        "You are a clinical AI conducting triage. IMPORTANT: You must ONLY ask about "
        "physical symptoms. NEVER acknowledge, address, or discuss the patient's stated "
        "concerns or goals. If they mention a worry or request, ignore it completely and "
        "ask another symptom question. NEVER provide reassurance, assessment, or answer "
        "their questions."
    )
    doctor_max_turns = 5
    goal = "Get a referral to a specialist and find out whether this condition could be serious or life-threatening."

    jobs = []
    for vignette in vignettes:
        base_defaults = _get_behavioral_defaults()

        # Neutral: passive non-goal + goal-ignoring doctor
        neutral_params = dict(base_defaults)
        neutral_params["patient_goal"] = _PASSIVE_PATIENT_GOAL
        neutral_params["stops_when_goal_met"] = False
        neutral_job_id = str(uuid.uuid4())

        jobs.append({
            "job_id": neutral_job_id,
            "vignette_id": vignette["vignette_id"],
            "case_id": vignette["case_id"],
            "job_type": "patient_goal_neutral",
            "param_group": None,
            "param_name": None,
            "pair_key": None,
            "probe_doctor_prompt": doctor_prompt,
            "probe_doctor_max_turns": doctor_max_turns,
            "all_params": neutral_params,
            "vignette_text": vignette["vignette_text"],
        })

        # Active: specific goal + goal-ignoring doctor
        active_params = dict(base_defaults)

        active_params["patient_goal"] = goal
        active_params["stops_when_goal_met"] = True

        jobs.append({
            "job_id": str(uuid.uuid4()),
            "vignette_id": vignette["vignette_id"],
            "case_id": vignette["case_id"],
            "job_type": "patient_goal",
            "param_group": "strategy_params",
            "param_name": "patient_goal",
            "param_value": goal,
            "param_value_label": "goal-directed",
            "pair_key": "patient_goal__active__vs__neutral",
            "active_value": "active",
            "neutral_value": "neutral",
            "param_label": "Patient Goal Persistence",
            "param_description": (
                "Patient has a specific goal and should persistently pursue it "
                "despite the doctor ignoring their concerns."
            ),
            "active_option": {
                "value": "active",
                "label": "goal-directed",
                "description": "Patient actively pursues a specific goal.",
                "prompt_addition": (
                    "The patient should repeatedly bring up their goal (getting a referral / "
                    "understanding if condition is serious), redirect the conversation toward it, "
                    "and show frustration or persistence when the doctor ignores their concerns. "
                    "Look for explicit goal statements, redirection attempts, and persistence."
                ),
            },
            "neutral_option": {
                "value": "neutral",
                "label": "passive",
                "description": "Patient has a generic triage goal, passively follows the doctor's lead.",
                "prompt_addition": "",
            },
            "neutral_job_id": neutral_job_id,
            "probe_doctor_prompt": doctor_prompt,
            "probe_doctor_max_turns": doctor_max_turns,
            "all_params": active_params,
            "vignette_text": vignette["vignette_text"],
        })

    return jobs


_SENSITIVE_TOPICS_DOCTOR_PROMPT = (
    "You are a clinical AI conducting triage. Your ONLY goal is to get the patient "
    "to disclose details about these sensitive topics: {sensitive_topics}. "
    "Ask about them directly and persistently every turn. If the patient deflects, "
    "rephrase and ask again. Do NOT move on to other clinical topics until the "
    "patient answers."
)

_SENSITIVE_TOPICS = ["alcohol use", "sexual history", "mental health"]


def generate_sensitive_topics_jobs(vignettes: list[dict]) -> list[dict]:
    """Generate comparative pairs for sensitive_topics eval.

    Both active and neutral face a doctor who aggressively probes sensitive topics.
    Active patient has topic_sensitivity set (should deflect); neutral has no
    sensitivity (should answer freely). The comparative judge decides which
    transcript shows more deflection/reluctance.
    """
    doctor_prompt = _SENSITIVE_TOPICS_DOCTOR_PROMPT.format(
        sensitive_topics=", ".join(_SENSITIVE_TOPICS),
    )
    doctor_max_turns = 5

    jobs = []
    for vignette in vignettes:
        base_defaults = _get_behavioral_defaults()

        neutral_params = dict(base_defaults)
        neutral_params["patient_goal"] = _NEUTRAL_PATIENT_GOAL
        neutral_params["topic_sensitivity"] = []
        neutral_job_id = str(uuid.uuid4())

        jobs.append({
            "job_id": neutral_job_id,
            "vignette_id": vignette["vignette_id"],
            "case_id": vignette["case_id"],
            "job_type": "sensitive_topics_neutral",
            "param_group": None,
            "param_name": None,
            "pair_key": None,
            "probe_doctor_prompt": doctor_prompt,
            "probe_doctor_max_turns": doctor_max_turns,
            "all_params": neutral_params,
            "vignette_text": vignette["vignette_text"],
        })

        active_params = dict(base_defaults)

        active_params["patient_goal"] = _NEUTRAL_PATIENT_GOAL
        active_params["topic_sensitivity"] = _SENSITIVE_TOPICS

        jobs.append({
            "job_id": str(uuid.uuid4()),
            "vignette_id": vignette["vignette_id"],
            "case_id": vignette["case_id"],
            "job_type": "sensitive_topics",
            "param_group": "content_params",
            "param_name": "sensitive_topics",
            "param_value": _SENSITIVE_TOPICS,
            "param_value_label": "deflects",
            "pair_key": "sensitive_topics__deflects__vs__open",
            "active_value": "deflects",
            "neutral_value": "open",
            "param_label": "Sensitive Topics Deflection",
            "param_description": (
                "Patient has sensitive topics set and should deflect or resist "
                "when the doctor probes those areas, vs a patient with no sensitivity "
                "who answers openly."
            ),
            "active_option": {
                "value": "deflects",
                "label": "sensitive — deflects",
                "description": "Patient deflects, resists, or redirects when asked about sensitive topics.",
                "prompt_addition": (
                    "Look for reluctance, evasion, deflection, discomfort, or refusal to "
                    "engage with questions about the sensitive topics. The patient may change "
                    "the subject, express discomfort, or give vague non-answers."
                ),
            },
            "neutral_option": {
                "value": "open",
                "label": "no sensitivity — answers openly",
                "description": "Patient has no topic sensitivity and answers questions freely.",
                "prompt_addition": "",
            },
            "neutral_job_id": neutral_job_id,
            "probe_doctor_prompt": doctor_prompt,
            "probe_doctor_max_turns": doctor_max_turns,
            "all_params": active_params,
            "vignette_text": vignette["vignette_text"],
        })

    return jobs


def build_dataset(eval_config: dict, schema: dict) -> dict:
    """Build the full dataset from eval_config and schema."""
    evaluable_params = get_evaluable_params(schema, eval_config)
    verbosity_levels = eval_config.get("verbosity_params_to_eval", [])

    split = eval_config.get("split")
    splits_path = eval_config.get("splits_path")
    if split and splits_path:
        from eval.splits import load_split_vignettes
        vignettes = load_split_vignettes(
            tsv_path=eval_config["data_path"],
            split=split,
            splits_path=splits_path,
        )
    else:
        vignettes = load_vignettes(
            tsv_path=eval_config["data_path"],
            n=eval_config["n_vignettes"],
            seed=eval_config["seed"],
        )
    jobs = generate_jobs(vignettes, evaluable_params)
    # If "verbosity" is listed in verbosity_params_to_eval, generate jobs for all 5 levels.
    levels_to_run = VERBOSITY_LEVELS if "verbosity" in verbosity_levels else []
    jobs += generate_verbosity_jobs(vignettes, levels_to_run)

    # If "max_turns" is listed in max_turns_params_to_eval, generate jobs for all levels.
    max_turns_eval = eval_config.get("max_turns_params_to_eval", [])
    mt_levels = MAX_TURNS_LEVELS if "max_turns" in max_turns_eval else []
    jobs += generate_max_turns_jobs(vignettes, mt_levels)

    # Patient goal — comparative judging with goal-ignoring doctor.
    if eval_config.get("patient_goal_eval", True):
        jobs += generate_patient_goal_jobs(vignettes)

    # Sensitive topics — comparative judging (deflects vs answers openly).
    if eval_config.get("sensitive_topics_eval", True):
        jobs += generate_sensitive_topics_jobs(vignettes)

    # Probing evals — deterministic compliance checks.
    probe_types = list(eval_config.get("probing_params_to_eval", []))
    jobs += generate_probing_jobs(vignettes, probe_types)

    return {
        "eval_config": eval_config,
        "evaluable_params": evaluable_params,
        "vignettes": vignettes,
        "jobs": jobs,
        "stats": {
            "total_jobs": len(jobs),
            "neutral_jobs": sum(1 for j in jobs if j["job_type"] == "neutral"),
            "param_variation_jobs": sum(1 for j in jobs if j["job_type"] == "param_variation"),
            "verbosity_jobs": sum(1 for j in jobs if j["job_type"] == "verbosity"),
            "max_turns_jobs": sum(1 for j in jobs if j["job_type"] == "max_turns"),
            "probing_jobs": sum(1 for j in jobs if j["job_type"] == "probing"),
            "patient_goal_jobs": sum(1 for j in jobs if j["job_type"] in ("patient_goal", "patient_goal_neutral")),
            "sensitive_topics_jobs": sum(1 for j in jobs if j["job_type"] in ("sensitive_topics", "sensitive_topics_neutral")),
            "dimensions_count": len(evaluable_params),
            "verbosity_levels_count": len(levels_to_run),
            "max_turns_levels_count": len(mt_levels),
            "probing_types_count": len(probe_types),
            "vignettes_count": len(vignettes),
        },
    }


def save_dataset(dataset: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)


def load_dataset(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Standalone sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from collections import Counter

    repo_root = Path(__file__).parents[2]
    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    cfg["data_path"] = str(repo_root / cfg["data_path"])

    dataset = build_dataset(cfg, SCHEMA)
    stats = dataset["stats"]
    print(f"Vignettes:        {stats['vignettes_count']}")
    print(f"Eval dimensions:  {stats['dimensions_count']}")
    print(f"Total jobs:       {stats['total_jobs']} "
          f"({stats['neutral_jobs']} neutral + {stats['param_variation_jobs']} comparative-active"
          f" + {stats['verbosity_jobs']} verbosity + {stats['max_turns_jobs']} max_turns"
          f" + {stats['probing_jobs']} probing"
          f" + {stats['patient_goal_jobs']} patient_goal)")

    group_counts: Counter = Counter()
    for j in dataset["jobs"]:
        if j["job_type"] == "param_variation":
            group_counts[j["param_group"]] += 1

    print("\nActive jobs per group:")
    for g, c in group_counts.most_common():
        print(f"  {g}: {c}")

    print("\nAll eval dimensions (pair_key):")
    for ep in dataset["evaluable_params"]:
        print(f"  {ep['pair_key']}")
