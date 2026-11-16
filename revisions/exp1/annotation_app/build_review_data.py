"""Build the Experiment 1 human-review task files used by grader_app.py.

Reads a downloaded parameter-adherence run and a downloaded clinical-fidelity run
and writes two self-contained JSON files (transcripts trimmed to role/content):

    data/parameter_adherence.json   N records x PARAMETER_PAIRS comparative tasks
    data/clinical_fidelity.json     N conversations with LLM-extracted claims

The run directories hold the pipeline outputs (adherence: conversations/ and comparative/; fidelity: conversations/
and fidelity/), each as one JSON file per job:

    python build_review_data.py --adherence_run_dir ./adherence_run --fidelity_run_dir ./fidelity_run
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent

# (param_name, active_value) pairs reviewed for every record, in display order.
PARAMETER_PAIRS = (
    ("anxiety", "anxious"),
    ("embarrassment", "embarrassed"),
    ("recall_level", "high"),
    ("recall_level", "low"),
    ("oversharing_tendency", "yes"),
    ("health_literacy", "high"),
    ("health_literacy", "low"),
    ("english_proficiency", "non_proficient"),
    ("typo_rate", "yes"),
    ("punctuation_style", "none"),
    ("capitalisation", "all_lower"),
    ("patient_goal", "active"),
)


def _load_dir(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def _turns(record: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": turn["role"], "content": turn["content"]} for turn in record.get("transcript", [])]


def _clean_vignette(text: str) -> str:
    text = text.strip()
    if len(text) > 1 and text[0] == text[-1] == '"':
        text = text[1:-1]
    return text.replace('\\"', '"')


def build_parameter_tasks(run_dir: Path, n_records: int, seed: int) -> list[dict[str, Any]]:
    conversations = {record["job_id"]: record for record in _load_dir(run_dir / "conversations")}
    comparatives = {
        (record["vignette_id"], record["param_name"], record["active_value"]): record
        for record in _load_dir(run_dir / "comparative")
    }
    vignette_ids = sorted({key[0] for key in comparatives})
    selected = sorted(random.Random(seed).sample(vignette_ids, n_records))
    rng = random.Random(seed)

    records = []
    for number, vignette_id in enumerate(selected, start=1):
        tasks = []
        for param_name, active_value in PARAMETER_PAIRS:
            comparative = comparatives[(vignette_id, param_name, active_value)]
            active = conversations[comparative["active_job_id"]]
            neutral = conversations[comparative["neutral_job_id"]]
            llm_picked_active = (comparative["more_active"] == "A") == comparative["a_is_active"]
            pair = [active, neutral]
            rng.shuffle(pair)  # independent of the order the LLM judge saw
            tasks.append({
                "task_id": f"{vignette_id}:{comparative['pair_key']}",
                "param_name": param_name,
                "param_label": active["param_label"],
                "param_description": active["param_description"],
                "active_value": active_value,
                "active_label": active["active_option"]["label"],
                "active_signals": active["active_option"].get("prompt_addition") or active["active_option"].get("description", ""),
                "neutral_label": active["neutral_option"]["label"],
                "patient_goal": active["param_value"] if param_name == "patient_goal" else "",
                "conversations": [{"job_id": record["job_id"], "transcript": _turns(record)} for record in pair],
                "activated_job_id": active["job_id"],
                "llm_selected_job_id": active["job_id"] if llm_picked_active else neutral["job_id"],
            })
        records.append({
            "record_id": f"R{number:02d}",
            "vignette_id": vignette_id,
            "case_id": comparative["case_id"],
            "tasks": tasks,
        })
    return records


def build_fidelity_tasks(run_dir: Path, n_records: int) -> list[dict[str, Any]]:
    conversations = {record["job_id"]: record for record in _load_dir(run_dir / "conversations")}
    fidelity = sorted(_load_dir(run_dir / "fidelity"), key=lambda record: record["job_id"])
    records = []
    for number, record in enumerate(fidelity[:n_records], start=1):
        records.append({
            "record_id": f"F{number:02d}",
            "job_id": record["job_id"],
            "vignette_id": record["vignette_id"],
            "case_id": record["case_id"],
            "vignette_text": _clean_vignette(record["vignette_text"]),
            "transcript": _turns(conversations[record["job_id"]]),
            "vignette_claims": record["vignette_claims"],
            "conversation_claims": record["conversation_patient_claims"],
            "llm_classified_claims": record["classified_claims"],
            "llm_vignette_claims_not_mentioned": [item["claim"] for item in record["vignette_claims_not_mentioned"]],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adherence_run_dir", type=Path, required=True)
    parser.add_argument("--fidelity_run_dir", type=Path, required=True)
    parser.add_argument("--n_records", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    parameter = build_parameter_tasks(args.adherence_run_dir, args.n_records, args.seed)
    fidelity = build_fidelity_tasks(args.fidelity_run_dir, args.n_records)
    (out_dir / "parameter_adherence.json").write_text(json.dumps(parameter, indent=1))
    (out_dir / "clinical_fidelity.json").write_text(json.dumps(fidelity, indent=1))
    n_tasks = sum(len(record["tasks"]) for record in parameter)
    print(f"Wrote {len(parameter)} adherence records ({n_tasks} tasks) and {len(fidelity)} fidelity records to {out_dir}")


if __name__ == "__main__":
    main()
