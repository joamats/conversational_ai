"""Data loading and extraction for the realism equivalence trial.

Conversation format (one JSON object per line in JSONL):
    {
        "id":         "conv_001",
        "transcript": [{"role": "doctor", "content": "..."}, {"role": "patient", "content": "..."}],
        "metadata":   {"case_id": "726", ...}   # optional, never shown to raters
    }

Two source files:
    simulated.jsonl  — AI-patient conversations (from patient simulator eval runs)
    real.jsonl       — real-patient conversations (placeholder until replaced with actual data)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

REALISM_DIR = Path(__file__).parent
DATA_DIR = REALISM_DIR.parent / "data"
DEFAULT_SIMULATED_PATH = DATA_DIR / "simulated.jsonl"
DEFAULT_REAL_PATH = DATA_DIR / "real.jsonl"


def load_conversations(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    convos = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                convos.append(json.loads(line))
    return convos


def save_conversations(convos: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in convos:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def sample_test_set(
    simulated: list[dict],
    real: list[dict],
    n_simulated: int = 5,
    n_real: int = 5,
    seed: int | None = None,
) -> list[dict]:
    """Sample and shuffle a balanced test set.

    Each item gets a `_ground_truth` field ("simulated" or "real") and
    `_source_file` for provenance. The order is randomised.

    Works when only one source is available — the other count is set to 0.
    """
    rng = random.Random(seed)

    if not simulated:
        n_simulated = 0
    if not real:
        n_real = 0

    sim_sample = rng.sample(simulated, min(n_simulated, len(simulated))) if simulated else []
    real_sample = rng.sample(real, min(n_real, len(real))) if real else []

    items = []
    for c in sim_sample:
        items.append({**c, "_ground_truth": "simulated", "_source_file": "simulated.jsonl"})
    for c in real_sample:
        items.append({**c, "_ground_truth": "real", "_source_file": "real.jsonl"})

    rng.shuffle(items)
    return items


def extract_from_eval_run(run_dir: str | Path, max_convos: int | None = None) -> list[dict]:
    """Extract conversations from an eval run's conversations/ directory.

    Filters to conversations with at least 3 turns and no errors.
    Returns list in the standard JSONL conversation format.
    """
    conv_dir = Path(run_dir) / "conversations"
    if not conv_dir.exists():
        return []

    convos = []
    for path in sorted(conv_dir.glob("*.json")):
        if path.stem.endswith(("_judged", "_comparative")):
            continue
        with open(path) as f:
            raw = json.load(f)

        if raw.get("error"):
            continue
        transcript = raw.get("transcript", [])
        if len(transcript) < 3:
            continue

        convos.append({
            "id": raw["job_id"],
            "transcript": transcript,
            "metadata": {
                "case_id": raw.get("case_id"),
                "job_type": raw.get("job_type"),
                "param_name": raw.get("param_name"),
                "param_value": raw.get("param_value"),
                "source_run": Path(run_dir).name,
            },
        })

    if max_convos and len(convos) > max_convos:
        convos = random.sample(convos, max_convos)
    return convos


def build_initial_data(run_dir: str | Path, n_per_file: int = 10) -> tuple[Path, Path]:
    """Extract conversations from an eval run and write both simulated.jsonl and real.jsonl.

    For now both files come from the simulator (real.jsonl is a placeholder).
    The two files sample different conversations so they don't overlap.
    """
    all_convos = extract_from_eval_run(run_dir)
    random.shuffle(all_convos)

    n = min(n_per_file, len(all_convos) // 2)
    simulated = all_convos[:n]
    real_placeholder = all_convos[n : n + n]

    for i, c in enumerate(simulated):
        c["id"] = f"sim_{i:03d}"
    for i, c in enumerate(real_placeholder):
        c["id"] = f"real_{i:03d}"

    save_conversations(simulated, DEFAULT_SIMULATED_PATH)
    save_conversations(real_placeholder, DEFAULT_REAL_PATH)

    return DEFAULT_SIMULATED_PATH, DEFAULT_REAL_PATH
