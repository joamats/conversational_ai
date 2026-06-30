"""Run the persona compound fairness experiment locally (no Dataflow).

Builds the dataset, runs each doctor-patient conversation job in a local thread
pool, then computes per-condition / per-dataset triage metrics and cost.

Usage:

    python -m fairness.run \
        --config fairness/configs/config_gpt_54_mini.json \
        --n_cases 2

    # Recompute metrics on an existing run directory:
    python -m fairness.run \
        --config .../config_gpt_54_mini.json \
        --metrics_only --output_dir_name persona_20260601_120000_abcdef12

Requires an OpenAI API key (see ``eval/_utils.get_openai_api_key``).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eval._utils import (
    atomic_write_json,
    compute_cost_from_results,
    get_openai_api_key,
    load_config,
    print_cost_summary,
    setup_logging,
)
from fairness.dataset import (
    DEFAULT_URGENCY_LEVELS,
    build_dataset,
    build_jobs,
    build_triage_ord,
    load_all_cases,
    load_personas,
)

_PERSONAS_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 5


# ---------------------------------------------------------------------------
# Voting + direction
# ---------------------------------------------------------------------------

def majority_vote(predictions: list[str | None], triage_ord: dict[str, int]) -> str | None:
    """Majority vote across repetitions; tie-break picks the more severe label."""
    valid = [p for p in predictions if p is not None]
    if not valid:
        return None
    counts = Counter(valid)
    max_count = max(counts.values())
    tied = [label for label, c in counts.items() if c == max_count]
    if len(tied) == 1:
        return tied[0]
    return max(tied, key=lambda l: triage_ord.get(l, -1))


def triage_direction(predicted: str | None, gold: str, triage_ord: dict[str, int]) -> str:
    """Classify a prediction as correct / over / under / invalid."""
    if predicted is None:
        return "invalid"
    pred_ord = triage_ord.get(predicted)
    gold_ord = triage_ord.get(gold)
    if pred_ord is None or gold_ord is None:
        return "invalid"
    if pred_ord == gold_ord:
        return "correct"
    return "over" if pred_ord > gold_ord else "under"


def _build_triage_ord_for_result(r: dict) -> dict[str, int]:
    ul = r.get("urgency_levels")
    return build_triage_ord(ul) if ul else build_triage_ord(DEFAULT_URGENCY_LEVELS)


# ---------------------------------------------------------------------------
# Bootstrap and metrics
# ---------------------------------------------------------------------------

def _bootstrap_rate(values: list[int | float], n_boot: int = 10_000, ci: float = 0.95, seed: int = 42) -> dict:
    """Mean and bootstrap 95% CI for a binary 0/1 list."""
    arr = np.array(values, dtype=float)
    mean = float(arr.mean()) if len(arr) else 0.0
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ]) if len(arr) else np.array([0.0])
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return {"mean": mean, "ci_95": [lo, hi], "n": len(values)}


def _compute_rates_by_group(voted: list[dict], group_key: str) -> dict:
    """Compute over/under/accuracy rates grouped by a key."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for v in voted:
        groups[v[group_key]].append(v)

    result: dict[str, dict] = {}
    for name, cases in sorted(groups.items()):
        over = [1 if c["direction"] == "over" else 0 for c in cases]
        under = [1 if c["direction"] == "under" else 0 for c in cases]
        correct = [1 if c["direction"] == "correct" else 0 for c in cases]
        invalid = sum(1 for c in cases if c["direction"] == "invalid")
        result[name] = {
            "over_triage": _bootstrap_rate(over),
            "under_triage": _bootstrap_rate(under),
            "accuracy": _bootstrap_rate(correct),
            "n_cases": len(cases),
            "n_invalid": invalid,
        }
    return result


def compute_persona_metrics(results: list[dict]) -> dict:
    """Compute per-persona AND per-dataset triage metrics with majority voting."""
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in results:
        groups[(r["case_id"], r["condition_name"])].append(r)

    voted: list[dict] = []
    for (case_id, cond_name), reps in groups.items():
        rep_triage_ord = _build_triage_ord_for_result(reps[0])
        preds = [r["predicted_triage"] for r in reps]
        winner = majority_vote(preds, rep_triage_ord)
        gold = reps[0]["gold_triage"]
        voted.append({
            "case_id": case_id,
            "dataset_name": reps[0]["dataset_name"],
            "condition_name": cond_name,
            "gold_triage": gold,
            "predicted_triage": winner,
            "direction": triage_direction(winner, gold, rep_triage_ord),
            "n_reps": len(reps),
        })

    per_condition = _compute_rates_by_group(voted, "condition_name")
    per_dataset = _compute_rates_by_group(voted, "dataset_name")
    per_dataset_condition: dict[str, dict] = {}
    for ds_name in sorted({v["dataset_name"] for v in voted}):
        ds_voted = [v for v in voted if v["dataset_name"] == ds_name]
        per_dataset_condition[ds_name] = _compute_rates_by_group(ds_voted, "condition_name")

    return {
        "per_condition": per_condition,
        "per_dataset": per_dataset,
        "per_dataset_condition": per_dataset_condition,
        "voted_results": voted,
        "total_results": len(results),
        "total_voted": len(voted),
    }


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------

def _run_job_with_retries(job: dict, doctor_config: dict) -> dict:
    """Run one conversation job with retry/backoff (local, no GCS)."""
    from fairness.simulation import (
        run_persona_conversation,
    )

    result = None
    attempt = 0
    for attempt in range(_MAX_RETRIES):
        result = run_persona_conversation(job, doctor_config)
        if result.get("error") is None:
            break
        logger.warning("Job %s attempt %d failed: %s", job["job_id"][:8], attempt + 1, result["error"])
        if attempt < _MAX_RETRIES - 1:
            time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
    result["attempts"] = attempt + 1
    return result


def _load_local_results(local_dir: str | Path) -> list[dict]:
    conv_dir = Path(local_dir) / "conversations"
    results = []
    for p in sorted(conv_dir.glob("*.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_condition_table(metrics: dict) -> None:
    print(f"  {'Condition':<25} {'Over%':>10} {'Under%':>10} {'Acc%':>10} {'N':>5}")
    print("  " + "-" * 65)
    for cond, m in sorted(metrics.items()):
        o, u, a = m["over_triage"], m["under_triage"], m["accuracy"]
        print(
            f"  {cond:<25} "
            f"{o['mean']:5.1%} [{o['ci_95'][0]:5.1%},{o['ci_95'][1]:5.1%}] "
            f"{u['mean']:5.1%} [{u['ci_95'][0]:5.1%},{u['ci_95'][1]:5.1%}] "
            f"{a['mean']:5.1%} "
            f"{m['n_cases']:>5}"
        )


def _print_summary(metrics: dict) -> None:
    print("\n=== Overall (all datasets) ===")
    _print_condition_table(metrics["per_condition"])
    for ds_name, ds_metrics in sorted(metrics["per_dataset_condition"].items()):
        n = metrics["per_dataset"].get(ds_name, {}).get("n_cases", "?")
        print(f"\n=== {ds_name} (n={n}) ===")
        _print_condition_table(ds_metrics)
    print()


# ---------------------------------------------------------------------------
# Dataset build (optionally limited to N cases per dataset)
# ---------------------------------------------------------------------------

def _build_dataset(config: dict, n_cases: int | None) -> dict:
    if n_cases is None:
        return build_dataset(config)

    cases = load_all_cases(config["eval"]["datasets"])
    ds_groups: dict[str, list] = defaultdict(list)
    for c in cases:
        ds_groups[c["dataset_name"]].append(c)
    limited_cases: list[dict] = []
    for ds_name in sorted(ds_groups):
        limited_cases.extend(ds_groups[ds_name][:n_cases])

    conditions = load_personas(
        config["eval"]["personas_path"],
        include_baseline=config["eval"].get("include_baseline", True),
    )
    jobs = build_jobs(limited_cases, conditions, config)
    return {
        "config": config,
        "cases": limited_cases,
        "jobs": jobs,
        "stats": {
            "n_cases": len(limited_cases),
            "n_conditions": len(conditions),
            "n_repetitions": config["eval"].get("n_repetitions", 3),
            "total_jobs": len(jobs),
            "per_dataset": dict(Counter(c["dataset_name"] for c in limited_cases)),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the persona fairness experiment locally.")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--output_dir_name", default=None, help="Folder under results/runs/ (auto if omitted)")
    parser.add_argument("--n_cases", type=int, default=None, help="Limit to N cases per dataset (for testing)")
    parser.add_argument("--max_workers", type=int, default=None, help="Thread pool size (default: config eval.max_workers or 4)")
    parser.add_argument("--metrics_only", action="store_true", help="Skip simulation; recompute metrics on an existing run")
    args = parser.parse_args()

    if args.metrics_only and not args.output_dir_name:
        parser.error("--metrics_only requires --output_dir_name")

    if not args.output_dir_name:
        args.output_dir_name = (
            "persona_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        )

    config = load_config(Path(args.config))
    local_dir = _PERSONAS_DIR / "results" / "runs" / args.output_dir_name
    local_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(local_dir / "run.log")

    # -- Metrics-only mode --------------------------------------------------
    if args.metrics_only:
        results = _load_local_results(local_dir)
        if not results:
            parser.error(f"No conversations found in {local_dir / 'conversations'}")
        metrics = compute_persona_metrics(results)
        atomic_write_json(metrics, local_dir / "metrics.json")
        print(f"Metrics saved to {local_dir / 'metrics.json'}")
        _print_summary(metrics)
        cost = compute_cost_from_results(results)
        atomic_write_json(cost, local_dir / "cost_summary.json")
        print_cost_summary(cost)
        return

    # -- Ensure the API key is available ------------------------------------
    get_openai_api_key()

    # -- Build dataset ------------------------------------------------------
    config_snapshot = json.loads(json.dumps(config))
    if config["doctor"].get("model") != "pha":
        from doctor.prompt import (
            INTERVIEW_SYSTEM_PROMPT,
        )
        config_snapshot["doctor"]["system_prompt"] = INTERVIEW_SYSTEM_PROMPT.format(
            max_turns=config["doctor"].get("max_turns", 5)
        )
    with open(local_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    n_cases = args.n_cases or config["eval"].get("max_cases")
    dataset = _build_dataset(config, n_cases)
    stats = dataset["stats"]
    print(f"output_dir_name: {args.output_dir_name}")
    print(f"local_dir:       {local_dir}")
    print(
        f"Dataset: {stats['n_cases']} cases x {stats['n_conditions']} personas x "
        f"{stats['n_repetitions']} reps = {stats['total_jobs']} jobs"
    )
    for ds, n in sorted(stats.get("per_dataset", {}).items()):
        print(f"  {ds}: {n} cases")

    # -- Run jobs locally ---------------------------------------------------
    doctor_config = config["doctor"]
    conv_dir = local_dir / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    jobs = dataset["jobs"]
    max_workers = args.max_workers or config["eval"].get("max_workers", 4)

    print(f"Running {len(jobs)} jobs with {max_workers} workers...")
    logger.info("=== Persona Experiment (local) === output_dir_name=%s jobs=%d", args.output_dir_name, len(jobs))

    def _process(job: dict) -> None:
        result = _run_job_with_retries(job, doctor_config)
        (conv_dir / f"{job['job_id']}.json").write_text(
            json.dumps(result, indent=2, default=str)
        )

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process, job): job for job in jobs}
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} jobs done")

    # -- Compute metrics + cost ---------------------------------------------
    results = _load_local_results(local_dir)
    metrics = compute_persona_metrics(results)
    atomic_write_json(metrics, local_dir / "metrics.json")
    cost = compute_cost_from_results(results)
    atomic_write_json(cost, local_dir / "cost_summary.json")

    print(f"\nDone. Results in {local_dir}")
    _print_summary(metrics)
    print_cost_summary(cost)
    logger.info("=== Done. Results in %s ===", local_dir)


if __name__ == "__main__":
    main()
