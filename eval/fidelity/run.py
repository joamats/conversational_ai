"""Clinical fidelity eval pipeline — single entry point.

Evaluates whether the patient simulator accurately reproduces clinical facts
from the input vignette. Three sequential steps:
  1. Simulate    — generate conversations (or load from an existing run)
  2. Evaluate    — two-phase LLM pipeline (extract claims → classify claims)
  3. Report      — aggregate metrics

All output lands in results/runs/{run_id}/:
  run_meta.json       — run config and summary stats
  dataset.json        — job manifest
  run.log             — full log for this run
  conversations/      — one JSON per conversation (simulation results)
  fidelity/           — one JSON per conversation (extraction + classification)

Usage:
    python -m eval.fidelity.run
    python -m eval.fidelity.run --config path/to/config.json
    python -m eval.fidelity.run --source-run-id <id>
    python -m eval.fidelity.run --evaluate-only --run-id <id>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from eval._utils import (
    atomic_write_json,
    compute_cost_from_results,
    get_openai_api_key,
    load_config,
    print_cost_summary,
    setup_logging,
)

_FIDELITY_DIR = Path(__file__).parent
_EVAL_DIR = _FIDELITY_DIR.parent
_REPO_ROOT = Path(__file__).parents[2]

logger = logging.getLogger(__name__)


def _resolve_data_path(config: dict) -> dict:
    cfg = dict(config)
    p = Path(cfg["data_path"])
    if not p.is_absolute():
        cfg["data_path"] = str(_REPO_ROOT / p)
    return cfg


# ---------------------------------------------------------------------------
# Dataset — simple fidelity jobs (one per vignette, all defaults)
# ---------------------------------------------------------------------------

def _build_fidelity_dataset(config: dict) -> dict:
    from eval.adherence.dataset import (
        load_vignettes,
        _get_behavioral_defaults,
        _NEUTRAL_PATIENT_GOAL,
    )

    split = config.get("split")
    splits_path = config.get("splits_path")
    if split and splits_path:
        from eval.splits import load_split_vignettes
        vignettes = load_split_vignettes(
            tsv_path=config["data_path"],
            split=split,
            splits_path=splits_path,
        )
    else:
        vignettes = load_vignettes(
            tsv_path=config["data_path"],
            n=config["n_vignettes"],
            seed=config["seed"],
        )

    jobs = []
    for v in vignettes:
        all_params = _get_behavioral_defaults()
        all_params["patient_goal"] = _NEUTRAL_PATIENT_GOAL
        jobs.append({
            "job_id": str(uuid.uuid4()),
            "vignette_id": v["vignette_id"],
            "case_id": v["case_id"],
            "job_type": "fidelity",
            "all_params": all_params,
            "vignette_text": v["vignette_text"],
        })

    return {
        "vignettes": vignettes,
        "jobs": jobs,
        "stats": {
            "total_jobs": len(jobs),
            "vignettes_count": len(vignettes),
        },
    }


# ---------------------------------------------------------------------------
# Step 1: Simulate conversations
# ---------------------------------------------------------------------------

def step_simulate(dataset: dict, doctor_config: dict, config: dict, run_id: str) -> dict:
    from eval.runner import run_eval

    max_workers = config.get("max_workers", 4)
    total = len(dataset["jobs"])
    logger.info("Simulating %d fidelity conversations (max_workers=%d)...", total, max_workers)
    print(f"Simulating {total} conversations (max_workers={max_workers})...")

    pbar = tqdm(total=total, desc="Simulate", unit="job", file=sys.stdout)

    def callback(completed, total, msg):
        pbar.set_postfix_str(msg.split("] ", 1)[-1][:60], refresh=False)
        pbar.update(1)

    run_meta = run_eval(
        dataset=dataset,
        doctor_config=doctor_config,
        eval_config=config,
        run_id=run_id,
        max_workers=max_workers,
        progress_callback=callback,
        use_tqdm=False,
    )
    pbar.close()
    logger.info("Simulation done: completed=%d failed=%d", run_meta["completed_jobs"], run_meta["failed_jobs"])
    return run_meta


# ---------------------------------------------------------------------------
# Step 1b: Load conversations from an existing run
# ---------------------------------------------------------------------------

def _load_source_conversations(source_run_dir: str, n_vignettes: int) -> list[dict]:
    """Load conversations from a prior run to evaluate without re-simulating."""
    from eval.runner import load_run_results

    all_results = load_run_results(source_run_dir)
    neutrals = [r for r in all_results if r.get("job_type") == "neutral" and r.get("transcript")]
    if not neutrals:
        neutrals = [r for r in all_results if r.get("transcript")]

    neutrals.sort(key=lambda r: r.get("vignette_id", 0))
    return neutrals[:n_vignettes]


# ---------------------------------------------------------------------------
# Step 2: Evaluate (extract + classify claims)
# ---------------------------------------------------------------------------

def step_evaluate(
    run_dir: str,
    judge_config: dict,
    max_workers: int = 4,
    source_conversations: list[dict] | None = None,
) -> list[dict]:
    from eval.fidelity.extractor import evaluate_conversation
    from eval.runner import load_run_results

    fidelity_dir = Path(run_dir) / "fidelity"
    fidelity_dir.mkdir(parents=True, exist_ok=True)

    if source_conversations:
        conversations = source_conversations
    else:
        conversations = load_run_results(run_dir)
        conversations = [c for c in conversations if c.get("transcript")]

    if not conversations:
        logger.warning("No conversations to evaluate.")
        return []

    already_done = {p.stem for p in fidelity_dir.glob("*.json")}
    to_evaluate = [c for c in conversations if c.get("job_id") not in already_done]

    if not to_evaluate:
        logger.info("All conversations already evaluated.")
        results = []
        for p in fidelity_dir.glob("*.json"):
            with open(p) as f:
                results.append(json.load(f))
        return results

    logger.info("Evaluating %d conversations for clinical fidelity (max_workers=%d)...", len(to_evaluate), max_workers)
    print(f"Evaluating {len(to_evaluate)} conversations for clinical fidelity...")

    pbar = tqdm(total=len(to_evaluate), desc="Fidelity eval", unit="conv", file=sys.stdout)
    results = []

    # Also load any previously completed results
    for p in fidelity_dir.glob("*.json"):
        try:
            with open(p) as f:
                results.append(json.load(f))
        except Exception:
            pass

    def _evaluate_one(conv: dict) -> dict:
        vid = conv.get("vignette_id", -1)
        job_id = conv.get("job_id", "unknown")

        # Need the vignette text — check conversation metadata or dataset
        vignette_text = conv.get("vignette_text", "")
        if not vignette_text:
            dataset_path = Path(run_dir) / "dataset.json"
            if dataset_path.exists():
                with open(dataset_path) as f:
                    dataset = json.load(f)
                for job in dataset.get("jobs", []):
                    if job.get("job_id") == job_id:
                        vignette_text = job.get("vignette_text", "")
                        break
                if not vignette_text:
                    for v in dataset.get("vignettes", []):
                        if v.get("vignette_id") == vid:
                            vignette_text = v.get("vignette_text", "")
                            break

        if not vignette_text:
            logger.error("No vignette text for job %s (vignette %d) — skipping.", job_id[:8], vid)
            return {}

        fidelity_result = evaluate_conversation(
            vignette=vignette_text,
            transcript=conv["transcript"],
            judge_config=judge_config,
        )

        result = {
            "job_id": job_id,
            "vignette_id": vid,
            "case_id": conv.get("case_id"),
            "turn_count": conv.get("turn_count"),
            **fidelity_result,
        }

        atomic_write_json(result, fidelity_dir / f"{job_id}.json")
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_evaluate_one, c): c for c in to_evaluate}
        for future in as_completed(futures):
            conv = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                logger.error("Fidelity eval failed for job %s: %s", conv.get("job_id", "?")[:8], exc)
            pbar.update(1)
    pbar.close()

    return results


# ---------------------------------------------------------------------------
# Step 3: Report metrics
# ---------------------------------------------------------------------------

def step_report(results: list[dict], run_dir: str) -> dict:
    from eval.fidelity.metrics import (
        compute_fidelity_metrics,
        fidelity_metrics_to_dataframes,
    )

    metrics = compute_fidelity_metrics(results)
    atomic_write_json(metrics, Path(run_dir) / "fidelity" / "metrics.json")

    overall = metrics["overall"]
    print(f"\nClinical Fidelity Eval ({overall.get('total_conversations', 0)} conversations)")
    print(f"  Total patient claims:       {overall['total_claims']}")
    print(f"  Supported:                  {overall['n_supported']}")
    print(f"  Unsupported:                {overall['n_unsupported']}")
    print(f"  Mean accuracy (sup, 1-4):   {overall['mean_accuracy']:.2f}  — avg correctness of claims matching vignette facts")
    print(f"  Mean plausibility (unsup):  {overall['mean_plausibility']:.2f}  — avg clinical plausibility of fabricated claims")
    print(f"  Coverage rate:              {overall['coverage_rate']:.1%}")
    print(f"  Unsupported rate:           {overall['unsupported_rate']:.1%}")
    print(f"  Confirmed rate (score=4):   {overall['confirmed_rate']:.1%}")
    print(f"  Contradiction rate (=1):    {overall['contradiction_rate']:.1%}")

    dfs = fidelity_metrics_to_dataframes(metrics)
    print("\nScore distribution:")
    print(dfs["distribution_df"].to_string(index=False))
    print("\nPer-vignette breakdown:")
    print(dfs["vignette_df"].to_string(index=False))

    logger.info(
        "Fidelity metrics: coverage=%.1f%% unsupported=%.1f%% confirmed=%.1f%% "
        "contradiction=%.1f%% accuracy=%.2f plausibility=%.2f",
        overall["coverage_rate"] * 100,
        overall["unsupported_rate"] * 100,
        overall["confirmed_rate"] * 100,
        overall["contradiction_rate"] * 100,
        overall["mean_accuracy"],
        overall["mean_plausibility"],
    )

    cost = compute_cost_from_results(results)
    print_cost_summary(cost)
    atomic_write_json(cost, Path(run_dir) / "fidelity" / "cost_summary.json")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clinical fidelity eval pipeline.")
    parser.add_argument(
        "--config", default=str(_FIDELITY_DIR / "config.json"),
        help="Path to fidelity eval config JSON",
    )
    parser.add_argument(
        "--source-run-id", default=None,
        help="Reuse conversations from an existing parameter-adherence run instead of generating new ones",
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Skip simulation, only run evaluation on existing conversations",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Existing fidelity run ID (for --evaluate-only)",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    config = _resolve_data_path(config)
    doctor_config = config.get("doctor", {})
    judge_config = config.get("judge", {})
    max_workers = config.get("max_workers", 4)

    results_base = Path(config.get("results_path", str(_FIDELITY_DIR / "results" / "runs")))

    # ── Evaluate-only mode ────────────────────────────────────────────────
    if args.evaluate_only:
        if not args.run_id:
            parser.error("--evaluate-only requires --run-id")
        run_dir = str(results_base / args.run_id)
        setup_logging(Path(run_dir) / "run.log")

        results = step_evaluate(run_dir, judge_config, max_workers)
        if results:
            step_report(results, run_dir)
        return

    # ── Full run ──────────────────────────────────────────────────────────
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    run_dir = results_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    get_openai_api_key()

    print(f"run_id: {run_id}")
    print(f"logs:   {run_dir / 'run.log'}")
    logger.info("=== Clinical Fidelity Eval ===")
    logger.info("run_id=%s", run_id)

    source_conversations = None

    if args.source_run_id:
        # Load conversations from a prior parameter-adherence run
        source_run_path = config.get("source_results_path")
        if not source_run_path:
            source_run_path = str(_EVAL_DIR / "adherence" / "results" / "runs")
        source_run_dir = str(Path(source_run_path) / args.source_run_id)
        logger.info("Loading conversations from source run: %s", source_run_dir)
        print(f"Loading conversations from: {source_run_dir}")
        source_conversations = _load_source_conversations(source_run_dir, config["n_vignettes"])
        logger.info("Loaded %d source conversations.", len(source_conversations))

        # Save the vignette info so dataset.json is meaningful
        dataset = {
            "source_run_id": args.source_run_id,
            "jobs": [
                {
                    "job_id": c.get("job_id"),
                    "vignette_id": c.get("vignette_id"),
                    "case_id": c.get("case_id"),
                    "job_type": "fidelity",
                }
                for c in source_conversations
            ],
            "stats": {"total_jobs": len(source_conversations), "vignettes_count": len(source_conversations)},
        }
        atomic_write_json(dataset, run_dir / "dataset.json")
    else:
        # Generate fresh conversations
        dataset = _build_fidelity_dataset(config)
        atomic_write_json(dataset, run_dir / "dataset.json")
        logger.info("Built dataset: %d vignettes, %d jobs", dataset["stats"]["vignettes_count"], dataset["stats"]["total_jobs"])
        step_simulate(dataset, doctor_config, config, run_id)

    # Evaluate
    results = step_evaluate(str(run_dir), judge_config, max_workers, source_conversations)

    # Report
    if results:
        step_report(results, str(run_dir))

    print(f"\nDone. Results in {run_dir}")
    logger.info("=== Done. Results in %s ===", run_dir)


if __name__ == "__main__":
    main()
