"""Parameter adherence eval pipeline — single entry point.

Runs three sequential steps:
  1. Build dataset  (sample vignettes, generate OFAT jobs)
  2. Simulate       (run conversations via ThreadPoolExecutor + tqdm)
  3. Judge          (blind LLM judge per param_variation job + tqdm)

Usage:
    python -m eval.adherence.run
    python -m eval.adherence.run --simulate-only
    python -m eval.adherence.run --judge-only --run-id <id>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
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
    split_config,
)

_ADHERENCE_DIR = Path(__file__).parent
_REPO_ROOT = Path(__file__).parents[2]

logger = logging.getLogger(__name__)


def _resolve_data_path(eval_config: dict) -> dict:
    cfg = dict(eval_config)
    p = Path(cfg["data_path"])
    if not p.is_absolute():
        cfg["data_path"] = str(_REPO_ROOT / p)
    return cfg


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_build(eval_config: dict, schema: dict) -> dict:
    from eval.adherence.dataset import build_dataset, save_dataset

    logger.info("Building dataset: n_vignettes=%d seed=%d", eval_config["n_vignettes"], eval_config["seed"])
    dataset = build_dataset(eval_config, schema)
    stats = dataset["stats"]
    logger.info(
        "Dataset built: %d vignettes, %d dimensions, %d jobs (%d neutral + %d active + %d probing)",
        stats["vignettes_count"], stats["dimensions_count"], stats["total_jobs"],
        stats["neutral_jobs"], stats["param_variation_jobs"], stats.get("probing_jobs", 0),
    )
    return dataset


def step_simulate(dataset: dict, doctor_config: dict, eval_config: dict, run_id: str, resume: bool = False) -> dict:
    from eval.runner import run_eval

    max_workers = eval_config.get("max_workers", 4)
    total = dataset["stats"]["total_jobs"]
    logger.info("Simulating %d conversations (max_workers=%d)...", total, max_workers)

    # When resuming, count existing results so the progress bar starts partway
    n_done = 0
    if resume:
        results_base = eval_config.get("results_path", "eval/results/runs")
        conv_dir = Path(results_base) / run_id / "conversations"
        if conv_dir.exists():
            n_done = sum(1 for p in conv_dir.glob("*.json") if not p.stem.endswith("_comparative"))
        print(f"Resuming: {n_done}/{total} already done, {total - n_done} remaining (max_workers={max_workers})...")
    else:
        print(f"Simulating {total} conversations (max_workers={max_workers})...")

    pbar = tqdm(total=total, initial=n_done, desc="Simulate", unit="job", file=sys.stdout)

    def callback(completed, total, msg):
        pbar.set_postfix_str(msg.split("] ", 1)[-1][:60], refresh=False)
        pbar.update(1)

    run_meta = run_eval(
        dataset=dataset,
        doctor_config=doctor_config,
        eval_config=eval_config,
        run_id=run_id,
        max_workers=max_workers,
        progress_callback=callback,
        use_tqdm=False,  # we manage tqdm here
        resume=resume,
    )
    pbar.close()

    logger.info(
        "Simulation done: completed=%d failed=%d run_dir=%s",
        run_meta["completed_jobs"], run_meta["failed_jobs"], run_meta["run_dir"],
    )
    return run_meta


def step_judge_comparative(
    run_dir: str,
    judge_config: dict,
    schema: dict,
    max_workers: int,
    seed: int = 42,
) -> list[dict]:
    from eval.adherence.judge import judge_run_comparative
    from eval.adherence.metrics import compute_comparative_metrics

    conv_dir = Path(run_dir) / "conversations"
    # Count eligible pairs (active + neutral for same vignette/param, not yet judged)
    n_existing = sum(1 for p in conv_dir.glob("*_comparative.json"))
    logger.info("Starting comparative judging (max_workers=%d, seed=%d)...", max_workers, seed)
    print(f"Comparative judging (max_workers={max_workers})...")

    pbar = tqdm(desc="Comparative Judge", unit="pair", file=sys.stdout)

    def callback(done, total, msg):
        pbar.total = total
        pbar.set_postfix_str(msg[:60], refresh=True)
        pbar.update(1)

    results = judge_run_comparative(
        run_dir=run_dir,
        judge_config=judge_config,
        schema=schema,
        max_workers=max_workers,
        progress_callback=callback,
        seed=seed,
    )
    pbar.close()

    correct = sum(1 for r in results if r.get("correct"))
    logger.info(
        "Comparative judging done: %d pairs, %d correct (%.1f%%)",
        len(results), correct, 100 * correct / len(results) if results else 0,
    )

    from eval.adherence.judge import load_comparative_results

    all_comparative = load_comparative_results(run_dir)
    if not all_comparative:
        return results

    cost = compute_cost_from_results(all_comparative)
    print_cost_summary(cost)
    atomic_write_json(cost, Path(run_dir) / "judge_cost_summary.json")

    metrics = compute_comparative_metrics(all_comparative, schema)
    macro_auc = metrics["macro_auc"]
    total_pairs = metrics["total_pairs"]
    total_correct = metrics["total_correct"]
    print(f"\nComparative eval ({total_pairs} pairs, macro-AUC: {macro_auc:.4f}):")
    logger.info("Comparative macro-AUC=%.4f", macro_auc)

    params_by_group: dict[str, list] = {}
    for pname, pdata in metrics["per_param"].items():
        gid = pdata.get("group_id", "other")
        params_by_group.setdefault(gid, []).append((pname, pdata))

    for gid, data in sorted(metrics.get("per_group", {}).items(), key=lambda x: -x[1]["auc"]):
        print(f"\n  {data['label']}  (AUC={data['auc']:.3f}, {data['n_pairs']} pairs)")
        logger.info("  [%s] AUC=%.3f  (%d pairs)", data["label"], data["auc"], data["n_pairs"])
        for pname, pdata in sorted(params_by_group.get(gid, []), key=lambda x: -x[1]["auc"]):
            status = "PASS" if pdata["auc"] >= 0.6 else "FAIL"
            print(f"    {status}  {pname:30s}  AUC={pdata['auc']:.3f}  ({pdata['n_correct']}/{pdata['n_pairs']})")
            logger.info("    %-30s AUC=%.3f  (%d/%d)", pname, pdata["auc"], pdata["n_correct"], pdata["n_pairs"])

    return results


def step_max_turns_metrics(run_dir: str) -> dict:
    from eval.runner import load_run_results
    from eval.adherence.metrics import (
        compute_max_turns_metrics, max_turns_metrics_to_dataframes,
    )

    results = load_run_results(run_dir)
    metrics = compute_max_turns_metrics(results)

    if metrics["total_jobs"] == 0:
        return metrics

    print(f"\nMax turns eval ({metrics['total_jobs']} jobs):")
    print(f"  Ordering holds: {metrics['ordering_holds']}")
    print(f"  {metrics['ordering_detail']}")
    df = max_turns_metrics_to_dataframes(metrics)["level_df"]
    print(df.to_string(index=False))
    logger.info("Max turns ordering_holds=%s  %s", metrics["ordering_holds"], metrics["ordering_detail"])

    return metrics


def step_probing_metrics(run_dir: str) -> dict:
    from eval.runner import load_run_results
    from eval.adherence.metrics import (
        compute_probing_metrics, probing_metrics_to_dataframes,
    )

    results = load_run_results(run_dir)
    metrics = compute_probing_metrics(results)

    if metrics["total_jobs"] == 0:
        return metrics

    print(f"\nProbing eval ({metrics['total_jobs']} jobs, overall compliance: {metrics['overall_compliance']:.1%}):")
    df = probing_metrics_to_dataframes(metrics)["probe_df"]
    print(df.to_string(index=False))

    for probe_type, pdata in sorted(metrics["per_probe"].items()):
        logger.info(
            "  %-25s compliance=%.1f%%  (%d/%d)",
            probe_type, pdata["compliance_rate"] * 100,
            pdata["n_compliant"], pdata["n_jobs"],
        )

    return metrics


def step_verbosity_metrics(run_dir: str) -> dict:
    from eval.runner import load_run_results
    from eval.adherence.metrics import (
        compute_verbosity_metrics, verbosity_metrics_to_dataframes,
    )

    results = load_run_results(run_dir)
    metrics = compute_verbosity_metrics(results)

    if metrics["total_jobs"] == 0:
        return metrics

    print(f"\nVerbosity eval ({metrics['total_jobs']} jobs):")
    print(f"  Ordering holds: {metrics['ordering_holds']}")
    print(f"  {metrics['ordering_detail']}")
    df = verbosity_metrics_to_dataframes(metrics)["level_df"]
    print(df.to_string(index=False))
    logger.info("Verbosity ordering_holds=%s  %s", metrics["ordering_holds"], metrics["ordering_detail"])

    return metrics


def step_combined_cost(run_dir: str) -> dict | None:
    """Load simulation + judge cost summaries and print a combined total."""
    rd = Path(run_dir)
    sim_path = rd / "cost_summary.json"
    judge_path = rd / "judge_cost_summary.json"

    all_rows: list[dict] = []
    for path in (sim_path, judge_path):
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            all_rows.extend(data.get("per_node", []))

    if not all_rows:
        return None

    grand = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
             "reasoning_tokens": 0, "calls": 0, "cost_usd": 0.0}
    for row in all_rows:
        for k in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "calls"):
            grand[k] += row.get(k, 0)
        grand["cost_usd"] += row.get("cost_usd", 0.0)
    grand["cost_usd"] = round(grand["cost_usd"], 4)

    cache_rate = (
        grand["cached_tokens"] / grand["input_tokens"] * 100
        if grand["input_tokens"] > 0 else 0.0
    )

    combined = {
        "per_node": all_rows,
        "total": grand,
        "cache_hit_rate_pct": round(cache_rate, 1),
    }
    atomic_write_json(combined, rd / "combined_cost_summary.json")
    print_cost_summary(combined)
    logger.info("Combined cost: $%.2f (cache %.1f%%)", grand["cost_usd"], cache_rate)
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the patient simulator eval pipeline.")
    parser.add_argument(
        "--config", default=str(_ADHERENCE_DIR / "config.json"),
        help="Path to merged config JSON (default: eval/config.json)",
    )
    parser.add_argument(
        "--simulate-only", action="store_true",
        help="Only run the simulation step (skip judging)",
    )
    parser.add_argument(
        "--judge-only", action="store_true",
        help="Only run the judge step on an existing run",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Existing run ID (required with --judge-only or --resume)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted run, skipping already-completed jobs (requires --run-id)",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    eval_config, doctor_config, judge_config = split_config(config)
    eval_config = _resolve_data_path(eval_config)

    results_base = Path(eval_config["results_path"])

    # ── Judge-only mode ───────────────────────────────────────────────────
    if args.judge_only:
        if not args.run_id:
            parser.error("--judge-only requires --run-id")
        run_dir = str(results_base / args.run_id)
        setup_logging(Path(run_dir) / "run.log")

        from src.param_prompts import SCHEMA
        max_workers = eval_config.get("max_workers", 4)
        step_judge_comparative(run_dir, judge_config, SCHEMA, max_workers,
                               seed=eval_config.get("seed", 42))
        step_verbosity_metrics(run_dir)
        step_max_turns_metrics(run_dir)
        step_probing_metrics(run_dir)
        step_combined_cost(run_dir)
        return

    # ── Resume mode ────────────────────────────────────────────────────────
    if args.resume:
        if not args.run_id:
            parser.error("--resume requires --run-id")
        run_id = args.run_id
        run_dir = results_base / run_id
        if not run_dir.exists():
            parser.error(f"Run directory not found: {run_dir}")
        setup_logging(run_dir / "run.log")

        get_openai_api_key()

        dataset_path = run_dir / "dataset.json"
        with open(dataset_path) as f:
            dataset = json.load(f)

        print(f"Resuming run_id: {run_id}")
        print(f"logs:   {run_dir / 'run.log'}")
        logger.info("=== Resuming Patient Simulator Eval ===")
        logger.info("run_id=%s", run_id)

        run_meta = step_simulate(dataset, doctor_config, eval_config, run_id, resume=True)

        if not args.simulate_only:
            from src.param_prompts import SCHEMA
            max_workers = eval_config.get("max_workers", 4)
            step_judge_comparative(str(run_dir), judge_config, SCHEMA, max_workers,
                                   seed=eval_config.get("seed", 42))
            step_verbosity_metrics(str(run_dir))
            step_max_turns_metrics(str(run_dir))
            step_probing_metrics(str(run_dir))

        step_combined_cost(str(run_dir))
        print(f"Done. Results in {run_dir}")
        logger.info("=== Done. Results in %s ===", run_dir)
        return

    # ── Full or simulate-only run ─────────────────────────────────────────
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    run_dir = results_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")

    # Save the config used for this run
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    from src.param_prompts import SCHEMA

    get_openai_api_key()  # fail fast before any work starts

    print(f"run_id: {run_id}")
    print(f"logs:   {run_dir / 'run.log'}")
    logger.info("=== Patient Simulator Eval ===")
    logger.info("run_id=%s", run_id)
    logger.info("config=%s", args.config)

    dataset = step_build(eval_config, SCHEMA)
    run_meta = step_simulate(dataset, doctor_config, eval_config, run_id)

    if not args.simulate_only:
        max_workers = eval_config.get("max_workers", 4)
        step_judge_comparative(str(run_dir), judge_config, SCHEMA, max_workers,
                               seed=eval_config.get("seed", 42))
        step_verbosity_metrics(str(run_dir))
        step_max_turns_metrics(str(run_dir))
        step_probing_metrics(str(run_dir))

    step_combined_cost(str(run_dir))
    print(f"Done. Results in {run_dir}")
    logger.info("=== Done. Results in %s ===", run_dir)


if __name__ == "__main__":
    main()
