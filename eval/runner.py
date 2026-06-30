"""Eval runner — executes dataset jobs against the PatientSimulator.

Each job:
  1. Builds SimulatorConfig from job['all_params'] (task fixed to urgency_assessment)
  2. Initialises a fresh PatientSimulator (one MemorySaver per job — never shared)
  3. Creates a DoctorAgent and runs the conversation
  4. Writes {job_id}.json atomically to run_dir/conversations/
  5. Returns a result summary dict

run_eval() fans out over ThreadPoolExecutor; each thread gets its own
PatientSimulator and DoctorAgent instances (no shared state).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from src.config_wrapper import SimulatorConfig
from src.nodes import STOP_TOKEN
from src.patient_simulator import PatientSimulator
from doctor.doctor import TriageDoctorAgent
from eval._utils import (
    atomic_write_json,
    compute_cost_from_results,
    get_openai_api_key,
    print_cost_summary,
)

logger = logging.getLogger(__name__)

_MODEL_MAP: dict | None = None


def run_conversation(
    doctor: TriageDoctorAgent,
    patient_sim,
    max_turns: int,
) -> dict:
    """Drive a full conversation between doctor and patient simulator.

    Returns:
        {transcript, turn_count, patient_stopped, patient_node_outputs_per_turn}
    """
    doctor_msg = doctor.start()
    patient_stopped = False
    node_outputs_per_turn: list[dict] = []

    for _ in range(max_turns + 5):
        result = patient_sim.chat(doctor_msg)
        patient_msg: str = result["response"]
        patient_stopped = result.get("stopped", False)
        clean_patient_msg = patient_msg.replace(STOP_TOKEN, "").strip()
        node_outputs_per_turn.append(result.get("node_outputs", {}))

        if patient_stopped:
            doctor.record_patient_message(clean_patient_msg)
            break

        doctor_response, is_last = doctor.respond(clean_patient_msg)
        doctor_msg = doctor_response

        if is_last:
            break

    transcript = doctor.get_transcript()
    patient_turns = sum(1 for t in transcript if t["role"] == "patient")

    return {
        "transcript": transcript,
        "turn_count": patient_turns,
        "patient_stopped": patient_stopped,
        "patient_node_outputs_per_turn": node_outputs_per_turn,
        "doctor_token_usage": doctor.token_usage,
    }


# ---------------------------------------------------------------------------
# Single job
# ---------------------------------------------------------------------------

def run_single_job(
    job: dict,
    doctor_config: dict,
    run_dir: str,
    verbose: bool = False,
) -> dict:
    """Run one eval job end-to-end. Thread-safe (no shared mutable state).

    Returns a summary dict (also written to run_dir/conversations/{job_id}.json).
    """
    job_id = job["job_id"]
    t_start = time.perf_counter()
    error = None
    patient_profile = {}

    try:
        # 1. Build SimulatorConfig — only vignette (patient_summary), no narrative
        all_params = job["all_params"]
        config = SimulatorConfig(
            vignette=job["vignette_text"],
            task="urgency_assessment",
            setting="general",
            model_map=_MODEL_MAP,
            **all_params,
        )
        # validate() ensures OFAT discipline is respected
        config.validate()

        sim_kwargs = config.to_simulator_kwargs()
        sim_kwargs["verbose"] = verbose

        # 2. Fresh PatientSimulator instance (own MemorySaver, own thread_id)
        sim_kwargs["thread_id"] = f"eval_{job_id}"
        patient_sim = PatientSimulator(**sim_kwargs)
        patient_profile = patient_sim.patient_profile

        # 3. Doctor + conversation
        effective_doctor_config = doctor_config
        if job.get("probe_doctor_prompt"):
            effective_doctor_config = dict(doctor_config)
            effective_doctor_config["system_prompt"] = job["probe_doctor_prompt"]
            probe_max_turns = job.get("probe_doctor_max_turns")
            if probe_max_turns:
                effective_doctor_config["max_turns"] = probe_max_turns

        max_turns = effective_doctor_config.get("max_turns", 7)
        doctor = TriageDoctorAgent(effective_doctor_config)
        conv_result = run_conversation(doctor, patient_sim, max_turns=max_turns)

    except Exception as exc:
        logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
        error = str(exc)
        conv_result = {
            "transcript": [],
            "turn_count": 0,
            "patient_stopped": False,
            "final_assessment": None,
            "reasoning": None,
            "patient_node_outputs_per_turn": [],
            "doctor_token_usage": [],
        }

    duration = round(time.perf_counter() - t_start, 2)

    result = {
        "job_id": job_id,
        "vignette_id": job["vignette_id"],
        "case_id": job.get("case_id"),
        "vignette_text": job.get("vignette_text"),
        "job_type": job["job_type"],
        "param_group": job.get("param_group"),
        "param_name": job.get("param_name"),
        "param_label": job.get("param_label"),
        "param_description": job.get("param_description"),
        "param_value": job.get("param_value"),
        "param_value_label": job.get("param_value_label"),
        "pair_key": job.get("pair_key"),
        "verbosity_level": job.get("verbosity_level"),
        "max_turns_level": job.get("max_turns_level"),
        "probe_type": job.get("probe_type"),
        "active_value": job.get("active_value"),
        "neutral_value": job.get("neutral_value"),
        "active_option": job.get("active_option"),
        "neutral_option": job.get("neutral_option"),
        "neutral_job_id": job.get("neutral_job_id"),
        "patient_profile": patient_profile,
        "symptom_category": getattr(patient_sim, "symptom_category", "") if not error else "",
        "input_quality": getattr(patient_sim, "input_quality", {}) if not error else {},
        "realism_check": getattr(patient_sim, "realism_check", {}) if not error else {},
        "transcript": conv_result["transcript"],
        "turn_count": conv_result["turn_count"],
        "patient_stopped": conv_result["patient_stopped"],
        "init_node_outputs": patient_sim.init_node_outputs if not error else {},
        "patient_node_outputs_per_turn": conv_result["patient_node_outputs_per_turn"],
        "doctor_token_usage": conv_result["doctor_token_usage"],
        "error": error,
        "duration_seconds": duration,
    }

    # 4. Atomic write to conversations/
    conv_dir = Path(run_dir) / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    out_path = conv_dir / f"{job_id}.json"
    atomic_write_json(result, out_path)

    logger.info(
        "Job %s done | type=%s param=%s val=%s turns=%d duration=%.1fs error=%s",
        job_id[:8],
        job["job_type"],
        job.get("param_name", "-"),
        job.get("param_value", "-"),
        conv_result["turn_count"],
        duration,
        error is not None,
    )
    return result


# ---------------------------------------------------------------------------
# Full eval run
# ---------------------------------------------------------------------------

def run_eval(
    dataset: dict,
    doctor_config: dict,
    eval_config: dict,
    run_id: str | None = None,
    max_workers: int = 4,
    progress_callback=None,
    use_tqdm: bool = True,
    resume: bool = False,
) -> dict:
    """Execute all jobs in the dataset using a thread pool.

    Each thread runs run_single_job() independently with its own PatientSimulator.

    Args:
        dataset:           Output of dataset.build_dataset()
        doctor_config:     Parsed doctor_agent_config.json
        eval_config:       Parsed eval_config.json
        run_id:            Optional run identifier (UUID4 if not provided)
        max_workers:       Thread pool size
        progress_callback: Optional callable(completed, total, message) for live progress
        resume:            If True, skip jobs whose conversation file already exists

    Returns:
        run meta dict {run_id, run_dir, started_at, completed_at,
                       total_jobs, completed_jobs, failed_jobs}
    """
    # Ensure the API key is set before spawning threads
    get_openai_api_key()

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

    results_base = eval_config.get("results_path", "eval/results/runs")
    run_dir = str(Path(results_base) / run_id)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    jobs = dataset["jobs"]
    started_at = datetime.now(timezone.utc).isoformat()

    # Filter out already-completed jobs when resuming
    skipped = 0
    if resume:
        conv_dir = Path(run_dir) / "conversations"
        pending = []
        for job in jobs:
            if (conv_dir / f"{job['job_id']}.json").exists():
                skipped += 1
            else:
                pending.append(job)
        logger.info("Resume: %d/%d jobs already done, %d remaining", skipped, len(jobs), len(pending))
        jobs = pending

    # Save run meta and dataset snapshot
    run_meta = {
        "run_id": run_id,
        "run_dir": run_dir,
        "started_at": started_at,
        "completed_at": None,
        "total_jobs": len(dataset["jobs"]),
        "completed_jobs": skipped,
        "failed_jobs": 0,
        "eval_config": eval_config,
    }
    if not resume:
        atomic_write_json(run_meta, Path(run_dir) / "run_meta.json")
        atomic_write_json(dataset, Path(run_dir) / "dataset.json")

    logger.info("run_eval started | run_id=%s total_jobs=%d pending=%d workers=%d", run_id, len(dataset["jobs"]), len(jobs), max_workers)

    completed = skipped
    failed = 0

    if use_tqdm:
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None
    else:
        _tqdm = None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_job, job, doctor_config, run_dir): job
            for job in jobs
        }
        pbar = _tqdm(total=len(jobs), unit="job") if _tqdm else None
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                if result.get("error"):
                    failed += 1
                else:
                    completed += 1
            except Exception as exc:
                logger.error("Unexpected future error for job %s: %s", job["job_id"][:8], exc)
                failed += 1
                completed += 1  # count as processed

            total_done = completed + failed
            desc = (
                f"param={job.get('param_name', 'baseline')} "
                f"val={job.get('param_value', '-')}"
            )
            if pbar:
                pbar.set_description(desc)
                pbar.update(1)
            if progress_callback:
                msg = f"[{total_done}/{len(jobs)}] {desc} vignette={job.get('vignette_id')}"
                progress_callback(total_done, len(jobs), msg)
        if pbar:
            pbar.close()

    completed_at = datetime.now(timezone.utc).isoformat()

    all_results = load_run_results(run_dir)
    cost = compute_cost_from_results(all_results)
    print_cost_summary(cost)
    atomic_write_json(cost, Path(run_dir) / "cost_summary.json")

    run_meta.update(
        {
            "completed_at": completed_at,
            "completed_jobs": completed,
            "failed_jobs": failed,
            "cost_usd": cost["total"]["cost_usd"],
            "cache_hit_rate_pct": cost["cache_hit_rate_pct"],
        }
    )
    atomic_write_json(run_meta, Path(run_dir) / "run_meta.json")

    logger.info(
        "run_eval done | run_id=%s completed=%d failed=%d cost=$%.2f cache=%.1f%%",
        run_id, completed, failed, cost["total"]["cost_usd"], cost["cache_hit_rate_pct"],
    )
    return run_meta


# ---------------------------------------------------------------------------
# Run management helpers
# ---------------------------------------------------------------------------

def list_runs(results_base: str) -> list[dict]:
    """Return a list of run meta dicts sorted by started_at descending."""
    base = Path(results_base)
    runs = []
    if not base.exists():
        return runs
    for run_dir in sorted(base.iterdir(), reverse=True):
        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    runs.append(json.load(f))
            except Exception:
                pass
    return runs


def load_run_results(run_dir: str) -> list[dict]:
    """Load all {job_id}.json conversation results from a run directory."""
    conv_dir = Path(run_dir) / "conversations"
    results = []
    if not conv_dir.exists():
        return results
    for path in conv_dir.glob("*.json"):
        if path.stem.endswith("_judged"):
            continue
        try:
            with open(path) as f:
                results.append(json.load(f))
        except Exception:
            pass
    return results
