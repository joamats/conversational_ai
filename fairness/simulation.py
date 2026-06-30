"""Single-case runner for the persona compound fairness experiment.

Orchestrates a conversation between the multi-node PatientSimulator and
the 3-level AtomicTriageDoctorAgent, then captures the prediction.
"""
from __future__ import annotations

import logging
import os
import time
import uuid

from collections import Counter

from doctor.doctor import (
    DEFAULT_URGENCY_LEVELS,
    TriageDoctorAgent,
)
from fairness.dataset import (
    build_triage_ord,
)
from src.nodes import (
    DEFAULT_MODEL_MAP,
    STOP_TOKEN,
)
from src.patient_simulator import (
    PatientSimulator,
)

logger = logging.getLogger(__name__)


_SIM_PARAM_KEYS = frozenset({
    "anxiety", "frustration", "patience", "trust", "embarrassment",
    "recall_level", "oversharing_tendency",
    "verbosity", "health_literacy", "english_proficiency",
    "communication_style", "profanity_level",
    "typo_rate", "punctuation_style", "capitalisation",
    "patient_goal",
})


def run_persona_conversation(job: dict, doctor_config: dict) -> dict:
    """Execute one doctor-patient conversation for a persona experiment job.

    Args:
        job: Job dict from ``build_jobs()`` with sim_params, vignette, etc.
        doctor_config: Dict with model, reasoning_effort, max_turns keys.

    Returns:
        Result dict with prediction, transcript, and metadata.
    """
    job_urgency_levels = job.get("urgency_levels") or doctor_config.get("urgency_levels", DEFAULT_URGENCY_LEVELS)
    triage_ord = build_triage_ord(job_urgency_levels)

    job_doctor_config = {**doctor_config, "urgency_levels": job_urgency_levels}

    t_start = time.perf_counter()
    error: str | None = None
    transcript: list[dict] = []
    assessments: list[dict] = []
    node_outputs_per_turn: list[dict] = []
    init_node_outputs: dict = {}
    doctor_token_usage: list[dict] = []

    try:
        sim_params = job["sim_params"]
        model_map = job.get("model_map", DEFAULT_MODEL_MAP)

        sim_kwargs: dict = {
            "vignette": job["patient_vignette"],
            "ehr_json": job.get("ehr_json"),
            "prior_conversation": job.get("prior_conversation"),
            "task": "urgency_assessment",
            "setting": "verily_me_app",
            "gives_up": False,
            "stops_when_goal_met": False,
            "enable_stop_check": False,
            "model_map": model_map,
        }
        for key in _SIM_PARAM_KEYS:
            if key in sim_params:
                sim_kwargs[key] = sim_params[key]

        patient = PatientSimulator(**sim_kwargs)
        init_node_outputs = patient.init_node_outputs
        doctor = TriageDoctorAgent(job_doctor_config)

        doctor_msg = doctor.start()
        transcript.append({"role": "doctor", "content": doctor_msg})

        is_final = False
        while True:
            result = patient.chat(doctor_msg)
            patient_msg = result["response"].replace(STOP_TOKEN, "").strip()
            transcript.append({"role": "patient", "content": patient_msg})
            node_outputs_per_turn.append(result.get("node_outputs", {}))

            if result.get("stopped", False) or is_final:
                doctor.record_patient_message(patient_msg)
                break

            doctor_msg, is_final = doctor.respond(patient_msg)
            if is_final and not doctor_msg:
                break
            transcript.append({"role": "doctor", "content": doctor_msg})

        assessments = doctor.assess_multi()
        doctor_token_usage = doctor.token_usage

    except Exception as exc:
        logger.exception("Job %s failed: %s", job["job_id"][:8], exc)
        error = str(exc)

    predicted_triage: str | None = None
    predicted_triage_ord: int | None = None
    if assessments:
        votes = [a.get("urgency", "").lower() for a in assessments]
        counts = Counter(votes)
        max_count = counts.most_common(1)[0][1]
        tied = [label for label, c in counts.items() if c == max_count and label in triage_ord]
        if tied:
            predicted_triage = max(tied, key=lambda l: triage_ord[l])
            predicted_triage_ord = triage_ord[predicted_triage]

    duration = round(time.perf_counter() - t_start, 2)
    patient_turns = sum(1 for t in transcript if t["role"] == "patient")

    logger.info(
        "Job %s: ds=%s, cond=%s, predicted=%s, gold=%s, turns=%d, %.1fs",
        job["job_id"][:8],
        job["dataset_name"],
        job["condition_name"],
        predicted_triage or "?",
        job["gold_triage"],
        patient_turns,
        duration,
    )

    return {
        "job_id": job["job_id"],
        "case_id": job["case_id"],
        "dataset_name": job["dataset_name"],
        "condition_name": job["condition_name"],
        "repetition": job["repetition"],
        "gold_triage": job["gold_triage"],
        "gold_triage_ord": job["gold_triage_ord"],
        "urgency_levels": job.get("urgency_levels"),
        "predicted_triage": predicted_triage,
        "predicted_triage_ord": predicted_triage_ord,
        "transcript": transcript,
        "assessments": assessments,
        "turn_count": patient_turns,
        "duration_seconds": duration,
        "error": error,
        "gender": job.get("gender", ""),
        "age": job.get("age", ""),
        "race": job.get("race", ""),
        "init_node_outputs": init_node_outputs,
        "patient_node_outputs_per_turn": node_outputs_per_turn,
        "doctor_token_usage": doctor_token_usage,
    }
