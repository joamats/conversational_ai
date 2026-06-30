"""Sequential LLM Turing judge — mirrors the human rater protocol.

Unlike the independent judge in llm_judge.py, this module presents
conversations one at a time inside a single multi-turn LLM session.
The judge sees every prior transcript and its own prior classification
before judging the next conversation, exactly as a human rater would.

Rater assignments are pre-built by build_rater_assignments.py so that
human rater i, LLM rater i (opus), and LLM rater i (gpt) all see the
exact same 10 conversations in the exact same order.

Usage:
    python -m eval.turing.llms.sequential_judge
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from eval.turing.llms.judge_clients import (
    JUDGE_CONFIGS,
    create_judge_client,
)
from eval.turing.llms.llm_judge import (
    _format_transcript,
    _parse_json_response,
)
from eval.turing.preprocessing.build_rater_assignments import (
    DEFAULT_ASSIGNMENTS_PATH,
    N_RATERS,
    load_rater_assignments,
    load_rater_items,
)
from eval.turing.preprocessing.data import (
    DEFAULT_REAL_PATH,
    load_conversations,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"

SYSTEM_PROMPT = """\
You are an expert evaluator in a study assessing whether the PATIENT in \
doctor-patient conversations is a real human or an AI simulator.

The DOCTOR is always an AI system — ignore the doctor's style entirely. \
Focus ONLY on the PATIENT's responses.

You will be shown {n} conversations one at a time. For each one, classify \
the patient and explain your reasoning.

Respond with EXACTLY this JSON (no markdown, no extra text):
{{
  "classification": "real" or "simulated",
  "likert_score": 1-4,
  "reasoning": "your detailed explanation"
}}

Likert scale:
  1 = Definitely simulated — clear AI patterns, too polished or formulaic
  2 = Probably simulated — some AI-like patterns but not entirely obvious
  3 = Probably real — mostly natural but with minor uncertainties
  4 = Definitely real — unmistakably human communication patterns
"""


def _compute_report(ratings: list[dict]) -> dict:
    """Compute confusion matrix and protocol metrics from a list of ratings."""
    sim = [r for r in ratings if r["ground_truth"] == "simulated"]
    real = [r for r in ratings if r["ground_truth"] == "real"]

    tp = sum(1 for r in sim if r["classification"] == "simulated")
    fn = sum(1 for r in sim if r["classification"] == "real")
    tn = sum(1 for r in real if r["classification"] == "real")
    fp = sum(1 for r in real if r["classification"] == "simulated")

    n = len(ratings)
    correct = tp + tn
    n_sim = tp + fn
    n_real = tn + fp
    detection_rate = tp / n_sim if n_sim else 0
    fooled_rate = fn / n_sim if n_sim else 0
    specificity = tn / n_real if n_real else 0
    balanced_acc = 0.5 * detection_rate + 0.5 * specificity

    avg_likert_real = (
        sum(r["likert_score"] for r in real) / len(real) if real else 0
    )
    avg_likert_sim = (
        sum(r["likert_score"] for r in sim) / len(sim) if sim else 0
    )

    return {
        "total": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "simulation_detection_rate": round(detection_rate, 4),
        "fooled_rate": round(fooled_rate, 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "avg_likert_real": round(avg_likert_real, 2),
        "avg_likert_sim": round(avg_likert_sim, 2),
    }


def run_sequential_session(
    items: list[dict],
    client: Any,
    model: str,
    reasoning_effort: str = "xhigh",
    rater_id: str = "",
) -> dict:
    """Run one sequential rater session (multi-turn LLM conversation).

    The message history accumulates so that conversation k is judged with
    full visibility of conversations 1..k-1 and the judge's own prior
    classifications.
    """
    session_id = str(uuid.uuid4())
    t0 = time.time()

    system = SYSTEM_PROMPT.format(n=len(items))
    messages: list[dict] = [{"role": "system", "content": system}]
    ratings: list[dict] = []

    for order, item in enumerate(items, start=1):
        transcript_text = _format_transcript(item["transcript"])
        user_msg = (
            f"Conversation {order} of {len(items)}.\n\n"
            f"Classify the PATIENT in this conversation:\n\n"
            f"{transcript_text}"
        )
        messages.append({"role": "user", "content": user_msg})

        kwargs: dict = {"model": model, "messages": list(messages)}
        if reasoning_effort != "none":
            kwargs["reasoning_effort"] = reasoning_effort

        t_conv = time.perf_counter()
        resp = client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t_conv

        raw = resp.choices[0].message.content.strip()
        parsed = _parse_json_response(raw)

        classification = parsed.get("classification", "simulated").lower()
        if classification not in ("real", "simulated"):
            classification = "simulated"

        likert = parsed.get("likert_score", 1)
        if not isinstance(likert, int) or not 1 <= likert <= 4:
            likert = 1

        reasoning = parsed.get("reasoning", f"Parse error: {raw[:200]}")

        # Append assistant reply so next turn sees the prior decision.
        messages.append({"role": "assistant", "content": raw})

        gt = item["_ground_truth"]
        rating = {
            "conversation_id": item["id"],
            "pair_id": item["_pair_id"],
            "ground_truth": gt,
            "classification": classification,
            "classified_simulated": classification == "simulated",
            "correct": classification == gt,
            "feedback": reasoning,
            "likert_score": likert,
            "time_seconds": round(latency, 1),
            "presentation_order": order,
            "rater_id": rater_id,
        }
        ratings.append(rating)

        logger.info(
            "  [%d/%d] gt=%-9s  pred=%-9s  correct=%s  (%.1fs)",
            order, len(items), gt, classification,
            classification == gt, latency,
        )

    session = {
        "session_id": session_id,
        "rater_id": rater_id,
        "judge_type": "llm_sequential",
        "started_at": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "report": _compute_report(ratings),
        "ratings": ratings,
    }
    return session


def run_sequential_judge(
    simulated_path: str | Path | None = None,
    real_path: str | Path = DEFAULT_REAL_PATH,
    assignments_path: str | Path = DEFAULT_ASSIGNMENTS_PATH,
    model: str | None = None,
    reasoning_effort: str | None = None,
    judge_configs: list[dict] | None = None,
) -> dict[str, list[dict]]:
    """Run sequential Turing sessions using pre-built rater assignments.

    Each rater index maps to a static set of 10 conversations (5 real,
    5 simulated) shared across all judge models and human raters.

    Returns dict mapping judge name -> list of session dicts.
    """
    if judge_configs is None:
        judge_configs = JUDGE_CONFIGS

    data = load_rater_assignments(assignments_path)
    n_raters = data["n_raters"]

    if simulated_path is None:
        simulated_path = data["simulated_path"]
    logger.info("Using simulated data from: %s", simulated_path)

    real = load_conversations(real_path)
    simulated = load_conversations(simulated_path)

    real_by_id = {c["id"]: c for c in real}
    sim_by_source = {}
    for s in simulated:
        src = s.get("metadata", {}).get("source_session_id", s["id"])
        sim_by_source[src] = s

    logger.info(
        "Sequential judge: %d raters, %d convos/rater",
        n_raters, data["n_conversations_per_rater"],
    )

    all_results: dict[str, list[dict]] = {}

    for config in judge_configs:
        name = config["name"]
        mdl = model or config["model"]
        effort = reasoning_effort or config.get("reasoning_effort", "xhigh")
        client = create_judge_client(config["client_type"])

        logger.info("Judge %s (%s): starting %d rater sessions", name, mdl, n_raters)
        sessions: list[dict] = []

        for i in range(n_raters):
            items = load_rater_items(i, real_by_id, sim_by_source, assignments_path)
            rater_id = f"{name}_rater_{i}"

            logger.info("  Rater %s", rater_id)
            session = run_sequential_session(
                items=items,
                client=client,
                model=mdl,
                reasoning_effort=effort,
                rater_id=rater_id,
            )
            session["rater_index"] = i
            sessions.append(session)

        all_results[name] = sessions

    # Save all results into runs/<timestamp>_sequential/
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = RUNS_DIR / f"{ts}_sequential"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, sessions in all_results.items():
        path = output_dir / f"sequential_{name}.json"
        with open(path, "w") as f:
            json.dump(sessions, f, indent=2, default=str)
        logger.info("Saved %d sessions -> %s", len(sessions), path)

    logger.info("All results saved to %s", output_dir)
    return all_results


def _print_aggregate(name: str, sessions: list[dict]) -> None:
    all_ratings = [r for s in sessions for r in s["ratings"]]
    report = _compute_report(all_ratings)

    print(f"\n{'=' * 60}")
    print(f"  Judge: {name}  ({len(sessions)} raters)")
    print(f"{'=' * 60}")
    c = report["confusion"]
    print(f"  Total judgements:            {report['total']}")
    print(f"  Accuracy:                    {report['correct']}/{report['total']} = {report['accuracy']:.0%}")
    print(f"  Simulation detection rate:   {report['simulation_detection_rate']:.0%}")
    print(f"  Fooled rate:                 {report['fooled_rate']:.0%}")
    print(f"  Specificity:                 {report['specificity']:.0%}")
    print(f"  Balanced accuracy:           {report['balanced_accuracy']:.0%}")
    print(f"  Confusion: TP={c['tp']} TN={c['tn']} FP={c['fp']} FN={c['fn']}")

    print(f"\n  Per-rater:")
    for s in sessions:
        r = s["report"]
        print(
            f"    {s['rater_id']:30s}  "
            f"acc={r['accuracy']:.0%}  "
            f"det={r['simulation_detection_rate']:.0%}  "
            f"fooled={r['fooled_rate']:.0%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential LLM Turing judge (mirrors human protocol)"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Override model for all judges")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        help="Override reasoning effort for all judges")
    parser.add_argument("--simulated", type=str, default=None,
                        help="Path to simulated.jsonl (default: from assignments)")
    parser.add_argument("--real", type=str, default=None,
                        help="Path to real.jsonl")
    parser.add_argument("--assignments", type=str, default=None,
                        help="Path to rater_assignments.json")
    parser.add_argument("--judges", nargs="*", default=None,
                        help="Only run these judges (e.g. --judges gpt-5.4). Default: all in JUDGE_CONFIGS")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    judge_configs = None
    if args.judges:
        judge_filter = set(args.judges)
        judge_configs = [c for c in JUDGE_CONFIGS if c["name"] in judge_filter]
        if not judge_configs:
            import sys
            print(f"No matching judges for {args.judges}. "
                  f"Available: {[c['name'] for c in JUDGE_CONFIGS]}", file=sys.stderr)
            sys.exit(1)

    all_results = run_sequential_judge(
        simulated_path=args.simulated or None,
        real_path=args.real or DEFAULT_REAL_PATH,
        assignments_path=args.assignments or DEFAULT_ASSIGNMENTS_PATH,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        judge_configs=judge_configs,
    )

    for name, sessions in sorted(all_results.items()):
        _print_aggregate(name, sessions)


if __name__ == "__main__":
    main()
