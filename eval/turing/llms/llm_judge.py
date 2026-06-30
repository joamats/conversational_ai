"""Three-mode LLM judge for the Turing test evaluation.

Modes:
  1. Absolute — classify a single conversation as real/simulated,
     assign a Likert score (1-4), and provide reasoning.
  2. Comparative (forced) — given two conversations from the same patient
     (one real, one simulated) in randomised order, decide which is which.
  3. Comparative (unsure allowed) — same as mode 2, but the judge may
     answer "unsure" if it genuinely cannot tell.

All modes use structured JSON output with a reasoning field.
"""
from __future__ import annotations

import json
import logging
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from eval.turing.preprocessing.data import (
    DEFAULT_REAL_PATH,
    DEFAULT_SIMULATED_PATH,
    load_conversations,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ABSOLUTE_SYSTEM_PROMPT = """\
You are an expert evaluator assessing whether the PATIENT in a doctor-patient \
conversation is a real human or an AI simulator.

The DOCTOR is always an AI — ignore the doctor's style entirely.
Focus ONLY on the PATIENT's responses.

Respond with EXACTLY this JSON (no markdown, no extra text):
{
  "classification": "real" or "simulated",
  "likert_score": 1-4,
  "reasoning": "detailed explanation of why you classified this conversation as real or simulated"
}

Likert scale:
  1 = Definitely simulated — clear AI patterns, too polished or formulaic
  2 = Probably simulated — some AI-like patterns but not entirely obvious
  3 = Probably real — mostly natural but with minor uncertainties
  4 = Definitely real — unmistakably human communication patterns
"""

COMPARATIVE_FORCED_SYSTEM_PROMPT = """\
You are an expert evaluator comparing two doctor-patient conversations \
involving the SAME patient case. One conversation has a REAL human patient, \
the other has an AI-simulated patient.

The DOCTOR is always an AI in both conversations — ignore the doctor's style.
Focus ONLY on the PATIENT's responses in each conversation.

You MUST choose which conversation has the real patient.

Respond with EXACTLY this JSON (no markdown, no extra text):
{
  "real_is": "A" or "B",
  "reasoning": "detailed explanation comparing the two conversations and why you chose your answer"
}
"""

COMPARATIVE_UNSURE_SYSTEM_PROMPT = """\
You are an expert evaluator comparing two doctor-patient conversations \
involving the SAME patient case. One conversation has a REAL human patient, \
the other has an AI-simulated patient.

The DOCTOR is always an AI in both conversations — ignore the doctor's style.
Focus ONLY on the PATIENT's responses in each conversation.

You may answer "unsure" if you genuinely cannot distinguish between the two.

Respond with EXACTLY this JSON (no markdown, no extra text):
{
  "real_is": "A" or "B" or "unsure",
  "reasoning": "detailed explanation comparing the two conversations and why you chose your answer (or why you are unsure)"
}
"""


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------

def _format_transcript(transcript: list[dict]) -> str:
    parts = []
    for turn in transcript:
        role = turn["role"].upper()
        parts.append(f"{role}: {turn['content']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Pair building
# ---------------------------------------------------------------------------

def build_matched_pairs(
    real: list[dict],
    simulated: list[dict],
) -> list[dict]:
    """Build matched (real, simulated) pairs using session ID linking.

    real.jsonl uses `id` as session_id.
    simulated.jsonl uses `metadata.source_session_id` to link back.
    """
    sim_by_source = {}
    for s in simulated:
        source_id = s.get("metadata", {}).get("source_session_id", "")
        if source_id:
            sim_by_source[source_id] = s

    pairs = []
    for r in real:
        real_id = r["id"]
        if real_id in sim_by_source:
            pairs.append({
                "session_id": real_id,
                "real": r,
                "simulated": sim_by_source[real_id],
            })
    return pairs


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_json_response(raw: str) -> dict:
    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end], strict=False)
        return {}


# ---------------------------------------------------------------------------
# Mode 1: Absolute judge
# ---------------------------------------------------------------------------

def judge_absolute(
    conversation: dict,
    ground_truth: str,
    client: OpenAI,
    model: str,
    reasoning_effort: str = "xhigh",
) -> dict:
    transcript_text = _format_transcript(conversation["transcript"])

    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": ABSOLUTE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify the PATIENT in this conversation:\n\n{transcript_text}"},
        ],
    }
    if reasoning_effort != "none":
        kwargs["reasoning_effort"] = reasoning_effort

    t0 = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    latency = time.perf_counter() - t0

    raw = response.choices[0].message.content.strip()
    parsed = _parse_json_response(raw)

    classification = parsed.get("classification", "simulated").lower()
    if classification not in ("real", "simulated"):
        classification = "simulated"

    likert = parsed.get("likert_score", 1)
    if not isinstance(likert, int) or likert < 1 or likert > 4:
        likert = 1

    is_correct = classification == ground_truth

    result = {
        "conversation_id": conversation["id"],
        "ground_truth": ground_truth,
        "classification": classification,
        "correct": is_correct,
        "likert_score": likert,
        "reasoning": parsed.get("reasoning", f"Parse error: {raw[:200]}"),
        "latency_s": round(latency, 2),
        "transcript": conversation["transcript"],
    }
    sim_params = conversation.get("metadata", {}).get("sim_params")
    if sim_params:
        result["sim_params"] = sim_params
    return result


# ---------------------------------------------------------------------------
# Mode 2 & 3: Comparative judge
# ---------------------------------------------------------------------------

def judge_comparative(
    pair: dict,
    client: OpenAI,
    model: str,
    reasoning_effort: str = "xhigh",
    allow_unsure: bool = False,
    rng: random.Random | None = None,
) -> dict:
    rng = rng or random.Random()

    real_conv = pair["real"]
    sim_conv = pair["simulated"]

    real_is_a = rng.random() < 0.5
    if real_is_a:
        conv_a, conv_b = real_conv, sim_conv
        answer_key = "A"
    else:
        conv_a, conv_b = sim_conv, real_conv
        answer_key = "B"

    text_a = _format_transcript(conv_a["transcript"])
    text_b = _format_transcript(conv_b["transcript"])

    system_prompt = COMPARATIVE_UNSURE_SYSTEM_PROMPT if allow_unsure else COMPARATIVE_FORCED_SYSTEM_PROMPT

    user_content = (
        "Compare the PATIENT in these two conversations from the same clinical case.\n\n"
        "=== CONVERSATION A ===\n"
        f"{text_a}\n\n"
        "=== CONVERSATION B ===\n"
        f"{text_b}"
    )

    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if reasoning_effort != "none":
        kwargs["reasoning_effort"] = reasoning_effort

    t0 = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    latency = time.perf_counter() - t0

    raw = response.choices[0].message.content.strip()
    parsed = _parse_json_response(raw)

    judge_answer = parsed.get("real_is", "").upper()
    valid = {"A", "B", "UNSURE"} if allow_unsure else {"A", "B"}
    if judge_answer not in valid:
        judge_answer = "UNSURE" if allow_unsure else "A"

    if judge_answer == "UNSURE":
        is_correct = None
    else:
        is_correct = judge_answer == answer_key

    result = {
        "session_id": pair["session_id"],
        "real_conversation_id": real_conv["id"],
        "simulated_conversation_id": sim_conv["id"],
        "real_is": answer_key,
        "judge_answer": judge_answer,
        "correct": is_correct,
        "reasoning": parsed.get("reasoning", f"Parse error: {raw[:200]}"),
        "latency_s": round(latency, 2),
        "transcript_a": conv_a["transcript"],
        "transcript_b": conv_b["transcript"],
    }
    sim_params = sim_conv.get("metadata", {}).get("sim_params")
    if sim_params:
        result["sim_params"] = sim_params
    return result


# ---------------------------------------------------------------------------
# Run all three modes
# ---------------------------------------------------------------------------

VALID_MODES = {"absolute", "comparative_unsure"}  # comparative_forced intentionally excluded


def run_turing_judge(
    simulated_path: str | Path = DEFAULT_SIMULATED_PATH,
    real_path: str | Path = DEFAULT_REAL_PATH,
    n_pairs: int = 10,
    model: str = "gpt-5.4-2026-03-05",
    reasoning_effort: str = "xhigh",
    seed: int = 42,
    max_workers: int = 4,
    modes: set[str] | None = None,
    split: str | None = None,
    splits_path: str | Path | None = None,
    real_absolute_results: list[dict] | None = None,
    client: "OpenAI | None" = None,
    judge_name: str | None = None,
) -> dict:
    """Run selected judge modes on matched pairs.

    Args:
        modes: Which modes to run. Defaults to both.
               Valid values: "absolute", "comparative_unsure".
        split: If set, only run on session IDs assigned to this split in splits.json.
               Overrides n_pairs.
        splits_path: Path to splits.json. Defaults to data/splits.json.
        client: Pre-created client with OpenAI-compatible chat.completions.create()
                interface. If None, creates a default OpenAI client.
        judge_name: Optional label included in the saved results filename.
    """
    if modes is None:
        modes = VALID_MODES
    modes = modes & VALID_MODES

    real = load_conversations(real_path)
    simulated = load_conversations(simulated_path)

    pairs = build_matched_pairs(real, simulated)

    if split:
        from eval.turing.preprocessing.build_splits import (
            DEFAULT_SPLITS_PATH,
            load_splits,
        )
        assignments = load_splits(splits_path or DEFAULT_SPLITS_PATH)
        split_names = set(split.split(","))
        split_ids = {sid for sid, s in assignments.items() if s in split_names}
        pairs = [p for p in pairs if p["session_id"] in split_ids]
        logger.info("Filtered to split '%s': %d pairs", split, len(pairs))
    else:
        pairs = pairs[:n_pairs]

    if not pairs:
        raise ValueError("No matched pairs found between real and simulated data.")

    logger.info("Running Turing judge on %d pairs with model %s (modes: %s)",
                len(pairs), model, ", ".join(sorted(modes)))

    if client is None:
        from eval._utils import get_openai_api_key
        api_key = get_openai_api_key()
        client = OpenAI(api_key=api_key, base_url="https://us.api.openai.com/v1/")
    rng = random.Random(seed)

    session_id = str(uuid.uuid4())
    t0 = time.time()

    session: dict = {
        "session_id": session_id,
        "model": model,
        "judge_name": judge_name,
        "reasoning_effort": reasoning_effort,
        "n_pairs": len(pairs),
        "seed": seed,
        "split": split,
        "modes": sorted(modes),
        "started_at": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
    }

    # --- Mode 1: Absolute ---
    if "absolute" in modes:
        absolute_results: list[dict] = []

        # Use cached real results if provided, only judge simulated
        if real_absolute_results is not None:
            absolute_results.extend(real_absolute_results)
            absolute_items = [(pair["simulated"], "simulated") for pair in pairs]
            logger.info("Using %d cached real results, judging %d simulated",
                        len(real_absolute_results), len(absolute_items))
        else:
            absolute_items = []
            for pair in pairs:
                absolute_items.append((pair["real"], "real"))
                absolute_items.append((pair["simulated"], "simulated"))

        if max_workers <= 1:
            for conv, gt in tqdm(absolute_items, desc="Judge (absolute)", unit="conv"):
                result = judge_absolute(conv, gt, client, model, reasoning_effort)
                absolute_results.append(result)
        else:
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for conv, gt in absolute_items:
                    f = executor.submit(judge_absolute, conv, gt, client, model, reasoning_effort)
                    futures[f] = (conv["id"], gt)
                with tqdm(total=len(futures), desc="Judge (absolute)", unit="conv") as pbar:
                    for f in as_completed(futures):
                        result = f.result()
                        absolute_results.append(result)
                        pbar.update(1)

        absolute_results.sort(key=lambda r: r["conversation_id"])
        session["absolute"] = _summarise_absolute(absolute_results)

    # --- Mode 2: Comparative unsure ---
    if "comparative_unsure" in modes:
        comparative_unsure_results: list[dict] = []
        rng2 = random.Random(seed + 1)
        if max_workers <= 1:
            for pair in tqdm(pairs, desc="Judge (comparative unsure)", unit="pair"):
                result = judge_comparative(pair, client, model, reasoning_effort,
                                           allow_unsure=True, rng=rng2)
                comparative_unsure_results.append(result)
        else:
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for pair in pairs:
                    per_pair_rng = random.Random(rng2.randint(0, 2**32))
                    f = executor.submit(judge_comparative, pair, client, model,
                                        reasoning_effort, True, per_pair_rng)
                    futures[f] = pair["session_id"]
                with tqdm(total=len(futures), desc="Judge (comparative unsure)", unit="pair") as pbar:
                    for f in as_completed(futures):
                        result = f.result()
                        comparative_unsure_results.append(result)
                        pbar.update(1)

        comparative_unsure_results.sort(key=lambda r: r["session_id"])
        session["comparative_unsure"] = _summarise_comparative(comparative_unsure_results, allow_unsure=True)

    # --- Save ---
    session["completed_at"] = datetime.now(timezone.utc).isoformat()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name_suffix = f"_{judge_name}" if judge_name else ""
    path = RESULTS_DIR / f"turing_{ts}{name_suffix}_{session_id[:8]}.json"
    with open(path, "w") as f:
        json.dump(session, f, indent=2, default=str)

    logger.info("Results saved to %s", path)
    return session


def _summarise_absolute(results: list[dict]) -> dict:
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    real_items = [r for r in results if r["ground_truth"] == "real"]
    sim_items = [r for r in results if r["ground_truth"] == "simulated"]
    tp = sum(1 for r in sim_items if r["classification"] == "simulated")
    tn = sum(1 for r in real_items if r["classification"] == "real")
    fp = sum(1 for r in real_items if r["classification"] == "simulated")
    fn = sum(1 for r in sim_items if r["classification"] == "real")
    avg_likert_real = (sum(r["likert_score"] for r in real_items) / len(real_items)) if real_items else 0
    avg_likert_sim = (sum(r["likert_score"] for r in sim_items) / len(sim_items)) if sim_items else 0
    accuracy_real = tn / len(real_items) if real_items else 0
    accuracy_sim = tp / len(sim_items) if sim_items else 0
    return {
        "total": n,
        "correct": correct,
        "accuracy": correct / n if n else 0,
        "accuracy_real": round(accuracy_real, 4),
        "accuracy_sim": round(accuracy_sim, 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "avg_likert_real": round(avg_likert_real, 2),
        "avg_likert_sim": round(avg_likert_sim, 2),
        "per_item": results,
    }


def _summarise_comparative(results: list[dict], allow_unsure: bool = False) -> dict:
    n = len(results)
    decided = [r for r in results if r["correct"] is not None]
    unsure = [r for r in results if r["correct"] is None]
    correct = sum(1 for r in decided if r["correct"])
    return {
        "total": n,
        "decided": len(decided),
        "unsure": len(unsure),
        "correct": correct,
        "accuracy_decided": correct / len(decided) if decided else 0,
        "accuracy_total": correct / n if n else 0,
        "per_item": results,
    }


# ---------------------------------------------------------------------------
# Multi-judge: run all judge models in parallel
# ---------------------------------------------------------------------------

def run_multi_judge_turing(
    simulated_path: str | Path = DEFAULT_SIMULATED_PATH,
    real_path: str | Path = DEFAULT_REAL_PATH,
    n_pairs: int = 10,
    reasoning_effort: str = "xhigh",
    seed: int = 42,
    max_workers: int = 4,
    modes: set[str] | None = None,
    split: str | None = None,
    splits_path: str | Path | None = None,
    real_absolute_results: dict[str, list[dict]] | None = None,
    judge_configs: list[dict] | None = None,
) -> dict[str, dict]:
    """Run the Turing judge with multiple models in parallel.

    Each model produces its own independent results. This avoids same-family
    evaluation bias (e.g. Claude judging Claude-simulated patients).

    Args:
        real_absolute_results: Per-judge cache, keyed by judge name.
        judge_configs: List of judge dicts with keys "name", "model",
            "client_type". Defaults to JUDGE_CONFIGS from judge_clients.py.

    Returns:
        Dict mapping judge name -> run_turing_judge result dict.
    """
    from eval.turing.llms.judge_clients import (
        JUDGE_CONFIGS,
        create_judge_client,
    )

    if judge_configs is None:
        judge_configs = JUDGE_CONFIGS

    real_absolute_results = real_absolute_results or {}

    def _run_single(config: dict) -> tuple[str, dict]:
        name = config["name"]
        model = config["model"]
        effort = config.get("reasoning_effort", reasoning_effort)
        logger.info("Starting judge: %s (%s)", name, model)
        client = create_judge_client(config["client_type"])
        cached = real_absolute_results.get(name)
        result = run_turing_judge(
            simulated_path=simulated_path,
            real_path=real_path,
            n_pairs=n_pairs,
            model=model,
            reasoning_effort=effort,
            seed=seed,
            max_workers=max_workers,
            modes=modes,
            split=split,
            splits_path=splits_path,
            real_absolute_results=cached,
            client=client,
            judge_name=name,
        )
        return name, result

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(judge_configs)) as executor:
        futures = {
            executor.submit(_run_single, cfg): cfg["name"]
            for cfg in judge_configs
        }
        for f in as_completed(futures):
            name = futures[f]
            try:
                _, result = f.result()
                results[name] = result
                logger.info("Judge %s complete", name)
            except Exception as e:
                logger.error("Judge %s failed: %s", name, e, exc_info=True)
                results[name] = {"error": str(e)}

    return results
