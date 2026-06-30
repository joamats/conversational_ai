"""Comparative LLM-as-judge for patient simulator parameter adherence.

Presents two transcripts (active vs neutral) and asks which shows more of the
active behavior. Discrimination accuracy is reported as AUC (0.5 = chance).

judge_conversation_comparative() → single pair judgment
judge_run_comparative()          → all pairs in a run directory
"""
from __future__ import annotations

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval._utils import (
    atomic_write_json,
    format_transcript,
    parse_json_response,
)
from eval.turing.llms.judge_clients import (
    create_judge_client,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Comparative judge
# ---------------------------------------------------------------------------

_COMPARATIVE_SYSTEM_PROMPT = (
    "You are an expert evaluator comparing two simulated patient conversations. "
    "Your task is to identify which conversation shows more of a specific behavioral "
    "trait — based solely on what you observe in the transcripts. "
    "Do NOT use any prior knowledge of how the patients were configured."
)

_COMPARATIVE_USER_TEMPLATE = """\
We are assessing the following behavioral dimension:

**{param_label}**
Definition: {param_description}

You will see two conversations (A and B). One patient was configured as "{active_label}" \
and one as "{neutral_label}". Identify which conversation shows more evidence of \
"{active_label}" behavior.

Observable signals for "{active_label}": {active_signals}

{conversations_block}

Which conversation shows more evidence of "{active_label}"?

Return a valid JSON object with exactly these fields:
{{"more_active": "A or B", "reasoning": "<1-2 sentences>", "evidence": ["<exact verbatim quote from the more-active conversation>", "..."]}}

Guidelines for the "evidence" field:
- Copy exact phrases or sentences from the patient's turns in the chosen conversation — verbatim, character-for-character.
- Include ONLY quotes that differentiate the active conversation from the neutral one. If a phrase could appear in either conversation, omit it.
- Be surgical: select the most revealing fragments, not entire turns. A few words or a single sentence is ideal.
- Do not omit important evidence — if multiple turns contain distinct signals, include one quote per turn.
- Do not paraphrase, summarise, or add commentary inside the quotes.
"""


def judge_conversation_comparative(
    active_transcript: list[dict],
    neutral_transcript: list[dict],
    param_name: str,
    param_label: str,
    param_description: str,
    active_option: dict,
    neutral_option: dict,
    judge_config: dict,
    rng: random.Random | None = None,
) -> dict:
    """Compare two transcripts and identify which shows the active behavior.

    Args:
        active_transcript:  Transcript from the active-configured conversation.
        neutral_transcript: Transcript from the neutral-configured conversation.
        param_name:         Parameter key.
        param_label:        Human-readable label.
        param_description:  Parameter description.
        active_option:      {"value", "label", "description", "prompt_addition"}
        neutral_option:     {"value", "label", "description", "prompt_addition"}
        judge_config:       {"model", "reasoning_effort"}
        rng:                Optional Random instance for A/B assignment (reproducibility).

    Returns:
        {
            "more_active":        "A" or "B",
            "reasoning":          str,
            "evidence":           list[str],   # verbatim quotes from the active transcript
            "latency_ms":         float,
            "a_is_active":        bool,         # which slot held the active transcript
            "correct":            bool,
        }
    """
    if rng is None:
        rng = random.Random()

    # Randomly assign which transcript goes in slot A vs B
    a_is_active = rng.random() < 0.5
    if a_is_active:
        transcript_a, transcript_b = active_transcript, neutral_transcript
    else:
        transcript_a, transcript_b = neutral_transcript, active_transcript

    conv_a = format_transcript(transcript_a)
    conv_b = format_transcript(transcript_b)
    conversations_block = (
        f"--- Conversation A ---\n{conv_a}\n\n"
        f"--- Conversation B ---\n{conv_b}"
    )

    active_signals = active_option.get("prompt_addition", "") or active_option.get("description", "")

    user_content = _COMPARATIVE_USER_TEMPLATE.format(
        param_label=param_label,
        param_description=param_description,
        active_label=active_option.get("label", active_option["value"]),
        neutral_label=neutral_option.get("label", neutral_option["value"]),
        active_signals=active_signals,
        conversations_block=conversations_block,
    )

    model = judge_config.get("model", "gpt-5.4-2026-03-05")
    reasoning_effort = judge_config.get("reasoning_effort", "medium")
    client_type = judge_config.get("client_type", "openai")

    client = create_judge_client(client_type)
    messages = [
        {"role": "system", "content": _COMPARATIVE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    create_kwargs: dict = {"model": model, "messages": messages}
    if client_type == "openai":
        create_kwargs["reasoning_effort"] = reasoning_effort
        create_kwargs["response_format"] = {"type": "json_object"}
        create_kwargs["seed"] = 42

    t0 = time.perf_counter()
    if client_type == "openai":
        response = client.with_options(timeout=180.0).chat.completions.create(**create_kwargs)
    else:
        response = client.chat.completions.create(**create_kwargs)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    usage = response.usage
    reasoning_tokens = getattr(
        getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0
    ) or 0
    cached_tokens = getattr(
        getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0
    ) or 0
    judge_token_usage = {
        "model_id": model,
        "node": "judge_comparative",
        "token_usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
        },
        "latency_ms": latency_ms,
    }

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from comparative judge model")

    parsed = parse_json_response(content)
    if not parsed:
        logger.error("Comparative judge failed to parse JSON — raw=%s", content)
        parsed = {"more_active": "A", "reasoning": content, "evidence": []}

    more_active = parsed.get("more_active", "A")
    if more_active not in ("A", "B"):
        more_active = "A" if "A" in str(more_active).upper() else "B"

    correct = (more_active == "A") == a_is_active

    parsed["more_active"] = more_active
    parsed.setdefault("reasoning", "")
    evidence = parsed.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    parsed["evidence"] = evidence if isinstance(evidence, list) else []
    parsed["latency_ms"] = latency_ms
    parsed["a_is_active"] = a_is_active
    parsed["correct"] = correct
    parsed["judge_token_usage"] = [judge_token_usage]
    return parsed


def judge_run_comparative(
    run_dir: str,
    judge_config: dict,
    schema: dict,
    max_workers: int = 4,
    progress_callback=None,
    seed: int = 42,
) -> list[dict]:
    """Run comparative judging on all (active, neutral) conversation pairs.

    Active param_variation jobs reference their neutral via neutral_job_id.
    The neutral transcript is shared across all dimensions for the same vignette.

    Writes one result to conversations/{vignette_id}_{pair_key}_comparative.json per pair.
    Returns list of all comparative result dicts.
    """
    conv_dir = Path(run_dir) / "conversations"
    if not conv_dir.exists():
        return []

    # Index all results by job_id
    results_by_job_id: dict[str, dict] = {}
    active_results: list[dict] = []

    for path in conv_dir.glob("*.json"):
        if path.stem.endswith("_judged") or path.stem.endswith("_comparative"):
            continue
        try:
            with open(path) as f:
                result = json.load(f)
        except Exception:
            continue
        job_id = result.get("job_id")
        if job_id:
            results_by_job_id[job_id] = result
        if result.get("neutral_job_id") and result.get("transcript"):
            active_results.append(result)

    # Pair each active job with its neutral via neutral_job_id
    pairs_to_judge = []
    for active_result in active_results:
        neutral_job_id = active_result.get("neutral_job_id")
        if not neutral_job_id:
            continue
        neutral_result = results_by_job_id.get(neutral_job_id)
        if not neutral_result or not neutral_result.get("transcript"):
            continue
        vid = active_result.get("vignette_id")
        pair_key = active_result.get("pair_key")
        out_path = conv_dir / f"{vid}_{pair_key}_comparative.json"
        if out_path.exists():
            continue
        pairs_to_judge.append({
            "vignette_id": vid,
            "pair_key": pair_key,
            "active_result": active_result,
            "neutral_result": neutral_result,
            "out_path": out_path,
        })

    if not pairs_to_judge:
        return []

    comparative_results = []
    total = len(pairs_to_judge)
    done = 0
    base_rng = random.Random(seed)

    def _judge_one(pair_seed):
        pair, pair_rng_seed = pair_seed
        rng = random.Random(pair_rng_seed)

        active_result = pair["active_result"]
        neutral_result = pair["neutral_result"]
        param_name = active_result.get("param_name", "")
        param_label = active_result.get("param_label", param_name)
        param_description = active_result.get("param_description", "")
        active_option = active_result.get("active_option") or {}
        neutral_option = active_result.get("neutral_option") or {}
        active_val = active_result.get("active_value")
        neutral_val = active_result.get("neutral_value")

        try:
            judgment = judge_conversation_comparative(
                active_transcript=active_result["transcript"],
                neutral_transcript=neutral_result["transcript"],
                param_name=param_name,
                param_label=param_label,
                param_description=param_description,
                active_option=active_option,
                neutral_option=neutral_option,
                judge_config=judge_config,
                rng=rng,
            )
        except Exception as exc:
            logger.error(
                "Comparative judge failed for %s pair=%s vignette=%s: %s",
                param_name, pair["pair_key"], pair["vignette_id"], exc,
            )
            judgment = {
                "more_active": "unknown",
                "reasoning": str(exc),
                "evidence": [],
                "latency_ms": 0,
                "a_is_active": True,
                "correct": False,
                "judge_token_usage": [],
            }

        result = {
            "vignette_id": pair["vignette_id"],
            "case_id": active_result.get("case_id"),
            "param_name": param_name,
            "param_label": param_label,
            "pair_key": pair["pair_key"],
            "active_job_id": active_result["job_id"],
            "neutral_job_id": neutral_result["job_id"],
            "active_value": active_val,
            "neutral_value": neutral_val,
            "a_is_active": judgment["a_is_active"],
            "more_active": judgment["more_active"],
            "correct": judgment["correct"],
            "judge_reasoning": judgment.get("reasoning", ""),
            "judge_evidence": judgment.get("evidence", []),
            "judge_latency_ms": judgment.get("latency_ms", 0),
            "judge_token_usage": judgment.get("judge_token_usage", []),
        }

        correct_flag = "OK   " if result["correct"] else "WRONG"
        logger.info(
            "[comparative] %s | pair=%-40s active=%-20s neutral=%-12s predicted=%s | %s",
            correct_flag,
            pair["pair_key"],
            str(active_val)[:20],
            str(neutral_val)[:12],
            result["more_active"],
            result["judge_reasoning"][:80],
        )

        atomic_write_json(result, pair["out_path"])
        return result

    pairs_with_seeds = [(pair, base_rng.randint(0, 2**31)) for pair in pairs_to_judge]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_judge_one, ps): ps for ps in pairs_with_seeds}
        for future in as_completed(futures):
            done += 1
            try:
                result = future.result()
                if result:
                    comparative_results.append(result)
            except Exception as exc:
                logger.error("Comparative future error: %s", exc)

            if progress_callback:
                info = pairs_with_seeds[done - 1][0]
                progress_callback(
                    done, total,
                    f"comparative pair={info['pair_key']} vignette={info['vignette_id']}",
                )

    return comparative_results


def load_comparative_results(run_dir: str) -> list[dict]:
    """Load all *_comparative.json files from a run's conversations directory."""
    conv_dir = Path(run_dir) / "conversations"
    results = []
    if not conv_dir.exists():
        return results
    for path in conv_dir.glob("*_comparative.json"):
        try:
            with open(path) as f:
                results.append(json.load(f))
        except Exception:
            pass
    return results


