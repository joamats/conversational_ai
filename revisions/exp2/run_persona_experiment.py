"""Experiment 2 — clinical fidelity across personas.


Design:
  - Sample 150 source cases, preferring cases where the five personas'
    predicted_triage disagreed (i.e. persona affected the clinician model's
    output) — these are the cases where a difference in clinical fidelity,
    if present, would help explain the difference in AI output.
  - For each sampled case, evaluate all 5 persona conversations (baseline +
    informed_advocate, limited_communicator, dismissive_patient,
    anxious_patient) against the *same* source vignette.
  - 150 cases x 5 personas x 2 judge calls (extract, classify) = 1,500 calls.
  - With ``shared_vignette_claims`` (config), each vignette's claims are
    extracted once (+150 calls) and all five persona conversations of a case
    are classified against that same list, so coverage has a common
    denominator within each case. Patient-claim extraction and classification
    prompts are unchanged from Experiment 1.
  - Primary comparison: clinical fidelity across personas *within the same
    source case* (not pooled across cases), plus a secondary split by
    whether the clinician model's predicted_triage was correct.

All output lands in results/persona_runs/{run_id}/:
  run_meta.json   — run config, case selection, and summary stats
  run.log         — full log for this run
  vignette_claims.json — shared per-vignette claims (when shared_vignette_claims)
  fidelity/       — one JSON per conversation (extraction + classification)
  fidelity/metrics.json — persona and cross-persona metrics
  fidelity/table_delta.csv  — within-case delta vs baseline per persona
  fidelity/table_delta_stats.csv — paired t-test, 95% CI, Holm and Wilcoxon p per delta
  fidelity/table_rubric.csv — 1–4 rubric scores, supported/unsupported, coverage (descriptive)
  fidelity/table_rubric_stats.csv — case-level bootstrap 95% CI per rubric value

Usage (from the repository root, so that ``eval`` and ``fairness`` resolve):
    python -m revisions.exp2.run_persona_experiment \\
        --conversations-dir fairness/results/runs/<run_id>/conversations \\
        --fairness-config fairness/configs/config_gemini.json

    python -m revisions.exp2.run_persona_experiment --evaluate-only --run-id <id>
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from tqdm import tqdm

from eval._utils import (
    atomic_write_json,
    compute_cost_from_results,
    load_config,
    print_cost_summary,
    setup_logging,
)
from .shared_claims import evaluate_conversation, extract_vignette_claims
from eval.fidelity.metrics import (
    SUPPORTED_LABELS,
    UNSUPPORTED_LABELS,
    _compute_single,
)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]

logger = logging.getLogger(__name__)

REQUIRED_PERSONAS: tuple[str, ...] = (
    "baseline",
    "informed_advocate",
    "limited_communicator",
    "dismissive_patient",
    "anxious_patient",
)


def _resolve(config: dict, *keys: str) -> dict:
    cfg = dict(config)
    for key in keys:
        p = Path(cfg[key])
        if not p.is_absolute():
            cfg[key] = str(_REPO_ROOT / p)
    return cfg


# ---------------------------------------------------------------------------
# Load already-run persona conversations
# ---------------------------------------------------------------------------

def _load_conversations(conversations_dir: Path) -> list[dict]:
    conversations = []
    for path in sorted(conversations_dir.glob("*.json")):
        try:
            conversations.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read %s — skipping.", path)
    return conversations


def _group_by_case(conversations: list[dict]) -> dict[tuple[str, str], dict[str, dict]]:
    """Group conversations by (dataset_name, case_id) -> {condition_name: conversation}."""
    by_case: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    skipped = 0
    for conv in conversations:
        if conv.get("error") or not conv.get("transcript"):
            skipped += 1
            continue
        key = (conv.get("dataset_name", ""), conv.get("case_id", ""))
        condition = conv.get("condition_name", "")
        if condition not in by_case[key]:
            by_case[key][condition] = conv
    if skipped:
        logger.info("Skipped %d conversations with an error or empty transcript.", skipped)
    return by_case


# ---------------------------------------------------------------------------
# Case selection — 150 cases, preferring persona-sensitive AI output
# ---------------------------------------------------------------------------

def select_cases(
    by_case: dict[tuple[str, str], dict[str, dict]],
    n_cases: int,
    seed: int,
    required_personas: tuple[str, ...] = REQUIRED_PERSONAS,
) -> list[tuple[str, str]]:
    """Pick ``n_cases`` (dataset_name, case_id) keys with all personas present.

    Cases where predicted_triage differs across personas are preferred
    (these are the cases where persona plausibly changed the clinician
    model's output); remaining slots are filled with a random sample of the
    other complete cases.
    """
    complete = [
        key for key, personas in by_case.items()
        if all(p in personas for p in required_personas)
    ]

    def _divergent(key: tuple[str, str]) -> bool:
        preds = {by_case[key][p].get("predicted_triage") for p in required_personas}
        preds.discard(None)
        return len(preds) > 1

    divergent = [key for key in complete if _divergent(key)]
    divergent_set = set(divergent)
    stable = [key for key in complete if key not in divergent_set]

    rng = random.Random(seed)
    rng.shuffle(divergent)
    rng.shuffle(stable)

    selected = (divergent + stable)[:n_cases]
    logger.info(
        "Case pool: %d complete (all %d personas present) out of %d cases seen. "
        "%d divergent (predicted_triage differs across personas), %d stable. "
        "Selected %d: %d divergent + %d stable.",
        len(complete), len(required_personas), len(by_case),
        len(divergent), len(stable),
        len(selected), min(len(divergent), len(selected)), max(0, len(selected) - len(divergent)),
    )
    if len(selected) < n_cases:
        logger.warning(
            "Only %d/%d cases have all %d personas present — fewer cases than requested.",
            len(selected), n_cases, len(required_personas),
        )
    return selected


# ---------------------------------------------------------------------------
# Vignette lookup — same source vignette every persona conversation used
# ---------------------------------------------------------------------------

def build_vignette_lookup(fairness_config: dict) -> dict[tuple[str, str], str]:
    from fairness.dataset import load_all_cases

    cases = load_all_cases(fairness_config["eval"]["datasets"])
    lookup = {(c["dataset_name"], c["case_id"]): c["patient_vignette"] for c in cases}
    logger.info("Loaded %d source-case vignettes for lookup.", len(lookup))
    return lookup


# ---------------------------------------------------------------------------
# Judge-call helpers
# ---------------------------------------------------------------------------

# Rate-limit / transient server errors worth retrying. With a high max_workers,
# Vertex 429s are expected; the judge client itself does not retry.
_RETRY_STATUS = {429, 500, 502, 503, 504, 529}
_RETRY_ATTEMPTS = 6


def _with_retries(fn, *args, **kwargs):
    """Call ``fn``, retrying rate-limit, server, and timeout errors with jittered exponential backoff."""
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError) as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            if (status is not None and status not in _RETRY_STATUS) or attempt == _RETRY_ATTEMPTS - 1:
                raise
            delay = min(60.0, 5.0 * 2 ** attempt) * (0.5 + random.random())
            logger.warning(
                "Judge call failed (%s) — retrying in %.0fs (attempt %d/%d).",
                status or type(exc).__name__, delay, attempt + 1, _RETRY_ATTEMPTS,
            )
            time.sleep(delay)


def _case_key(conv: dict) -> str:
    return f"{conv.get('dataset_name', '')}:{conv.get('case_id', '')}"


# ---------------------------------------------------------------------------
# Step: shared vignette claims — one extraction per source vignette
# ---------------------------------------------------------------------------

def step_extract_vignette_claims(
    case_keys: list[tuple[str, str]],
    vignette_lookup: dict[tuple[str, str], str],
    judge_config: dict,
    run_dir: Path,
    max_workers: int = 4,
) -> dict[str, dict]:
    """Extract each vignette's claims once, so all persona conversations for a
    case are scored against the same list (a common coverage denominator).

    Cached in run_dir/vignette_claims.json, keyed by "dataset_name:case_id".
    """
    path = run_dir / "vignette_claims.json"
    cached: dict[str, dict] = json.loads(path.read_text()) if path.exists() else {}
    todo = [k for k in case_keys if f"{k[0]}:{k[1]}" not in cached and vignette_lookup.get(k)]
    if not todo:
        return cached

    logger.info("Extracting shared vignette claims for %d cases (max_workers=%d)...", len(todo), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_with_retries, extract_vignette_claims, vignette_lookup[k], judge_config): k
            for k in todo
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Vignette claims", unit="case", file=sys.stdout):
            dataset_name, case_id = futures[future]
            try:
                cached[f"{dataset_name}:{case_id}"] = future.result()
            except Exception as exc:
                logger.error("Vignette claim extraction failed for %s:%s: %s", dataset_name, case_id, exc)
    atomic_write_json(cached, path)
    return cached


# ---------------------------------------------------------------------------
# Step: evaluate every conversation for the selected cases
# ---------------------------------------------------------------------------

def step_evaluate(
    selected_cases: list[tuple[str, str]],
    by_case: dict[tuple[str, str], dict[str, dict]],
    vignette_lookup: dict[tuple[str, str], str],
    judge_config: dict,
    run_dir: Path,
    max_workers: int = 4,
    required_personas: tuple[str, ...] = REQUIRED_PERSONAS,
    shared_vignette_claims: bool = False,
) -> list[dict]:
    fidelity_dir = run_dir / "fidelity"
    fidelity_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[dict, str]] = []
    missing_vignette = 0
    for key in selected_cases:
        vignette_text = vignette_lookup.get(key)
        if not vignette_text:
            missing_vignette += 1
            continue
        for persona in required_personas:
            conv = by_case[key].get(persona)
            if conv is not None:
                jobs.append((conv, vignette_text))
    if missing_vignette:
        logger.error("No vignette text for %d selected cases — those cases were skipped entirely.", missing_vignette)

    already_done = {p.stem for p in fidelity_dir.glob("*.json")}
    to_evaluate = [(conv, v) for conv, v in jobs if conv.get("job_id") not in already_done]

    results: list[dict] = []
    for p in fidelity_dir.glob("*.json"):
        if p.name == "metrics.json" or p.name == "cost_summary.json":
            continue
        try:
            results.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            pass

    if not to_evaluate:
        logger.info("All %d persona conversations already evaluated.", len(jobs))
        return results

    vignette_claims: dict[str, dict] = {}
    if shared_vignette_claims:
        needed = sorted({(conv.get("dataset_name", ""), conv.get("case_id", "")) for conv, _ in to_evaluate})
        vignette_claims = step_extract_vignette_claims(needed, vignette_lookup, judge_config, run_dir, max_workers)
        missing = [(conv, v) for conv, v in to_evaluate if _case_key(conv) not in vignette_claims]
        if missing:
            logger.error(
                "No shared vignette claims for %d conversations — skipped; re-run with --evaluate-only to retry.",
                len(missing),
            )
        to_evaluate = [(conv, v) for conv, v in to_evaluate if _case_key(conv) in vignette_claims]

    logger.info(
        "Evaluating %d persona conversations (%d cases x %d personas; max_workers=%d; shared_vignette_claims=%s)...",
        len(to_evaluate), len(selected_cases), len(required_personas), max_workers, shared_vignette_claims,
    )
    print(f"Evaluating {len(to_evaluate)} persona conversations for clinical fidelity...")

    pbar = tqdm(total=len(to_evaluate), desc="Fidelity eval (Exp 2)", unit="conv", file=sys.stdout)

    def _evaluate_one(conv: dict, vignette_text: str) -> dict:
        shared = vignette_claims.get(_case_key(conv))
        fidelity_result = _with_retries(
            evaluate_conversation,
            vignette=vignette_text,
            transcript=conv["transcript"],
            judge_config=judge_config,
            vignette_claims=shared["vignette_claims"] if shared else None,
        )
        result = {
            "job_id": conv["job_id"],
            "dataset_name": conv.get("dataset_name"),
            "case_id": conv.get("case_id"),
            "condition_name": conv.get("condition_name"),
            "repetition": conv.get("repetition", 0),
            "gold_triage": conv.get("gold_triage"),
            "predicted_triage": conv.get("predicted_triage"),
            "turn_count": conv.get("turn_count"),
            "vignette_text": vignette_text,
            "shared_vignette_claims": shared is not None,
            **fidelity_result,
        }
        atomic_write_json(result, fidelity_dir / f"{conv['job_id']}.json")
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_evaluate_one, conv, v): conv for conv, v in to_evaluate}
        for future in as_completed(futures):
            conv = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.error("Fidelity eval failed for job %s: %s", conv.get("job_id", "?")[:8], exc)
            pbar.update(1)
    pbar.close()

    return results


# ---------------------------------------------------------------------------
# Step: report — fidelity by persona, and cross-persona spread per case
# ---------------------------------------------------------------------------

def compute_persona_fidelity_metrics(
    results: list[dict],
    required_personas: tuple[str, ...] = REQUIRED_PERSONAS,
) -> dict:
    """Aggregate fidelity metrics per persona and per (case, persona).

    The primary comparison is ``per_case`` — clinical fidelity across
    personas for the *same* source case — since a pooled, across-case
    comparison would conflate persona effects with case difficulty.
    """
    by_persona: dict[str, list[dict]] = defaultdict(list)
    by_case: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)

    for r in results:
        by_persona[r.get("condition_name", "")].append(r)
        key = (r.get("dataset_name", ""), r.get("case_id", ""))
        by_case[key][r.get("condition_name", "")] = r

    def _case_metrics(r: dict) -> dict:
        classified = r.get("classified_claims", [])
        vc = r.get("vignette_claims", [])
        not_mentioned = r.get("vignette_claims_not_mentioned", [])
        n_vc = len(vc)
        n_not_mentioned = len(not_mentioned)
        n_mentioned = n_vc - n_not_mentioned
        m = _compute_single(classified, n_vc, n_mentioned, n_not_mentioned)
        m["predicted_triage"] = r.get("predicted_triage")
        m["gold_triage"] = r.get("gold_triage")
        m["correct"] = (
            r.get("predicted_triage") == r.get("gold_triage")
            if r.get("predicted_triage") is not None and r.get("gold_triage") is not None
            else None
        )
        return m

    persona_overall = {}
    for persona in required_personas:
        rs = by_persona.get(persona, [])
        classified = [c for r in rs for c in r.get("classified_claims", [])]
        n_vc = sum(len(r.get("vignette_claims", [])) for r in rs)
        n_not_mentioned = sum(len(r.get("vignette_claims_not_mentioned", [])) for r in rs)
        n_mentioned = n_vc - n_not_mentioned
        m = _compute_single(classified, n_vc, n_mentioned, n_not_mentioned)
        m["n_conversations"] = len(rs)
        m["n_correct"] = sum(1 for r in rs if r.get("predicted_triage") == r.get("gold_triage"))
        persona_overall[persona] = m

    per_case: dict[str, dict] = {}
    for (dataset_name, case_id), personas in by_case.items():
        case_key = f"{dataset_name}:{case_id}"
        persona_metrics = {p: _case_metrics(personas[p]) for p in personas}
        preds = {m["predicted_triage"] for m in persona_metrics.values()} - {None}
        ranges = {}
        for range_key, metric in _RANGE_METRICS.items():
            values = [m[metric] for m in persona_metrics.values() if _has_score(m, metric)]
            ranges[range_key] = round(max(values) - min(values), 4) if values else 0.0
        per_case[case_key] = {
            "dataset_name": dataset_name,
            "case_id": case_id,
            "n_personas": len(persona_metrics),
            "personas": persona_metrics,
            "predicted_triage_divergent": len(preds) > 1,
            **ranges,
        }

    n_divergent_cases = sum(1 for c in per_case.values() if c["predicted_triage_divergent"])

    summary = {"n_cases": len(per_case), "n_divergent_cases": n_divergent_cases}
    for range_key in _RANGE_METRICS:
        summary[f"mean_{range_key}_across_personas"] = round(
            sum(c[range_key] for c in per_case.values()) / len(per_case), 4
        ) if per_case else 0.0

    delta_vs_baseline = {
        p: _paired_deltas(per_case, p, reference=required_personas[0])
        for p in required_personas[1:]
    }
    # Holm–Bonferroni across every persona x metric test in the delta table.
    tests = [s for d in delta_vs_baseline.values() for s in d["stats"].values()]
    for s, p_holm in zip(tests, _holm([s["p"] for s in tests])):
        s["p_holm"] = p_holm

    # Table 2 is descriptive only: persona-vs-baseline inference lives in the
    # delta table, so the same comparisons are not tested twice.
    return {
        "by_persona": persona_overall,
        "per_case": per_case,
        "delta_vs_baseline": delta_vs_baseline,
        "baseline_mean": _case_means(per_case, required_personas[0]),
        "rubric_ci": {p: _rubric_bootstrap(per_case, p) for p in required_personas},
        "summary": summary,
    }


# Per-case range key -> per-persona metric it is the (max - min) of.
_RANGE_METRICS: dict[str, str] = {
    "n_supported_range": "n_supported",
    "n_unsupported_range": "n_unsupported",
    "accuracy_range": "mean_accuracy",
    "plausibility_range": "mean_plausibility",
    "coverage_range": "coverage_rate",
    "unsupported_rate_range": "unsupported_rate",
}

# A 1–4 mean is reported as 0.0 when the conversation has no claims of that
# kind, which is not a real score — skip it in ranges and deltas.
_SCORE_COUNT_KEY: dict[str, str] = {
    "mean_accuracy": "n_supported",
    "mean_plausibility": "n_unsupported",
}


def _has_score(m: dict, metric: str) -> bool:
    count_key = _SCORE_COUNT_KEY.get(metric)
    return count_key is None or m[count_key] > 0


def _paired_deltas(per_case: dict[str, dict], persona: str, reference: str) -> dict:
    """Mean within-case (persona - reference) difference for each metric, with paired tests."""
    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    n_paired = 0
    for case in per_case.values():
        ref = case["personas"].get(reference)
        cur = case["personas"].get(persona)
        if ref is None or cur is None:
            continue
        n_paired += 1
        for metric in _RANGE_METRICS.values():
            if _has_score(ref, metric) and _has_score(cur, metric):
                pairs[metric].append((cur[metric], ref[metric]))
    out = {
        f"delta_{metric}": round(sum(c - r for c, r in pairs[metric]) / len(pairs[metric]), 4) if pairs[metric] else 0.0
        for metric in _RANGE_METRICS.values()
    }
    out["n_paired_cases"] = n_paired
    out["stats"] = {metric: _paired_test(pairs[metric]) for metric in _RANGE_METRICS.values()}
    return out


def _paired_test(pairs: list[tuple[float, float]]) -> dict:
    """Paired t-test of (persona, reference) values with the 95% CI of the mean
    difference, plus a Wilcoxon signed-rank p-value as a non-parametric check
    (the 1–4 means are bounded and skewed toward 4).

    Test fields are None when undefined (fewer than 2 pairs, or every pair has
    the same difference).
    """
    cur = np.array([c for c, _ in pairs], dtype=float)
    ref = np.array([r for _, r in pairs], dtype=float)
    diffs = cur - ref
    out = {
        "n": len(pairs),
        "reference_mean": float(ref.mean()) if len(pairs) else None,
        "persona_mean": float(cur.mean()) if len(pairs) else None,
        "delta": float(diffs.mean()) if len(pairs) else None,
        "ci_low": None,
        "ci_high": None,
        "t": None,
        "p": None,
        "p_wilcoxon": None,
    }
    if len(pairs) < 2 or np.all(diffs == diffs[0]):
        return out
    res = ttest_rel(cur, ref)
    ci = res.confidence_interval(confidence_level=0.95)
    out.update(t=float(res.statistic), p=float(res.pvalue), ci_low=float(ci.low), ci_high=float(ci.high))
    out["p_wilcoxon"] = float(wilcoxon(diffs).pvalue)
    return out


def _holm(p_values: list[float | None]) -> list[float | None]:
    """Holm–Bonferroni step-down adjusted p-values (None entries stay None)."""
    ranked = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(ranked):
        running = max(running, min(1.0, (len(ranked) - rank) * p))
        adjusted[i] = running
    return adjusted


def _case_means(per_case: dict[str, dict], persona: str) -> dict:
    """Mean per-case value of each metric for one persona — the reference the deltas are relative to."""
    values: dict[str, list[float]] = defaultdict(list)
    n_cases = 0
    for case in per_case.values():
        m = case["personas"].get(persona)
        if m is None:
            continue
        n_cases += 1
        for metric in _RANGE_METRICS.values():
            if _has_score(m, metric):
                values[metric].append(m[metric])
    out = {
        metric: round(sum(values[metric]) / len(values[metric]), 4) if values[metric] else 0.0
        for metric in _RANGE_METRICS.values()
    }
    out["n_cases"] = n_cases
    return out


# Table 2 metric -> (numerator, denominator) per conversation. The pooled
# value is sum(numerator) / sum(denominator), or sum(numerator) when the
# denominator is None — i.e. exactly the by_persona numbers.
_RUBRIC_METRICS: dict[str, tuple[str, str | None]] = {
    "supported": ("n_supported", None),
    "unsupported": ("n_unsupported", None),
    "coverage": ("n_vignette_mentioned", "n_vignette_claims"),
    "accuracy": ("sup_score_sum", "sup_scored"),
    **{f"{s} {SUPPORTED_LABELS[s]}": (f"sup_{s}", "n_supported") for s in (4, 3, 2, 1)},
    "plausibility": ("unsup_score_sum", "unsup_scored"),
    **{f"{s} {UNSUPPORTED_LABELS[s]}": (f"unsup_{s}", "n_unsupported") for s in (4, 3, 2, 1)},
}

_N_RESAMPLES = 10_000
_SEED = 42


def _dist_count(dist: dict, score: int) -> int:
    # Keys are ints in memory and strings after a JSON round-trip.
    return dist.get(score, dist.get(str(score), 0))


def _rubric_counts(m: dict) -> dict[str, float]:
    sup = {s: _dist_count(m["supported_distribution"], s) for s in (1, 2, 3, 4)}
    unsup = {s: _dist_count(m["unsupported_distribution"], s) for s in (1, 2, 3, 4)}
    return {
        "n_supported": m["n_supported"],
        "n_unsupported": m["n_unsupported"],
        "n_vignette_mentioned": m["n_vignette_mentioned"],
        "n_vignette_claims": m["n_vignette_claims"],
        "sup_score_sum": sum(s * c for s, c in sup.items()),
        "sup_scored": sum(sup.values()),
        "unsup_score_sum": sum(s * c for s, c in unsup.items()),
        "unsup_scored": sum(unsup.values()),
        **{f"sup_{s}": c for s, c in sup.items()},
        **{f"unsup_{s}": c for s, c in unsup.items()},
    }


def _pooled(num: np.ndarray, den: np.ndarray | None) -> np.ndarray:
    """Pooled value over the last (case) axis."""
    total = num.sum(axis=-1)
    if den is None:
        return total
    d = den.sum(axis=-1)
    return np.divide(total, d, out=np.zeros_like(total, dtype=float), where=d > 0)


def _rubric_arrays(cases: list[dict], persona: str) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    counts = [_rubric_counts(c[persona]) for c in cases]
    return {
        name: (
            np.array([c[num] for c in counts], dtype=float),
            np.array([c[den] for c in counts], dtype=float) if den else None,
        )
        for name, (num, den) in _RUBRIC_METRICS.items()
    }


def _rubric_bootstrap(per_case: dict[str, dict], persona: str) -> dict[str, dict]:
    """Pooled Table 2 values for one persona with case-level bootstrap 95% CIs
    (cases are resampled, so claims stay clustered within their conversation)."""
    cases = [c["personas"] for c in per_case.values() if persona in c["personas"]]
    if not cases:
        return {}
    idx = np.random.default_rng(_SEED).integers(0, len(cases), size=(_N_RESAMPLES, len(cases)))
    out = {}
    for name, (num, den) in _rubric_arrays(cases, persona).items():
        boot = _pooled(num[idx], den[idx] if den is not None else None)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[name] = {"value": float(_pooled(num, den)), "ci_low": float(lo), "ci_high": float(hi)}
    return out


def _display_name(persona: str) -> str:
    return persona.replace("_", " ").title()


_ALPHA = 0.05

# Per-persona metric -> column name in the delta tables.
_METRIC_COLUMNS: dict[str, str] = {
    "n_supported": "supported",
    "n_unsupported": "unsupported",
    "mean_accuracy": "accuracy",
    "mean_plausibility": "plausibility",
    "coverage_rate": "coverage",
    "unsupported_rate": "unsup_rate",
}


def _nonzero_decimals(scaled: float, decimals: int) -> int:
    """Add up to two decimals so a non-zero value (e.g. a CI bound of +0.03 pp)
    is not displayed as zero."""
    extra = 0
    while scaled != 0 and round(scaled, decimals + extra) == 0 and extra < 2:
        extra += 1
    return decimals + extra


def _fmt(metric: str, value: float | None, signed: bool = False, unit: bool = True) -> str:
    """Format a metric value; rates are shown as % (or pp when signed)."""
    if value is None:
        return "-"
    sign = "+" if signed else ""
    if metric in ("coverage_rate", "unsupported_rate"):
        text = f"{value * 100:{sign}.{_nonzero_decimals(value * 100, 1)}f}"
        return text + ((" pp" if signed else "%") if unit else "")
    decimals = 3 if metric in _SCORE_COUNT_KEY else 2
    return f"{value:{sign}.{_nonzero_decimals(value, decimals)}f}"


def _fmt_p(p: float | None) -> str:
    """Exact p-value to two significant figures (e.g. 0.047, 0.0036, 1.1e-25)."""
    if p is None:
        return "-"
    return f"{p:#.2g}"


def _fmt_ci(metric: str, s: dict) -> str:
    if s["ci_low"] is None:
        return "-"
    return f"[{_fmt(metric, s['ci_low'], signed=True, unit=False)}, {_fmt(metric, s['ci_high'], signed=True, unit=False)}]"


def _is_significant(s: dict) -> bool:
    return s["p"] is not None and s["p"] < _ALPHA


def _fmt_rubric(name: str, value: float, unit: bool = True) -> str:
    """Format a Table 2 value: claim totals, 1–4 means, or % of claims."""
    if name in ("supported", "unsupported"):
        return f"{value:.0f}"
    if name in ("accuracy", "plausibility"):
        return f"{value:.{_nonzero_decimals(value, 2)}f}"
    text = f"{value * 100:.{_nonzero_decimals(value * 100, 1)}f}"
    return text + ("%" if unit else "")


def _fmt_rubric_ci(name: str, s: dict) -> str:
    return f"[{_fmt_rubric(name, s['ci_low'], unit=False)}, {_fmt_rubric(name, s['ci_high'], unit=False)}]"


def persona_metrics_to_dataframes(metrics: dict) -> dict[str, pd.DataFrame]:
    """Build the Experiment 2 report tables.

    delta_df       — one row per persona with its mean within-case delta
                     (persona - baseline), 95% CI and * for paired t-test
                     p < 0.05, then a "Baseline (mean)" row with the
                     baseline's mean per-case value of each metric.
    delta_stats_df — one row per persona x metric: baseline and persona means,
                     delta, 95% CI, t, and p (paired t-test, Holm-adjusted
                     across all rows, Wilcoxon signed-rank).
    rubric_df      — descriptive, pooled per-persona results on the 1–4 rubric:
                     supported / unsupported counts, coverage, and mean score +
                     score distribution (% of that persona's supported claims on
                     the accuracy scale, % of unsupported claims on the
                     plausibility scale). No significance markers: the
                     persona-vs-baseline tests are in delta_df only.
    rubric_stats_df — one row per persona x Table 2 metric: value with its
                     case-level bootstrap 95% CI.
    """
    delta_rows = []
    stats_rows = []
    for persona, d in metrics["delta_vs_baseline"].items():
        row = {"persona": _display_name(persona), "n_cases": d["n_paired_cases"]}
        for metric, col in _METRIC_COLUMNS.items():
            s = d["stats"][metric]
            star = "*" if _is_significant(s) else ""
            # Format the unrounded delta; d["delta_*"] is pre-rounded to 4 dp and
            # rounding it again can shift the last displayed digit.
            row[col] = f"{_fmt(metric, s['delta'], signed=True)} {_fmt_ci(metric, s)}{star}"
            stats_rows.append({
                "persona": _display_name(persona),
                "metric": col,
                "n": s["n"],
                "baseline": _fmt(metric, s["reference_mean"]),
                "persona_mean": _fmt(metric, s["persona_mean"]),
                "delta": _fmt(metric, s["delta"], signed=True),
                "95% CI": _fmt_ci(metric, s),
                "t": f"{s['t']:.2f}" if s["t"] is not None else "-",
                "p (t-test)": _fmt_p(s["p"]),
                "p (Holm)": _fmt_p(s.get("p_holm")),
                "p (Wilcoxon)": _fmt_p(s["p_wilcoxon"]),
                "sig": star,
            })
        delta_rows.append(row)
    b = metrics["baseline_mean"]
    delta_rows.append({
        "persona": "Baseline (mean)",
        "n_cases": b["n_cases"],
        **{col: _fmt(metric, b[metric]) for metric, col in _METRIC_COLUMNS.items()},
    })

    # Same row order as delta_df: the non-baseline personas first, baseline last.
    baseline = "baseline"
    order = [p for p in metrics["by_persona"] if p != baseline] + [baseline]
    rubric_rows = []
    rubric_stats_rows = []
    for persona in order:
        m = metrics["by_persona"][persona]
        ci = metrics["rubric_ci"][persona]
        row = {"persona": _display_name(persona), "n_conv": m["n_conversations"], "n_correct": m["n_correct"]}
        for name in _RUBRIC_METRICS:
            row[name] = _fmt_rubric(name, ci[name]["value"])
            rubric_stats_rows.append({
                "persona": _display_name(persona),
                "metric": name,
                "value": _fmt_rubric(name, ci[name]["value"]),
                "95% CI": _fmt_rubric_ci(name, ci[name]),
            })
        rubric_rows.append(row)

    return {
        "delta_df": pd.DataFrame(delta_rows),
        "delta_stats_df": pd.DataFrame(stats_rows),
        "rubric_df": pd.DataFrame(rubric_rows),
        "rubric_stats_df": pd.DataFrame(rubric_stats_rows),
    }


def step_report(results: list[dict], run_dir: Path) -> dict:
    metrics = compute_persona_fidelity_metrics(results)
    atomic_write_json(metrics, run_dir / "fidelity" / "metrics.json")

    dfs = persona_metrics_to_dataframes(metrics)
    dfs["delta_df"].to_csv(run_dir / "fidelity" / "table_delta.csv", index=False)
    dfs["delta_stats_df"].to_csv(run_dir / "fidelity" / "table_delta_stats.csv", index=False)
    dfs["rubric_df"].to_csv(run_dir / "fidelity" / "table_rubric.csv", index=False)
    dfs["rubric_stats_df"].to_csv(run_dir / "fidelity" / "table_rubric_stats.csv", index=False)

    print(f"\nClinical Fidelity Eval — Experiment 2 ({metrics['summary']['n_cases']} cases)")
    print(f"  Cases with divergent predicted_triage across personas: {metrics['summary']['n_divergent_cases']}")
    print(f"  Mean per-case accuracy range across personas:  {metrics['summary']['mean_accuracy_range_across_personas']:.3f}")
    print(f"  Mean per-case coverage range across personas:   {metrics['summary']['mean_coverage_range_across_personas']:.1%}")
    print("\nTable 1 — Within-case delta vs baseline. Persona rows: mean of (persona - baseline) over the")
    print("          same cases [95% CI]. Baseline (mean) row: baseline's mean per-case value. Counts are per")
    print(f"          conversation. * = two-sided paired t-test p < {_ALPHA}, unadjusted:")
    print(dfs["delta_df"].to_string(index=False))
    n_tests = len(dfs["delta_stats_df"])
    print(f"\nTable 1b — Paired tests vs baseline, exact p (Holm adjusts across all {n_tests} tests; Wilcoxon")
    print("           signed-rank as a non-parametric sensitivity check):")
    print(dfs["delta_stats_df"].to_string(index=False))
    print("\nTable 2 — 1–4 rubric by persona, descriptive. Pooled over all claims (conversations with more")
    print("          claims weigh more; per-conversation means and the tests vs baseline are in Table 1).")
    print("          Score columns are % of that persona's supported claims (accuracy scale) /")
    print("          unsupported claims (plausibility scale):")
    print(dfs["rubric_df"].to_string(index=False))
    print(f"\nTable 2b — Table 2 values with case-level bootstrap 95% CIs ({_N_RESAMPLES:,} resamples, seed {_SEED}):")
    print(dfs["rubric_stats_df"].to_string(index=False))

    logger.info(
        "Persona fidelity metrics written. n_cases=%d n_divergent=%d",
        metrics["summary"]["n_cases"], metrics["summary"]["n_divergent_cases"],
    )

    # Shared vignette-claim extractions are per case, not per conversation.
    vignette_path = run_dir / "vignette_claims.json"
    vignette_records = list(json.loads(vignette_path.read_text()).values()) if vignette_path.exists() else []
    cost = compute_cost_from_results(results + vignette_records)
    print_cost_summary(cost)
    atomic_write_json(cost, run_dir / "fidelity" / "cost_summary.json")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Experiment 2 clinical-fidelity-across-personas eval.",
    )
    parser.add_argument(
        "--config", default=str(_HERE / "persona_config.json"),
        help="Path to the Experiment 2 config JSON",
    )
    parser.add_argument(
        "--conversations-dir", default=None,
        help="Override: directory of already-run persona conversation JSONs "
             "(one file per job_id, from fairness/simulation.py output)",
    )
    parser.add_argument(
        "--fairness-config", default=None,
        help="Override: path to the fairness config used for the source run "
             "(used only to load source-vignette text by case_id)",
    )
    parser.add_argument(
        "--n-cases", type=int, default=None,
        help="Override: number of source cases to sample (default 150)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Override: number of parallel judge workers",
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Skip case (re-)selection; resume/extend an existing run's evaluation",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Existing run ID (required with --evaluate-only, optional otherwise)",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    config = _resolve(config, "conversations_dir", "fairness_config_path", "results_path")
    if args.conversations_dir:
        config["conversations_dir"] = args.conversations_dir
    if args.fairness_config:
        config["fairness_config_path"] = args.fairness_config
    if args.n_cases:
        config["n_cases"] = args.n_cases
    if args.max_workers:
        config["max_workers"] = args.max_workers

    judge_config = config.get("judge", {"model": "claude-opus-4-6", "client_type": "anthropic_vertex"})
    max_workers = config.get("max_workers", 8)
    shared_vignette_claims = config.get("shared_vignette_claims", False)
    seed = config.get("seed", 42)
    n_cases = config.get("n_cases", 150)

    results_base = Path(config.get("results_path", str(_HERE / "results" / "persona_runs")))

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    run_dir = results_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")

    print(f"run_id: {run_id}")
    print(f"logs:   {run_dir / 'run.log'}")
    logger.info("=== Experiment 2: Clinical Fidelity Across Personas ===")
    logger.info("run_id=%s", run_id)

    conversations = _load_conversations(Path(config["conversations_dir"]))
    logger.info("Loaded %d conversation records from %s.", len(conversations), config["conversations_dir"])
    by_case = _group_by_case(conversations)

    selection_path = run_dir / "case_selection.json"
    if args.evaluate_only and selection_path.exists():
        selected_cases = [tuple(k) for k in json.loads(selection_path.read_text())]
        logger.info("Reusing existing case selection (%d cases) from %s.", len(selected_cases), selection_path)
    else:
        selected_cases = select_cases(by_case, n_cases, seed)
        atomic_write_json([list(k) for k in selected_cases], selection_path)

    with open(config["fairness_config_path"]) as f:
        fairness_config = json.load(f)
    vignette_lookup = build_vignette_lookup(fairness_config)

    results = step_evaluate(
        selected_cases, by_case, vignette_lookup, judge_config, run_dir, max_workers,
        shared_vignette_claims=shared_vignette_claims,
    )

    run_meta = {
        "run_id": run_id,
        "config": config,
        "n_cases_selected": len(selected_cases),
        "n_conversations_evaluated": len(results),
    }

    if results:
        run_meta["metrics"] = step_report(results, run_dir)
    atomic_write_json(run_meta, run_dir / "run_meta.json")

    print(f"\nDone. Results in {run_dir}")
    logger.info("=== Done. Results in %s ===", run_dir)


if __name__ == "__main__":
    main()
