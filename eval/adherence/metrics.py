"""Metrics computation for the patient simulator eval.

Consumes judged result dicts (from judge.load_judge_results()) and produces
a structured metrics dict. All param names and labels are sourced from the
schema — no hardcoded param names.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Main metrics function
# ---------------------------------------------------------------------------

def compute_all_metrics(results: list[dict], schema: dict) -> dict:
    """Compute all evaluation metrics from judged results.

    Args:
        results: List of judged result dicts (from judge.load_judge_results()).
                 Each dict must have: param_name, param_value, predicted_option,
                 correct, confidence, vignette_id.
        schema:  Full param_schema.json dict (for labels, option lists).

    Returns:
        {
            macro_accuracy: float,
            per_param:  {param_name: {accuracy, total, correct, label, group_id}},
            per_option: {param_name: {option: {correct, total, accuracy, label}}},
            per_group:  {group_id: {accuracy, label, param_count, total}},
            per_vignette: {vignette_id: {accuracy, total}},
            confusion:  {param_name: {options, matrix, labels}},
            confidence_breakdown: {low/medium/high: {count, accuracy}},
            total_judged: int,
            total_correct: int,
        }
    """
    param_meta = _build_param_meta(schema)
    group_meta = _build_group_meta(schema)

    # Only include param_variation jobs with a valid predicted_option
    judged = [
        r for r in results
        if r.get("job_type") == "param_variation"
        and r.get("param_name")
        and r.get("predicted_option") is not None
    ]

    if not judged:
        return _empty_metrics()

    # ── per_param ─────────────────────────────────────────────────────────
    per_param_correct: dict[str, int] = defaultdict(int)
    per_param_total: dict[str, int] = defaultdict(int)

    # ── per_option ────────────────────────────────────────────────────────
    per_option_correct: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_option_total: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # ── per_vignette ──────────────────────────────────────────────────────
    per_vignette_correct: dict[Any, int] = defaultdict(int)
    per_vignette_total: dict[Any, int] = defaultdict(int)

    # ── confusion ─────────────────────────────────────────────────────────
    # confusion[param_name][actual][predicted] = count
    confusion_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    # ── confidence ────────────────────────────────────────────────────────
    conf_correct: dict[str, int] = defaultdict(int)
    conf_total: dict[str, int] = defaultdict(int)

    for r in judged:
        pname = r["param_name"]
        actual = r.get("param_value", "")
        predicted = r.get("predicted_option", "")
        correct = bool(r.get("correct", False))
        vid = r.get("vignette_id", "unknown")
        confidence = r.get("confidence", "low")

        per_param_total[pname] += 1
        per_option_total[pname][actual] += 1
        per_vignette_total[vid] += 1
        conf_total[confidence] += 1

        if correct:
            per_param_correct[pname] += 1
            per_option_correct[pname][actual] += 1
            per_vignette_correct[vid] += 1
            conf_correct[confidence] += 1

        confusion_counts[pname][actual][predicted] += 1

    # Build per_param
    per_param = {}
    for pname, total in per_param_total.items():
        correct_n = per_param_correct[pname]
        meta = param_meta.get(pname, {})
        per_param[pname] = {
            "accuracy": correct_n / total if total else 0.0,
            "total": total,
            "correct": correct_n,
            "label": meta.get("label", pname),
            "group_id": meta.get("group_id", ""),
        }

    # macro_accuracy = mean of per-param accuracies
    macro_accuracy = (
        sum(v["accuracy"] for v in per_param.values()) / len(per_param)
        if per_param else 0.0
    )

    # Build per_option
    per_option = {}
    for pname, option_totals in per_option_total.items():
        meta = param_meta.get(pname, {})
        option_labels = meta.get("option_labels", {})
        per_option[pname] = {}
        for option_val, total in option_totals.items():
            correct_n = per_option_correct[pname][option_val]
            per_option[pname][option_val] = {
                "correct": correct_n,
                "total": total,
                "accuracy": correct_n / total if total else 0.0,
                "label": option_labels.get(option_val, option_val),
            }

    # Build per_group
    group_param_correct: dict[str, int] = defaultdict(int)
    group_param_total: dict[str, int] = defaultdict(int)
    group_param_count: dict[str, set] = defaultdict(set)

    for pname, pdata in per_param.items():
        gid = pdata["group_id"]
        group_param_correct[gid] += pdata["correct"]
        group_param_total[gid] += pdata["total"]
        group_param_count[gid].add(pname)

    per_group = {}
    for gid, total in group_param_total.items():
        correct_n = group_param_correct[gid]
        meta = group_meta.get(gid, {})
        per_group[gid] = {
            "accuracy": correct_n / total if total else 0.0,
            "label": meta.get("label", gid),
            "param_count": len(group_param_count[gid]),
            "total": total,
            "correct": correct_n,
        }

    # Build per_vignette
    per_vignette = {
        vid: {
            "accuracy": per_vignette_correct[vid] / total if total else 0.0,
            "total": total,
            "correct": per_vignette_correct[vid],
        }
        for vid, total in per_vignette_total.items()
    }

    # Build confusion matrices
    confusion = {}
    for pname, actual_preds in confusion_counts.items():
        meta = param_meta.get(pname, {})
        all_options = meta.get("option_order", sorted(actual_preds.keys()))
        # Ensure all options that appear are included
        all_keys = sorted(
            set(all_options)
            | set(actual_preds.keys())
            | {p for row in actual_preds.values() for p in row.keys()}
        )
        matrix = [
            [confusion_counts[pname][act][pred] for pred in all_keys]
            for act in all_keys
        ]
        option_labels = meta.get("option_labels", {})
        labels = [option_labels.get(k, k) for k in all_keys]
        confusion[pname] = {"options": all_keys, "matrix": matrix, "labels": labels}

    # Build confidence_breakdown
    confidence_breakdown = {}
    for level in ["low", "medium", "high"]:
        total = conf_total[level]
        correct_n = conf_correct[level]
        confidence_breakdown[level] = {
            "count": total,
            "accuracy": correct_n / total if total else 0.0,
        }

    total_correct = sum(per_param_correct.values())
    total_judged = sum(per_param_total.values())

    return {
        "macro_accuracy": round(macro_accuracy, 4),
        "per_param": per_param,
        "per_option": per_option,
        "per_group": per_group,
        "per_vignette": per_vignette,
        "confusion": confusion,
        "confidence_breakdown": confidence_breakdown,
        "total_judged": total_judged,
        "total_correct": total_correct,
    }


def metrics_to_dataframes(metrics: dict) -> dict[str, pd.DataFrame]:
    """Convert metrics dict to a set of DataFrames for display/download.

    Returns:
        {
            "param_df":    per-param accuracy table
            "option_df":   per-option accuracy table (flattened)
            "group_df":    per-group accuracy table
            "vignette_df": per-vignette accuracy table
        }
    """
    # param_df
    param_rows = []
    for pname, data in metrics.get("per_param", {}).items():
        param_rows.append(
            {
                "param_name": pname,
                "label": data["label"],
                "group": data["group_id"],
                "accuracy": round(data["accuracy"], 3),
                "correct": data["correct"],
                "total": data["total"],
            }
        )
    param_df = pd.DataFrame(param_rows).sort_values(
        ["group", "accuracy"], ascending=[True, False]
    ).reset_index(drop=True)

    # option_df
    option_rows = []
    for pname, options in metrics.get("per_option", {}).items():
        for opt_val, data in options.items():
            option_rows.append(
                {
                    "param_name": pname,
                    "option_value": opt_val,
                    "option_label": data["label"],
                    "accuracy": round(data["accuracy"], 3),
                    "correct": data["correct"],
                    "total": data["total"],
                }
            )
    option_df = pd.DataFrame(option_rows).sort_values(
        ["param_name", "option_value"]
    ).reset_index(drop=True)

    # group_df
    group_rows = []
    for gid, data in metrics.get("per_group", {}).items():
        group_rows.append(
            {
                "group_id": gid,
                "label": data["label"],
                "accuracy": round(data["accuracy"], 3),
                "param_count": data["param_count"],
                "total": data["total"],
            }
        )
    group_df = pd.DataFrame(group_rows).sort_values("accuracy", ascending=False).reset_index(drop=True)

    # vignette_df
    vignette_rows = []
    for vid, data in metrics.get("per_vignette", {}).items():
        vignette_rows.append(
            {
                "vignette_id": vid,
                "accuracy": round(data["accuracy"], 3),
                "correct": data["correct"],
                "total": data["total"],
            }
        )
    vignette_df = pd.DataFrame(vignette_rows).sort_values("vignette_id").reset_index(drop=True)

    return {
        "param_df": param_df,
        "option_df": option_df,
        "group_df": group_df,
        "vignette_df": vignette_df,
    }


# ---------------------------------------------------------------------------
# Comparative metrics (discrimination AUC)
# ---------------------------------------------------------------------------

def compute_comparative_metrics(results: list[dict], schema: dict) -> dict:
    """Compute discrimination accuracy from comparative judge results.

    For each (vignette_id, param_name) pair the judge identified which of two
    transcripts was the 'active' one.  Per-param accuracy = fraction correct.
    Macro-AUC = mean of per-param accuracies (0.5 = random chance, 1.0 = perfect).

    Args:
        results: List of comparative result dicts (from load_comparative_results).
                 Each must have: param_name, correct, vignette_id.
        schema:  Full param_schema.json dict (for labels).

    Returns:
        {
            macro_auc:    float,
            per_param:    {param_name: {auc, n_pairs, n_correct, label, group_id}},
            per_group:    {group_id: {auc, label, param_count, n_pairs}},
            per_vignette: {vignette_id: {auc, n_pairs, n_correct}},
            total_pairs:  int,
            total_correct: int,
        }
    """
    param_meta = _build_param_meta(schema)
    _EXTRA_PARAM_META = {
        "sensitive_topics":             {"label": "Sensitive Topics",             "group_id": "strategy_params"},
        "patient_goal":                 {"label": "Patient Goal",                "group_id": "strategy_params"},
    }
    for k, v in _EXTRA_PARAM_META.items():
        param_meta.setdefault(k, v)
    group_meta = _build_group_meta(schema)

    valid = [
        r for r in results
        if r.get("param_name") and r.get("correct") is not None
    ]

    if not valid:
        return _empty_comparative_metrics()

    from collections import defaultdict

    param_correct: dict[str, int] = defaultdict(int)
    param_total:   dict[str, int] = defaultdict(int)
    vignette_correct: dict[str, int] = defaultdict(int)
    vignette_total:   dict[str, int] = defaultdict(int)

    for r in valid:
        pname   = r["param_name"]
        correct = bool(r["correct"])
        vid     = str(r.get("vignette_id", "unknown"))

        param_total[pname] += 1
        vignette_total[vid] += 1
        if correct:
            param_correct[pname] += 1
            vignette_correct[vid] += 1

    # per_param
    per_param = {}
    for pname, n in param_total.items():
        meta = param_meta.get(pname, {})
        n_correct = param_correct[pname]
        per_param[pname] = {
            "auc":      round(n_correct / n, 4),
            "n_pairs":  n,
            "n_correct": n_correct,
            "label":    meta.get("label", pname),
            "group_id": meta.get("group_id", ""),
        }

    macro_auc = (
        sum(v["auc"] for v in per_param.values()) / len(per_param)
        if per_param else 0.5
    )

    # per_group (micro-averaged within group)
    group_correct: dict[str, int] = defaultdict(int)
    group_total:   dict[str, int] = defaultdict(int)
    group_params:  dict[str, set] = defaultdict(set)
    for pname, pdata in per_param.items():
        gid = pdata["group_id"]
        group_params[gid].add(pname)
        group_correct[gid] += pdata["n_correct"]
        group_total[gid]   += pdata["n_pairs"]

    per_group = {}
    for gid, n in group_total.items():
        meta = group_meta.get(gid, {})
        per_group[gid] = {
            "auc":         round(group_correct[gid] / n, 4),
            "label":       meta.get("label", gid),
            "param_count": len(group_params[gid]),
            "n_pairs":     n,
        }

    # per_vignette
    per_vignette = {
        vid: {
            "auc":      round(vignette_correct[vid] / n, 4),
            "n_pairs":  n,
            "n_correct": vignette_correct[vid],
        }
        for vid, n in vignette_total.items()
    }

    total_pairs   = sum(v["n_pairs"]   for v in per_param.values())
    total_correct = sum(v["n_correct"] for v in per_param.values())

    return {
        "macro_auc":    round(macro_auc, 4),
        "per_param":    per_param,
        "per_group":    per_group,
        "per_vignette": per_vignette,
        "total_pairs":  total_pairs,
        "total_correct": total_correct,
    }


def comparative_metrics_to_dataframes(metrics: dict) -> dict[str, pd.DataFrame]:
    """Convert comparative metrics to DataFrames for display/download.

    Returns:
        {
            "param_df":    per-param AUC table
            "group_df":    per-group AUC table
            "vignette_df": per-vignette AUC table
        }
    """
    group_labels = {
        gid: gdata["label"]
        for gid, gdata in metrics.get("per_group", {}).items()
    }
    param_rows = []
    for pname, data in metrics.get("per_param", {}).items():
        gid = data["group_id"]
        param_rows.append({
            "param_name": pname,
            "label":      data["label"],
            "group":      group_labels.get(gid, gid),
            "auc":        data["auc"],
            "n_correct":  data["n_correct"],
            "n_pairs":    data["n_pairs"],
        })
    param_df = (
        pd.DataFrame(param_rows)
        .sort_values(["group", "auc"], ascending=[True, False])
        .reset_index(drop=True)
    )

    group_rows = []
    for gid, data in metrics.get("per_group", {}).items():
        group_rows.append({
            "group_id":    gid,
            "label":       data["label"],
            "auc":         data["auc"],
            "param_count": data["param_count"],
            "n_pairs":     data["n_pairs"],
        })
    group_df = (
        pd.DataFrame(group_rows)
        .sort_values("auc", ascending=False)
        .reset_index(drop=True)
    )

    vignette_rows = []
    for vid, data in metrics.get("per_vignette", {}).items():
        vignette_rows.append({
            "vignette_id": vid,
            "auc":         data["auc"],
            "n_correct":   data["n_correct"],
            "n_pairs":     data["n_pairs"],
        })
    vignette_df = (
        pd.DataFrame(vignette_rows)
        .sort_values("vignette_id")
        .reset_index(drop=True)
    )

    return {"param_df": param_df, "group_df": group_df, "vignette_df": vignette_df}


def compute_verbosity_metrics(results: list[dict]) -> dict:
    """Compute word-count compliance and ordering from verbosity job results.

    For each verbosity level: average words per patient turn, compliance rate
    (fraction of turns within the target range), and whether the level ordering
    holds across vignettes (very_brief < brief < moderate < verbose).

    Args:
        results: All conversation result dicts from the run (any job_type).
                 Verbosity jobs are selected by job_type == "verbosity".

    Returns:
        {
            per_level: {
                level: {
                    mean_words:       float,
                    compliance_rate:  float,   # fraction of turns within target range
                    target_range:     [min, max],
                    n_turns:          int,
                    n_compliant:      int,
                    word_counts:      list[float],  # per-turn counts (for distribution)
                }
            },
            ordering_holds:  bool,   # True if mean_words is strictly increasing across levels
            ordering_detail: list[str],  # e.g. ["very_brief(8.2) < brief(21.4) < ..."]
            total_jobs:      int,
        }
    """
    from eval.adherence.dataset import (
        VERBOSITY_LEVELS, VERBOSITY_RANGES,
    )

    verbosity_results = [r for r in results if r.get("job_type") == "verbosity"]

    # Collect per-turn word counts per level
    level_words: dict[str, list[int]] = {lvl: [] for lvl in VERBOSITY_LEVELS}

    for r in verbosity_results:
        level = r.get("verbosity_level")
        if level not in level_words:
            continue
        for turn in r.get("transcript", []):
            if turn.get("role") == "patient":
                level_words[level].append(len(turn["content"].split()))

    per_level = {}
    for level in VERBOSITY_LEVELS:
        words = level_words[level]
        lo, hi = VERBOSITY_RANGES[level]
        compliant = [w for w in words if lo <= w <= hi]
        mean_words = round(sum(words) / len(words), 2) if words else 0.0
        per_level[level] = {
            "mean_words":      mean_words,
            "compliance_rate": round(len(compliant) / len(words), 4) if words else 0.0,
            "target_range":    [lo, hi],
            "n_turns":         len(words),
            "n_compliant":     len(compliant),
            "word_counts":     words,
        }

    # Check ordering: mean_words strictly increasing across VERBOSITY_LEVELS
    means = [per_level[lvl]["mean_words"] for lvl in VERBOSITY_LEVELS]
    ordering_holds = all(means[i] < means[i + 1] for i in range(len(means) - 1))
    ordering_detail = " < ".join(
        f"{lvl}({per_level[lvl]['mean_words']})" for lvl in VERBOSITY_LEVELS
    )

    return {
        "per_level":       per_level,
        "ordering_holds":  ordering_holds,
        "ordering_detail": ordering_detail,
        "total_jobs":      len(verbosity_results),
    }


def verbosity_metrics_to_dataframes(metrics: dict) -> dict[str, pd.DataFrame]:
    """Convert verbosity metrics to a DataFrame for display."""
    from eval.adherence.dataset import VERBOSITY_LEVELS

    rows = []
    for level in VERBOSITY_LEVELS:
        data = metrics["per_level"].get(level, {})
        lo, hi = data.get("target_range", [None, None])
        rows.append({
            "level":           level,
            "target_range":    f"{lo}–{hi}",
            "mean_words":      data.get("mean_words", 0.0),
            "compliance_rate": data.get("compliance_rate", 0.0),
            "n_compliant":     data.get("n_compliant", 0),
            "n_turns":         data.get("n_turns", 0),
        })
    return {"level_df": pd.DataFrame(rows)}


def compute_max_turns_metrics(results: list[dict]) -> dict:
    """Compute turn-count compliance from max_turns job results.

    For each max_turns level: compliance rate (fraction of conversations where
    patient_turns <= max_turns), mean patient turns, and whether ordering holds
    (mean turns increases with max_turns).

    Args:
        results: All conversation result dicts from the run (any job_type).
                 max_turns jobs are selected by job_type == "max_turns".

    Returns:
        {
            per_level: {
                level: {
                    mean_turns:       float,
                    compliance_rate:  float,
                    n_conversations:  int,
                    n_compliant:      int,
                    turn_counts:      list[int],
                }
            },
            ordering_holds:  bool,
            ordering_detail: str,
            total_jobs:      int,
        }
    """
    from eval.adherence.dataset import MAX_TURNS_LEVELS

    mt_results = [r for r in results if r.get("job_type") == "max_turns"]

    level_turns: dict[int, list[int]] = {lvl: [] for lvl in MAX_TURNS_LEVELS}

    for r in mt_results:
        level = r.get("max_turns_level")
        if level not in level_turns:
            continue
        patient_turns = sum(1 for t in r.get("transcript", []) if t.get("role") == "patient")
        level_turns[level].append(patient_turns)

    per_level = {}
    for level in MAX_TURNS_LEVELS:
        turns = level_turns[level]
        compliant = [t for t in turns if t <= level]
        mean_turns = round(sum(turns) / len(turns), 2) if turns else 0.0
        per_level[level] = {
            "mean_turns":      mean_turns,
            "compliance_rate": round(len(compliant) / len(turns), 4) if turns else 0.0,
            "n_conversations": len(turns),
            "n_compliant":     len(compliant),
            "turn_counts":     turns,
        }

    means = [per_level[lvl]["mean_turns"] for lvl in MAX_TURNS_LEVELS]
    ordering_holds = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
    ordering_detail = " <= ".join(
        f"max_turns={lvl}({per_level[lvl]['mean_turns']})" for lvl in MAX_TURNS_LEVELS
    )

    return {
        "per_level":       per_level,
        "ordering_holds":  ordering_holds,
        "ordering_detail": ordering_detail,
        "total_jobs":      len(mt_results),
    }


def max_turns_metrics_to_dataframes(metrics: dict) -> dict[str, pd.DataFrame]:
    """Convert max_turns metrics to a DataFrame for display."""
    from eval.adherence.dataset import MAX_TURNS_LEVELS

    rows = []
    for level in MAX_TURNS_LEVELS:
        data = metrics["per_level"].get(level, {})
        rows.append({
            "max_turns":       level,
            "mean_turns":      data.get("mean_turns", 0.0),
            "compliance_rate": data.get("compliance_rate", 0.0),
            "n_compliant":     data.get("n_compliant", 0),
            "n_conversations": data.get("n_conversations", 0),
        })
    return {"level_df": pd.DataFrame(rows)}


# ---------------------------------------------------------------------------
# Probing metrics (deterministic compliance checks)
# ---------------------------------------------------------------------------

def _check_stop_gives_up(result: dict) -> dict:
    """Check that patient eventually stops when doctor gives no assessment."""
    patient_stopped = result.get("patient_stopped", False)
    turn_count = result.get("turn_count", 0)
    return {
        "compliant": patient_stopped,
        "patient_stopped": patient_stopped,
        "turn_count": turn_count,
        "detail": f"stopped={patient_stopped} after {turn_count} turns",
    }


def _check_stop_goal_met(result: dict) -> dict:
    """Check that patient stops early when doctor satisfies their goal."""
    patient_stopped = result.get("patient_stopped", False)
    turn_count = result.get("turn_count", 0)
    early = turn_count <= 5
    compliant = patient_stopped and early
    return {
        "compliant": compliant,
        "patient_stopped": patient_stopped,
        "early_stop": early,
        "turn_count": turn_count,
        "detail": f"stopped={patient_stopped} turns={turn_count} (early={early})",
    }


_PROBE_CHECKERS: dict[str, callable] = {
    "stop_gives_up": _check_stop_gives_up,
    "stop_goal_met": _check_stop_goal_met,
}


def compute_probing_metrics(results: list[dict]) -> dict:
    """Compute compliance metrics for probing eval jobs.

    Args:
        results: All conversation result dicts from the run (any job_type).

    Returns:
        {
            per_probe: {
                probe_type: {
                    compliance_rate: float,
                    n_jobs: int,
                    n_compliant: int,
                    label: str,
                    details: list[dict],
                }
            },
            overall_compliance: float,
            total_jobs: int,
        }
    """
    from eval.adherence.dataset import PROBING_DEFS

    probing_results = [r for r in results if r.get("job_type") == "probing"]

    by_probe: dict[str, list[dict]] = defaultdict(list)
    for r in probing_results:
        by_probe[r.get("probe_type", "unknown")].append(r)

    per_probe = {}
    total_compliant = 0

    for probe_type, probe_results in by_probe.items():
        checker = _PROBE_CHECKERS.get(probe_type)
        if not checker:
            continue

        details = []
        n_compliant = 0
        for r in probe_results:
            check = checker(r)
            details.append({
                "job_id": r.get("job_id"),
                "vignette_id": r.get("vignette_id"),
                **check,
            })
            if check["compliant"]:
                n_compliant += 1

        probe_def = PROBING_DEFS.get(probe_type, {})
        per_probe[probe_type] = {
            "compliance_rate": round(n_compliant / len(probe_results), 4) if probe_results else 0.0,
            "n_jobs": len(probe_results),
            "n_compliant": n_compliant,
            "label": probe_def.get("label", probe_type),
            "details": details,
        }
        total_compliant += n_compliant

    overall = round(total_compliant / len(probing_results), 4) if probing_results else 0.0

    return {
        "per_probe": per_probe,
        "overall_compliance": overall,
        "total_jobs": len(probing_results),
    }


def probing_metrics_to_dataframes(metrics: dict) -> dict[str, pd.DataFrame]:
    """Convert probing metrics to a DataFrame for display."""
    rows = []
    for probe_type, data in sorted(metrics.get("per_probe", {}).items()):
        rows.append({
            "probe_type": probe_type,
            "label": data.get("label", probe_type),
            "compliance_rate": data.get("compliance_rate", 0.0),
            "n_compliant": data.get("n_compliant", 0),
            "n_jobs": data.get("n_jobs", 0),
        })
    return {"probe_df": pd.DataFrame(rows)}


def _empty_comparative_metrics() -> dict:
    return {
        "macro_auc":    0.5,
        "per_param":    {},
        "per_group":    {},
        "per_vignette": {},
        "total_pairs":  0,
        "total_correct": 0,
    }


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _build_param_meta(schema: dict) -> dict[str, dict]:
    """Build param_name → {label, group_id, option_labels, option_order}."""
    result = {}
    for group in schema["groups"]:
        for pname, pdef in group["params"].items():
            cats = pdef.get("categories", {})
            result[pname] = {
                "label": pdef.get("label", pname),
                "group_id": group["id"],
                "option_labels": {k: v.get("label", k) for k, v in cats.items()},
                "option_order": list(cats.keys()),
            }
    return result


def _build_group_meta(schema: dict) -> dict[str, dict]:
    """Build group_id → {label}."""
    return {g["id"]: {"label": g.get("label", g["id"])} for g in schema["groups"]}


def _empty_metrics() -> dict:
    return {
        "macro_accuracy": 0.0,
        "per_param": {},
        "per_option": {},
        "per_group": {},
        "per_vignette": {},
        "confusion": {},
        "confidence_breakdown": {"low": {"count": 0, "accuracy": 0.0},
                                  "medium": {"count": 0, "accuracy": 0.0},
                                  "high": {"count": 0, "accuracy": 0.0}},
        "total_judged": 0,
        "total_correct": 0,
    }
