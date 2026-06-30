"""Aggregate metrics for clinical fidelity evaluation."""
from __future__ import annotations

import pandas as pd


SUPPORTED_LABELS = {4: "accurate", 3: "imprecise", 2: "distorted", 1: "contradicted"}
UNSUPPORTED_LABELS = {4: "expected", 3: "plausible", 2: "questionable", 1: "implausible"}


def compute_fidelity_metrics(results: list[dict]) -> dict:
    """Compute aggregate fidelity metrics from a list of per-conversation results.

    Each result dict should contain:
        vignette_id, vignette_claims, conversation_patient_claims,
        classified_claims, vignette_claims_not_mentioned

    Returns a metrics dict with overall, per-vignette, and distribution breakdowns.
    """
    all_classified: list[dict] = []
    all_vignette_claims: int = 0
    all_mentioned: int = 0
    all_not_mentioned: int = 0
    per_vignette: dict[int, dict] = {}

    for result in results:
        vid = result.get("vignette_id", -1)
        classified = result.get("classified_claims", [])
        vc = result.get("vignette_claims", [])
        not_mentioned = result.get("vignette_claims_not_mentioned", [])

        n_vc = len(vc)
        n_not_mentioned = len(not_mentioned)
        n_mentioned = n_vc - n_not_mentioned

        all_classified.extend(classified)
        all_vignette_claims += n_vc
        all_mentioned += n_mentioned
        all_not_mentioned += n_not_mentioned

        v_metrics = _compute_single(classified, n_vc, n_mentioned, n_not_mentioned)
        v_metrics["summary"] = result.get("summary", "")
        per_vignette[vid] = v_metrics

    overall = _compute_single(
        all_classified, all_vignette_claims, all_mentioned, all_not_mentioned,
    )
    overall["total_conversations"] = len(results)

    return {
        "overall": overall,
        "per_vignette": per_vignette,
    }


def _compute_single(
    classified: list[dict],
    n_vignette_claims: int,
    n_mentioned: int,
    n_not_mentioned: int,
) -> dict:
    supported = [c for c in classified if c.get("category") == "supported"]
    unsupported = [c for c in classified if c.get("category") == "unsupported"]

    n_total = len(classified)
    n_supported = len(supported)
    n_unsupported = len(unsupported)

    sup_scores = [c["score"] for c in supported if "score" in c]
    unsup_scores = [c["score"] for c in unsupported if "score" in c]

    sup_dist = _score_distribution(sup_scores)
    unsup_dist = _score_distribution(unsup_scores)

    coverage_rate = n_mentioned / n_vignette_claims if n_vignette_claims else 0
    unsupported_rate = n_unsupported / n_total if n_total else 0
    confirmed_rate = (
        sum(1 for s in sup_scores if s == 4) / n_supported if n_supported else 0
    )
    contradiction_rate = (
        sum(1 for s in sup_scores if s == 1) / n_supported if n_supported else 0
    )

    return {
        "total_claims": n_total,
        "n_supported": n_supported,
        "n_unsupported": n_unsupported,
        "mean_accuracy": _mean(sup_scores),
        "mean_plausibility": _mean(unsup_scores),
        "supported_distribution": sup_dist,
        "unsupported_distribution": unsup_dist,
        "n_vignette_claims": n_vignette_claims,
        "n_vignette_mentioned": n_mentioned,
        "n_vignette_not_mentioned": n_not_mentioned,
        "coverage_rate": round(coverage_rate, 4),
        "unsupported_rate": round(unsupported_rate, 4),
        "confirmed_rate": round(confirmed_rate, 4),
        "contradiction_rate": round(contradiction_rate, 4),
    }


def _score_distribution(scores: list[int]) -> dict[int, int]:
    dist = {1: 0, 2: 0, 3: 0, 4: 0}
    for s in scores:
        if s in dist:
            dist[s] += 1
    return dist


def _mean(values: list) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


# ---------------------------------------------------------------------------
# DataFrame conversion for display / notebooks
# ---------------------------------------------------------------------------

def fidelity_metrics_to_dataframes(metrics: dict) -> dict[str, pd.DataFrame]:
    """Convert fidelity metrics to DataFrames for display."""
    overall = metrics["overall"]

    summary_rows = [
        {"metric": "Total conversations", "value": overall.get("total_conversations", 0)},
        {"metric": "Total patient claims", "value": overall["total_claims"]},
        {"metric": "Supported claims", "value": overall["n_supported"]},
        {"metric": "Unsupported claims", "value": overall["n_unsupported"]},
        {"metric": "Mean accuracy (supported, 1-4)", "value": overall["mean_accuracy"]},
        {"metric": "Mean plausibility (unsupported, 1-4)", "value": overall["mean_plausibility"]},
        {"metric": "Coverage rate", "value": f"{overall['coverage_rate']:.1%}"},
        {"metric": "Unsupported rate", "value": f"{overall['unsupported_rate']:.1%}"},
        {"metric": "Confirmed rate (score=4)", "value": f"{overall['confirmed_rate']:.1%}"},
        {"metric": "Contradiction rate (score=1)", "value": f"{overall['contradiction_rate']:.1%}"},
    ]
    summary_df = pd.DataFrame(summary_rows)

    sup_dist = overall["supported_distribution"]
    unsup_dist = overall["unsupported_distribution"]
    dist_rows = []
    for score in [4, 3, 2, 1]:
        dist_rows.append({
            "score": score,
            "supported_label": SUPPORTED_LABELS[score],
            "supported_count": sup_dist.get(score, 0),
            "unsupported_label": UNSUPPORTED_LABELS[score],
            "unsupported_count": unsup_dist.get(score, 0),
        })
    dist_df = pd.DataFrame(dist_rows)

    vignette_rows = []
    for vid, vm in sorted(metrics["per_vignette"].items()):
        vignette_rows.append({
            "vignette_id": vid,
            "claims": vm["total_claims"],
            "supported": vm["n_supported"],
            "unsupported": vm["n_unsupported"],
            "accuracy": vm["mean_accuracy"],
            "plausibility": vm["mean_plausibility"],
            "coverage": f"{vm['coverage_rate']:.1%}",
            "unsup_rate": f"{vm['unsupported_rate']:.1%}",
            "confirmed": f"{vm['confirmed_rate']:.1%}",
            "contradiction": f"{vm['contradiction_rate']:.1%}",
        })
    vignette_df = pd.DataFrame(vignette_rows)

    return {
        "summary_df": summary_df,
        "distribution_df": dist_df,
        "vignette_df": vignette_df,
    }
