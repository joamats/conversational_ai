"""Analyse the Experiment 1 human-validation workbook: Claude Opus 4.6 judge vs three human graders per record.

    python analyze_exp1_validation.py --workbook grader_annotations.xlsx

Takes the LLM judge's output and the ground truth from annotation_app/data/*.json, writes tables and CSVs to results/.
Needs pandas, numpy, scipy and openpyxl. Deterministic (bootstrap seed below).

Conventions
- The workbook is append-only; the latest row per key is kept.
- Fidelity is computed from `fidelity_claims`, not `fidelity_summary` (see KNOWN_BAD_FIDELITY).
- 95% CIs: percentile cluster bootstrap over records (B = 2000), since items within a conversation are not independent.
- Parameter adherence is a forced choice between two conversations, coded by on-screen position (A/B). Position is
  randomised independently of which conversation carries the setting, so chance agreement is ~50%.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).parent
DATA = ROOT / "annotation_app" / "data"
OUT = ROOT / "results"
SEED = 20260603
N_BOOT = 2000

# One stale duplicate submission (grader DS-04, record F07): treated as not rated.
KNOWN_BAD_FIDELITY = {("DS-04", "F07")}

# Punctuation and capitalisation are rule-based; graders were later not asked (EXCLUDED_PARAMETERS in grader_app.py),
# so only 4/20 records have all three graders. They are reported separately.
PARAM_LABELS = {
    ("anxiety", "anxious"): "Anxiety",
    ("embarrassment", "embarrassed"): "Embarrassment",
    ("recall_level", "low"): "Recall (low)",
    ("recall_level", "high"): "Recall (high)",
    ("oversharing_tendency", "yes"): "Oversharing",
    ("health_literacy", "low"): "Health literacy (low)",
    ("health_literacy", "high"): "Health literacy (high)",
    ("english_proficiency", "non_proficient"): "English proficiency (non-proficient)",
    ("typo_rate", "yes"): "Typos",
    ("patient_goal", "active"): "Patient goal",
    ("punctuation_style", "none"): "Punctuation (none)",
    ("capitalisation", "all_lower"): "Capitalisation (all lower)",
}
PRIMARY = [label for key, label in PARAM_LABELS.items() if key[0] not in ("punctuation_style", "capitalisation")]
FORCED_CHOICE = [label for label in PRIMARY if label != "Patient goal"]

# Health literacy has a low and a high setting (20 records each), shown as one pooled row by default;
# `--split-health-literacy` restores two rows. Rows pooled over all items are unaffected.
HEALTH_LITERACY = ("Health literacy (low)", "Health literacy (high)")
HEALTH_LITERACY_MERGED = "Health literacy (low + high)"
MERGE_HEALTH_LITERACY = True


# ---------------------------------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


def kendall_tau_b(a, b) -> float:
    """Kendall's tau-b (tie-corrected); NaN when either rating is constant."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.nan if np.ptp(a) == 0 or np.ptp(b) == 0 else float(kendalltau(a, b)[0])


def cohen_kappa(x, y, labels, weights: str | None = None) -> float:
    """Cohen's kappa between two label sequences; weights in (None, 'quadratic') for ordinal labels."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) == 0:
        return np.nan
    k = len(labels)
    pos = {v: i for i, v in enumerate(labels)}
    obs = np.zeros((k, k))
    for a, b in zip(x, y):
        obs[pos[a], pos[b]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    if weights == "quadratic":
        idx = np.arange(k)
        w = (idx[:, None] - idx[None, :]) ** 2 / max(k - 1, 1) ** 2
    else:
        w = 1 - np.eye(k)
    de = (w * exp).sum()
    return np.nan if de == 0 else 1 - (w * obs).sum() / de


def kripp_alpha(R, level: str = "nominal") -> float:
    """Krippendorff's alpha for an items x raters matrix (NaN = missing); nominal or ordinal."""
    R = np.asarray(R, dtype=float)
    vals = np.unique(R[~np.isnan(R)])
    k = len(vals)
    if k < 2:
        return np.nan
    counts = np.stack([(R == v).sum(1) for v in vals], axis=1).astype(float)
    m = counts.sum(1)
    keep = m >= 2
    counts, m = counts[keep], m[keep]
    w = counts / (m - 1)[:, None]
    o = counts.T @ w - np.diag(w.sum(0))  # coincidence matrix
    n_c = o.sum(1)
    n = n_c.sum()
    if level == "nominal":
        d = 1 - np.eye(k)
    else:  # ordinal
        d = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                d[i, j] = d[j, i] = (n_c[i:j + 1].sum() - (n_c[i] + n_c[j]) / 2) ** 2
    d_obs = (o * d).sum() / n
    d_exp = (np.outer(n_c, n_c) * d).sum() / (n * (n - 1))
    return np.nan if d_exp == 0 else 1 - d_obs / d_exp


def fleiss_kappa(R, labels) -> float:
    """Fleiss' kappa for an items x raters array of labels; every item must have the same number of ratings (3 here)."""
    R = np.asarray(R)
    n = R.shape[1]
    counts = np.stack([(R == v).sum(1) for v in labels], axis=1).astype(float)
    p_item = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    p_chance = ((counts.sum(0) / counts.sum()) ** 2).sum()
    return np.nan if p_chance == 1 else (p_item.mean() - p_chance) / (1 - p_chance)


def llm_human_kappa(llm, h, labels) -> float:
    """Cohen's kappa between the LLM and the humans, pooling the LLM-vs-grader pairs of every item (3 per item)."""
    h = np.asarray(h)
    return cohen_kappa(np.tile(np.asarray(llm), h.shape[1]), np.concatenate([h[:, j] for j in range(h.shape[1])]), labels)


def cluster_boot(df: pd.DataFrame, cluster: str, stat, n_boot: int = N_BOOT, seed: int = SEED):
    """Percentile CI resampling whole clusters (records) with replacement.

    `stat(df)` returns a float -> (estimate, lo, hi), or a dict of floats -> {key: (estimate, lo, hi)} from one pass.
    """
    df = df.reset_index(drop=True)
    est = stat(df)
    keys = list(est) if isinstance(est, dict) else [None]
    groups = [g.index.to_numpy() for _, g in df.groupby(cluster)]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(keys)))
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        res = stat(df.iloc[np.concatenate([groups[i] for i in pick])])
        draws[b] = [res[k] for k in keys] if keys[0] is not None else [res]
    out = {}
    for j, k in enumerate(keys):
        col = draws[:, j][~np.isnan(draws[:, j])]
        e = est[k] if k is not None else est
        out[k] = (e, np.nan, np.nan) if len(col) < n_boot * 0.9 else (e, *np.percentile(col, [2.5, 97.5]))
    return out if keys[0] is not None else out[None]


def fmt(est, lo, hi, pct: bool = False, digits: int = 2) -> str:
    if est is None or np.isnan(est):
        return "n/a"
    s = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.{digits}f}")
    return f"{s(est)} ({s(lo)}, {s(hi)})" if not np.isnan(lo) else s(est)


# ---------------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------------

def latest(df: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    df = df.assign(_ts=pd.to_datetime(df["saved_at"]), _row=np.arange(len(df)))
    return df.sort_values(["_ts", "_row"]).groupby(key, sort=False).tail(1).drop(columns=["_ts", "_row"]).reset_index(drop=True)

ADDED_COLUMNS = {"is_latest_for_key", "use_in_analysis", "activated_job_id", "activated_conversation",
                 "selected_is_activated", "category_matches_llm"}


def load_workbook(path: Path) -> dict[str, pd.DataFrame]:
    sheets = {s: pd.read_excel(path, s, dtype=str, keep_default_na=False)
              for s in ("parameter_adherence", "fidelity_claims", "fidelity_summary")}
    return {s: df.drop(columns=[c for c in df.columns if c.startswith("llm_") or c in ADDED_COLUMNS])
            for s, df in sheets.items()}


# ---------------------------------------------------------------------------------------------------------------------
# parameter adherence
# ---------------------------------------------------------------------------------------------------------------------

def adherence_items(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tasks = {}
    for rec in json.loads((DATA / "parameter_adherence.json").read_text()):
        for t in rec["tasks"]:
            a, b = (c["job_id"] for c in t["conversations"])
            tasks[t["task_id"]] = {"truth": "A" if t["activated_job_id"] == a else "B",
                                   "llm": "A" if t["llm_selected_job_id"] == a else "B"}
    rows = latest(raw, ["grader_id", "task_id"])
    rows["param"] = [PARAM_LABELS[(n, v)] for n, v in zip(rows.parameter_name, rows.parameter_value)]
    goal = rows.parameter_name == "patient_goal"
    rows["pick"] = rows.selected_conversation
    # goal: Yes/No per conversation; a unique "Yes" is the pick, Yes/Yes or No/No is a tie ("T")
    rows.loc[goal, "pick"] = [{"Yes|No": "A", "No|Yes": "B"}.get(f"{a}|{b}", "T")
                              for a, b in zip(rows.goal_conveyed_a[goal], rows.goal_conveyed_b[goal])]
    rows["truth"] = rows.task_id.map(lambda t: tasks[t]["truth"])
    rows["llm"] = rows.task_id.map(lambda t: tasks[t]["llm"])
    rows = rows.sort_values(["record_id", "param", "grader_id"])
    items = []
    for (rec, param, task), g in rows.groupby(["record_id", "param", "task_id"], sort=True):
        items.append({"record_id": rec, "param": param, "n_humans": len(g), "h1": g.pick.iloc[0], "h2": g.pick.iloc[1] if len(g) > 1 else None,
                      "h3": g.pick.iloc[2] if len(g) > 2 else None, "graders": ",".join(g.grader_id),
                      "truth": g.truth.iloc[0], "llm": g.llm.iloc[0]})
    items = pd.DataFrame(items)
    return items, {"rows": rows}


def majority(picks: list) -> str:
    picks = [p for p in picks if p is not None and p == p]
    vals, cnt = np.unique(picks, return_counts=True)
    return vals[cnt.argmax()] if cnt.max() >= 2 else "none"


def majority3(h: np.ndarray, labels: list[str]) -> np.ndarray:
    """Majority label per row of an (n, 3) array; 'none' when all three differ."""
    counts = np.stack([(h == v).sum(1) for v in labels], axis=1)
    return np.where(counts.max(1) >= 2, np.array(labels)[counts.argmax(1)], "none")


def adherence_metrics(items: pd.DataFrame, light: bool = False) -> dict:
    """Metrics for one set of items (3 human picks + LLM pick + design truth). `light` skips Krippendorff's alpha (bootstrap speed)."""
    h = items[["h1", "h2", "h3"]].to_numpy()
    cons = majority3(h, ["A", "B", "T"])
    llm, truth = items.llm.to_numpy(), items.truth.to_numpy()
    n = len(items)
    ind_acc = np.mean([(h[:, j] == truth).mean() for j in range(3)])
    pair_hh = np.mean([(h[:, i] == h[:, j]).mean() for i, j in ((0, 1), (0, 2), (1, 2))])
    pair_lh = np.mean([(h[:, j] == llm).mean() for j in range(3)])
    labels = ["A", "B", "T"]  # T = tie / no unique pick; an empty category does not change kappa
    cons_k = np.where(cons == "none", "T", cons)
    code = {v: i for i, v in enumerate(labels)}
    alpha = np.nan
    if not light:
        R = np.vectorize(lambda v: code.get(v, np.nan), otypes=[float])(h)
        alpha = kripp_alpha(R, "nominal")
    return {
        "n_items": n,
        "llm_acc_vs_design": (llm == truth).mean(),
        "human_consensus_acc_vs_design": (cons == truth).mean(),
        "human_individual_acc_vs_design": ind_acc,
        "llm_vs_consensus_agree": (llm == cons).mean(),
        "llm_vs_consensus_kappa": cohen_kappa(llm, cons_k, labels),
        "llm_vs_human_pairwise_agree": pair_lh,
        "llm_vs_human_pairwise_kappa": llm_human_kappa(llm, h, labels),
        "human_vs_human_pairwise_agree": pair_hh,
        "human_fleiss_kappa": fleiss_kappa(h, labels),
        "human_alpha": alpha,
        "n_no_majority": int((cons == "none").sum()),
    }


def param_rows(merge_hl: bool) -> list[tuple[str, list[str]]]:
    """(row label, item-level parameter labels pooled into the row). Health literacy low+high can be pooled into one row."""
    out = []
    for p in PRIMARY + ["Punctuation (none)", "Capitalisation (all lower)"]:
        if merge_hl and p == HEALTH_LITERACY[0]:
            out.append((HEALTH_LITERACY_MERGED, list(HEALTH_LITERACY)))
        elif merge_hl and p == HEALTH_LITERACY[1]:
            continue
        else:
            out.append((p, [p]))
    return out


# per-row statistics bootstrapped over records (the k/n columns use Wilson intervals instead)
ROW_BOOT_KEYS = ("llm_vs_consensus_kappa", "llm_vs_human_pairwise_agree", "llm_vs_human_pairwise_kappa",
                 "human_vs_human_pairwise_agree", "human_fleiss_kappa")


def run_adherence(raw: pd.DataFrame, merge_hl: bool = MERGE_HEALTH_LITERACY) -> dict:
    items, extra = adherence_items(raw)
    rows = extra["rows"]
    complete = items[items.n_humans == 3]
    partial = items[items.n_humans < 3]
    # per parameter (row); numeric CI bounds are stored as <metric>_lo / <metric>_hi
    per_param = []
    for label, members in param_rows(merge_hl):
        sub = complete[complete.param.isin(members)]
        if len(sub) == 0:
            continue
        n = len(sub)
        m = adherence_metrics(sub)
        ks = {"llm_acc_vs_design": int(round(m["llm_acc_vs_design"] * n)),
              "human_consensus_acc_vs_design": int(round(m["human_consensus_acc_vs_design"] * n)),
              "llm_vs_consensus_agree": int(round(m["llm_vs_consensus_agree"] * n))}
        m.update({"param": label, "llm_correct": f"{ks['llm_acc_vs_design']}/{n}",
                  "human_consensus_correct": f"{ks['human_consensus_acc_vs_design']}/{n}",
                  "llm_consensus_agree_k": f"{ks['llm_vs_consensus_agree']}/{n}"})
        for key, k in ks.items():
            m[f"{key}_lo"], m[f"{key}_hi"] = wilson(k, n)
        boot = cluster_boot(sub, "record_id", lambda d: {k: adherence_metrics(d, light=True)[k] for k in ROW_BOOT_KEYS})
        for key, (_, lo, hi) in boot.items():
            m[f"{key}_lo"], m[f"{key}_hi"] = lo, hi
        per_param.append(m)
    per_param = pd.DataFrame(per_param)

    # pooled, cluster bootstrap over records
    pooled = {}
    sets = {"9 forced-choice parameters (excl. goal)": complete[complete.param.isin(FORCED_CHOICE)],
            "all 10 parameters (goal ties = no pick)": complete[complete.param.isin(PRIMARY)],
            "Patient goal only": complete[complete.param == "Patient goal"]}
    keep = ("llm_acc_vs_design", "human_consensus_acc_vs_design", "human_individual_acc_vs_design", "llm_vs_consensus_agree",
            "llm_vs_consensus_kappa", "human_alpha", "llm_vs_human_pairwise_agree", "llm_vs_human_pairwise_kappa",
            "human_vs_human_pairwise_agree", "human_fleiss_kappa")

    def pooled_stat(d: pd.DataFrame) -> dict:
        m = adherence_metrics(d)
        return {**{k: m[k] for k in keep}, "pairwise_diff_llm_minus_hh": m["llm_vs_human_pairwise_agree"] - m["human_vs_human_pairwise_agree"]}

    for name, sub in sets.items():
        res = cluster_boot(sub, "record_id", pooled_stat)
        res["n_items"] = len(sub)
        res["n_no_majority"] = adherence_metrics(sub)["n_no_majority"]
        pooled[name] = res

    # patient goal: per-conversation Yes/No against the design (active conversation should convey the goal, neutral should not)
    g = rows[rows.parameter_name == "patient_goal"].copy()
    g["active_yes"] = np.where(g.truth == "A", g.goal_conveyed_a, g.goal_conveyed_b) == "Yes"
    g["neutral_yes"] = np.where(g.truth == "A", g.goal_conveyed_b, g.goal_conveyed_a) == "Yes"
    goal = {"n_ratings": len(g), "active_yes": int(g.active_yes.sum()), "neutral_yes": int(g.neutral_yes.sum()),
            "tie_both_no": int((~g.active_yes & ~g.neutral_yes).sum()), "tie_both_yes": int((g.active_yes & g.neutral_yes).sum()),
            "unique_correct": int((g.active_yes & ~g.neutral_yes).sum()), "unique_wrong": int((~g.active_yes & g.neutral_yes).sum())}
    gi = complete[complete.param == "Patient goal"]
    # human-consensus Yes/No per conversation (majority of 3) vs the LLM's forced choice
    g_items = {}
    for rec, d in g.groupby("record_id"):
        g_items[rec] = {"active_yes_votes": int(d.active_yes.sum()), "neutral_yes_votes": int(d.neutral_yes.sum()), "n": len(d)}
    goal["consensus_active_yes"] = sum(v["active_yes_votes"] >= 2 for v in g_items.values())
    goal["consensus_neutral_yes"] = sum(v["neutral_yes_votes"] >= 2 for v in g_items.values())
    goal["n_records"] = len(g_items)
    llm_goal_correct = int((gi.llm == gi.truth).sum())
    goal["llm_correct"] = llm_goal_correct
    # inter-rater on per-conversation Yes/No (2 conversations x 20 records, 3 raters each)
    mats = []
    for rec, d in g.groupby("record_id"):
        mats.append(d.active_yes.astype(float).to_numpy()); mats.append(d.neutral_yes.astype(float).to_numpy())
    goal["yesno_alpha"] = kripp_alpha(np.array(mats), "nominal")
    # graders could not say "can't tell" in the forced choice; many typo ratings carry a comment saying no typos were visible
    typo = rows[rows.parameter_name == "typo_rate"]
    typo_comments = {"n_ratings": len(typo), "n_with_comment": int((typo.comments.str.strip() != "").sum())}
    return {"items": items, "per_param": per_param, "pooled": pooled, "goal": goal, "partial": partial, "rows": rows, "typo_comments": typo_comments}


# ---------------------------------------------------------------------------------------------------------------------
# clinical fidelity
# ---------------------------------------------------------------------------------------------------------------------

def fidelity_frames(claims_raw: pd.DataFrame, summary_raw: pd.DataFrame):
    recs = {r["record_id"]: r for r in json.loads((DATA / "clinical_fidelity.json").read_text())}
    claims = latest(claims_raw, ["grader_id", "record_id", "claim_number"])
    claims = claims[[(g, r) not in KNOWN_BAD_FIDELITY for g, r in zip(claims.grader_id, claims.record_id)]].copy()
    claims["score"] = claims.score.astype(int)
    claims["claim_number"] = claims.claim_number.astype(int)
    claims["flag"] = claims.score <= 2
    claims["vignette_idx"] = pd.to_numeric(claims.supporting_vignette_claim_number, errors="coerce") - 1

    llm_rows = []
    for rid, rec in recs.items():
        vtext = {t: i for i, t in enumerate(rec["vignette_claims"])}
        for i, c in enumerate(rec["llm_classified_claims"], start=1):
            llm_rows.append({"record_id": rid, "claim_number": i, "llm_cat": c["category"], "llm_score": int(c["score"]),
                             "llm_vidx": vtext.get(c.get("matched_vignette_claim"), np.nan) if c["category"] == "supported" else np.nan})
    llm = pd.DataFrame(llm_rows)
    # one row per human rating with the LLM rating of the same claim
    long = claims.merge(llm, on=["record_id", "claim_number"], how="left", validate="many_to_one")
    assert long.llm_cat.notna().all(), "claim missing from LLM classification"

    summ = latest(summary_raw, ["grader_id", "record_id"])
    summ = summ[[(g, r) not in KNOWN_BAD_FIDELITY for g, r in zip(summ.grader_id, summ.record_id)]].copy()
    return recs, llm, long, summ


def wide_claims(long: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    """One row per claim: LLM rating plus up to 3 human ratings (slots ordered by grader id)."""
    rows = []
    for (rid, cn), g in long.sort_values("grader_id").groupby(["record_id", "claim_number"]):
        r = {"record_id": rid, "claim_number": cn, "n_humans": len(g), "graders": ",".join(g.grader_id),
             "claim": g.claim.iloc[0], "llm_cat": g.llm_cat.iloc[0], "llm_score": g.llm_score.iloc[0], "llm_vidx": g.llm_vidx.iloc[0]}
        for j, (_, h) in enumerate(g.iterrows(), start=1):
            r.update({f"h{j}_cat": h.category, f"h{j}_score": h.score, f"h{j}_vidx": h.vignette_idx})
        rows.append(r)
    w = pd.DataFrame(rows)
    for j in (1, 2, 3):
        for c in ("cat", "score", "vidx"):
            if f"h{j}_{c}" not in w:
                w[f"h{j}_{c}"] = np.nan
    return w


def claim_consensus(w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    cats = w[["h1_cat", "h2_cat", "h3_cat"]].to_numpy(dtype=object)
    w["cons_cat"] = [majority([c for c in r if isinstance(c, str)]) if w.n_humans.iloc[i] == 3 else "none" for i, r in enumerate(cats)]
    flags = (w[["h1_score", "h2_score", "h3_score"]] <= 2).sum(axis=1)
    w["cons_flag"] = np.where(w.n_humans == 3, flags >= 2, np.nan)
    w["n_flag_humans"] = flags
    return w


def category_stats(w: pd.DataFrame) -> dict:
    sub = w[(w.n_humans == 3) & (w.cons_cat != "none")]
    llm, cons = sub.llm_cat.to_numpy(), sub.cons_cat.to_numpy()
    h = sub[["h1_cat", "h2_cat", "h3_cat"]].to_numpy(dtype=object)
    pair_hh = np.mean([(h[:, i] == h[:, j]).mean() for i, j in ((0, 1), (0, 2), (1, 2))])
    pair_lh = np.mean([(h[:, j] == llm).mean() for j in range(3)])
    cats = ["supported", "unsupported"]
    return {"n_claims": len(sub), "llm_vs_consensus_agree": (llm == cons).mean(),
            "llm_vs_consensus_kappa": cohen_kappa(llm, cons, cats),
            "llm_vs_human_pairwise_agree": pair_lh, "human_vs_human_pairwise_agree": pair_hh,
            "llm_vs_human_pairwise_kappa": llm_human_kappa(llm, h, cats), "human_fleiss_kappa": fleiss_kappa(h, cats),
            "llm_unsupported_rate": (llm == "unsupported").mean(), "human_consensus_unsupported_rate": (cons == "unsupported").mean()}


def cat_alpha(w: pd.DataFrame) -> float:
    """Krippendorff's alpha (nominal) among the humans over every claim, including the claims with only 2 raters."""
    R = np.vectorize(lambda v: {"supported": 0.0, "unsupported": 1.0}.get(v, np.nan), otypes=[float])(w[["h1_cat", "h2_cat", "h3_cat"]].to_numpy(dtype=object))
    return kripp_alpha(R, "nominal")


def pair_table(w: pd.DataFrame, kind: str, category: str) -> pd.DataFrame:
    """Rating pairs (LLM-human or human-human) on the same claim where both labelled the claim `category`."""
    out = []
    for _, r in w.iterrows():
        hs = [(r[f"h{j}_cat"], r[f"h{j}_score"]) for j in (1, 2, 3) if isinstance(r[f"h{j}_cat"], str)]
        if kind == "llm_human":
            pairs = [(r.llm_score, s) for c, s in hs if c == category and r.llm_cat == category]
        else:
            pairs = [(hs[i][1], hs[j][1]) for i in range(len(hs)) for j in range(i + 1, len(hs)) if hs[i][0] == category and hs[j][0] == category]
        out += [{"record_id": r.record_id, "a": a, "b": b} for a, b in pairs]
    return pd.DataFrame(out)


def score_pair_stats(pairs: pd.DataFrame) -> dict:
    a, b = pairs.a.to_numpy(), pairs.b.to_numpy()
    if len(a) == 0:
        return {"n_pairs": 0, "exact": np.nan, "within1": np.nan, "wkappa": np.nan, "kendall": np.nan, "mean_diff": np.nan}
    return {"n_pairs": len(a), "exact": (a == b).mean(), "within1": (np.abs(a - b) <= 1).mean(),
            "wkappa": cohen_kappa(a, b, [1, 2, 3, 4], "quadratic"), "mean_diff": (a - b).mean(),
            "kendall": kendall_tau_b(a, b)}


def run_fidelity(claims_raw: pd.DataFrame, summary_raw: pd.DataFrame) -> dict:
    recs, llm, long, summ = fidelity_frames(claims_raw, summary_raw)
    w = claim_consensus(wide_claims(long, llm))
    out = {"recs": recs, "llm": llm, "long": long, "wide": w, "summ": summ}

    cat_set = w[(w.n_humans == 3) & (w.cons_cat != "none")]

    def cat_stat(d: pd.DataFrame) -> dict:
        m = category_stats(d)
        return {k: m[k] for k in ("llm_vs_consensus_agree", "llm_vs_consensus_kappa", "llm_vs_human_pairwise_agree",
                                  "llm_vs_human_pairwise_kappa", "human_vs_human_pairwise_agree", "human_fleiss_kappa",
                                  "llm_unsupported_rate", "human_consensus_unsupported_rate")} | {
            "pair_diff": m["llm_vs_human_pairwise_agree"] - m["human_vs_human_pairwise_agree"], "human_alpha": cat_alpha(d)}
    out["category"] = cluster_boot(cat_set, "record_id", cat_stat)
    out["category"]["human_alpha"] = cluster_boot(w, "record_id", cat_alpha)  # all claims, including those with only 2 raters
    out["category"]["n_claims"] = len(cat_set)
    out["category"]["n_no_majority"] = int(((w.n_humans == 3) & (w.cons_cat == "none")).sum())
    out["category"]["n_two_raters"] = int((w.n_humans < 3).sum())

    # scores within matched category: LLM-human pairs vs human-human pairs, bootstrapped jointly
    scores = {}
    for cat, scale in (("supported", "accuracy"), ("unsupported", "plausibility")):
        lh, hh = pair_table(w, "llm_human", cat), pair_table(w, "human_human", cat)
        pairs = pd.concat([lh.assign(kind="lh"), hh.assign(kind="hh")], ignore_index=True)

        def pair_stat(d: pd.DataFrame) -> dict:
            res = {}
            for kind in ("lh", "hh"):
                st = score_pair_stats(d[d.kind == kind])
                res.update({f"{kind}_{k}": st[k] for k in ("exact", "within1", "wkappa", "kendall")})
            res["lh_mean_diff"] = score_pair_stats(d[d.kind == "lh"])["mean_diff"]  # LLM minus human; human-human has no direction
            res["diff_exact_lh_minus_hh"] = res["lh_exact"] - res["hh_exact"]
            return res
        entry = {"n_llm_human_pairs": len(lh), "n_human_pairs": len(hh)} | cluster_boot(pairs, "record_id", pair_stat)
        # human ratings alone: ordinal alpha over all claims, using only ratings labelled `cat`
        R = np.where(w[["h1_cat", "h2_cat", "h3_cat"]].to_numpy(dtype=object) == cat, w[["h1_score", "h2_score", "h3_score"]].to_numpy(dtype=float), np.nan)
        entry["human_alpha_ordinal"] = cluster_boot(w.assign(_i=np.arange(len(w))), "record_id", lambda d: kripp_alpha(R[d._i.to_numpy()], "ordinal"))
        hr = long[long.category == cat]
        entry["human_dist"] = hr.score.value_counts().reindex([1, 2, 3, 4], fill_value=0).to_dict()
        lr = llm[llm.llm_cat == cat]
        entry["llm_dist"] = lr.llm_score.value_counts().reindex([1, 2, 3, 4], fill_value=0).to_dict()
        entry["human_mean"] = cluster_boot(hr.assign(_s=hr.score.astype(float)), "record_id", lambda d: d._s.mean())
        entry["llm_mean"] = lr.llm_score.mean()
        entry["llm_mean_ci"] = cluster_boot(lr.assign(_s=lr.llm_score.astype(float)), "record_id", lambda d: d._s.mean())
        entry["human_low_rate"] = cluster_boot(hr.assign(_f=hr.flag.astype(float)), "record_id", lambda d: d._f.mean())
        entry["llm_low_rate"] = (lr.llm_score <= 2).mean()
        scores[scale] = entry
    out["scores"] = scores

    # leave-one-grader-out sensitivity (point estimates): is any result driven by one grader?
    sens = []
    for drop in [None] + sorted(long.grader_id.unique()):
        l2 = long if drop is None else long[long.grader_id != drop]
        w2 = claim_consensus(wide_claims(l2, llm))
        unsup = l2[l2.category == "unsupported"]
        sup = l2[l2.category == "supported"]
        pu = pair_table(w2, "llm_human", "unsupported")
        Ru = np.where(w2[["h1_cat", "h2_cat", "h3_cat"]].to_numpy(dtype=object) == "unsupported", w2[["h1_score", "h2_score", "h3_score"]].to_numpy(dtype=float), np.nan)
        sens.append({"excluded_grader": drop or "none", "n_ratings": len(l2),
                     "category_agree_llm_human": (l2.category == l2.llm_cat).mean(),
                     "accuracy_le2_rate_human": (sup.score <= 2).mean(),
                     "plausibility_le2_rate_human": (unsup.score <= 2).mean(),
                     "plausibility_mean_human": unsup.score.mean(),
                     "plausibility_mean_diff_llm_minus_human": (pu.a - pu.b).mean(),
                     "plausibility_exact_agree": (pu.a == pu.b).mean(),
                     "plausibility_alpha_ordinal_humans": kripp_alpha(Ru, "ordinal")})
    out["sensitivity"] = pd.DataFrame(sens)

    # claims flagged by a human majority (score <= 2 in own scale) that the LLM rated 3-4
    three = w[w.n_humans == 3]
    out["flag"] = {"n_claims": len(three), "n_cons_flag": int(three.cons_flag.sum()), "n_llm_flag": int((three.llm_score <= 2).sum()),
                   "n_cons_flag_llm_flag": int(((three.cons_flag == 1) & (three.llm_score <= 2)).sum()),
                   "cons_flag_by_llm_cat": three[three.cons_flag == 1].llm_cat.value_counts().to_dict(),
                   "n_any_human_flag": int((three.n_flag_humans >= 1).sum()), "n_unanimous_flag": int((three.n_flag_humans == 3).sum()),
                   "cons_flag_rate": cluster_boot(three.assign(_f=three.cons_flag.astype(float)), "record_id", lambda d: d._f.mean())}

    # supporting vignette claim, both-supported ratings
    sv = long[(long.category == "supported") & (long.llm_cat == "supported")].copy()
    sv["match"] = (sv.vignette_idx == sv.llm_vidx).astype(float)
    sv = sv[sv.llm_vidx.notna()]
    out["vclaim"] = {"n": len(sv), "match": cluster_boot(sv.assign(_m=sv.match), "record_id", lambda d: d._m.mean())}

    # per-grader
    pg = []
    for gid, g in long.groupby("grader_id"):
        sup, uns = g[g.category == "supported"], g[g.category == "unsupported"]
        pg.append({"grader_id": gid, "n_records": g.record_id.nunique(), "n_claims": len(g),
                   "pct_unsupported": (g.category == "unsupported").mean(), "mean_accuracy": sup.score.mean(), "pct_accuracy_le2": (sup.score <= 2).mean(),
                   "mean_plausibility": uns.score.mean(), "pct_plausibility_le2": (uns.score <= 2).mean(),
                   "category_agree_with_llm": (g.category == g.llm_cat).mean(),
                   "score_exact_agree_when_same_category": (g[g.category == g.llm_cat].score == g[g.category == g.llm_cat].llm_score).mean()})
    out["per_grader"] = pd.DataFrame(pg)

    # conversation-level metrics (the paper's fidelity metrics); humans from claim rows, LLM from the judge json
    def conv_metrics(cat: pd.Series, score: pd.Series) -> dict:
        sup, uns = score[cat == "supported"], score[cat == "unsupported"]
        return {"unsupported_rate": (cat == "unsupported").mean(), "mean_accuracy": sup.mean() if len(sup) else np.nan,
                "mean_plausibility": uns.mean() if len(uns) else np.nan, "confirmed_rate": (sup == 4).mean() if len(sup) else np.nan,
                "contradiction_rate": (sup == 1).mean() if len(sup) else np.nan}
    conv = []
    for rid, rec in out["recs"].items():
        lrows = llm[llm.record_id == rid]
        lm = conv_metrics(lrows.llm_cat, lrows.llm_score)
        # coverage from not-mentioned lists
        nv = len(rec["vignette_claims"])
        vtext = {t: i for i, t in enumerate(rec["vignette_claims"])}
        llm_nm = {vtext[t] for t in rec["llm_vignette_claims_not_mentioned"] if t in vtext}
        lm["coverage_rate"] = 1 - len(llm_nm) / nv
        humans = []
        for gid, g in long[long.record_id == rid].groupby("grader_id"):
            hm = conv_metrics(g.category, g.score)
            srow = summ[(summ.record_id == rid) & (summ.grader_id == gid)]
            if len(srow):
                nm = {int(v[1:]) - 1 for v in srow.not_mentioned_vignette_claims.iloc[0].split(", ") if v}
                hm["coverage_rate"] = 1 - len(nm) / nv
            else:
                hm["coverage_rate"] = np.nan
            hm["grader_id"] = gid
            humans.append(hm)
        hdf = pd.DataFrame(humans)
        for metric in lm:
            conv.append({"record_id": rid, "metric": metric, "llm": lm[metric], "human_mean": hdf[metric].mean(), "n_humans": hdf[metric].notna().sum(),
                         "human_min": hdf[metric].min(), "human_max": hdf[metric].max()})
    conv = pd.DataFrame(conv)
    out["conv"] = conv
    conv_stats = {}
    for metric, d in conv.groupby("metric"):
        d = d.dropna(subset=["llm", "human_mean"]).reset_index(drop=True)
        conv_stats[metric] = {
            "n": len(d), "llm_mean": d.llm.mean(), "human_mean": d.human_mean.mean(),
            "diff": cluster_boot(d.assign(_r=d.record_id, _d=d.llm - d.human_mean), "_r", lambda x: x._d.mean()),
            "spearman": cluster_boot(d, "record_id", lambda x: x.llm.corr(x.human_mean, method="spearman")) if d.llm.nunique() > 1 else (np.nan,) * 3}
    out["conv_stats"] = conv_stats

    # coverage at the vignette-claim level
    cov_rows = []
    for rid, rec in out["recs"].items():
        nv = len(rec["vignette_claims"])
        vtext = {t: i for i, t in enumerate(rec["vignette_claims"])}
        llm_nm = {vtext[t] for t in rec["llm_vignette_claims_not_mentioned"] if t in vtext}
        s = summ[summ.record_id == rid]
        hl = [{int(v[1:]) - 1 for v in x.split(", ") if v} for x in s.not_mentioned_vignette_claims]
        for i in range(nv):
            votes = [i not in h for h in hl]  # True = mentioned
            cov_rows.append({"record_id": rid, "vclaim": i, "n_humans": len(hl), "llm": float(i not in llm_nm),
                             "h1": float(votes[0]), "h2": float(votes[1]) if len(votes) > 1 else np.nan, "h3": float(votes[2]) if len(votes) > 2 else np.nan})
    cov = pd.DataFrame(cov_rows)
    c3 = cov[cov.n_humans == 3].copy()
    c3["cons"] = ((c3[["h1", "h2", "h3"]].sum(axis=1)) >= 2).astype(float)

    def cov_stat(d: pd.DataFrame) -> dict:
        h = d[["h1", "h2", "h3"]].to_numpy()
        llm_v = d.llm.to_numpy()
        return {"agree": (d.llm == d.cons).mean(), "kappa": cohen_kappa(d.llm.astype(int), d.cons.astype(int), [0, 1]),
                "alpha": kripp_alpha(h, "nominal"), "llm_h": np.mean([(h[:, j] == llm_v).mean() for j in range(3)]),
                "llm_h_kappa": llm_human_kappa(llm_v.astype(int), h.astype(int), [0, 1]),
                "h_h": np.mean([(h[:, i] == h[:, j]).mean() for i, j in ((0, 1), (0, 2), (1, 2))]),
                "h_h_kappa": fleiss_kappa(h.astype(int), [0, 1])}
    out["coverage"] = cluster_boot(c3, "record_id", cov_stat)
    out["coverage"]["n_items"] = len(c3)
    out["coverage"]["llm_mentioned_rate"] = c3.llm.mean()
    out["coverage"]["human_consensus_mentioned_rate"] = c3.cons.mean()
    out["cov_items"] = cov
    return out


# ---------------------------------------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------------------------------------

def ci(t, pct: bool = False, digits: int = 2) -> str:
    return fmt(*t, pct=pct, digits=digits)


def count_cell(r: pd.Series, key: str, k_over_n: str) -> str:
    """'k/n (Wilson lo%, hi%)' for a per-parameter row."""
    return f"{k_over_n} ({100 * r[key + '_lo']:.0f}%, {100 * r[key + '_hi']:.0f}%)"


def row_cell(r: pd.Series, key: str, pct: bool = False) -> str:
    """'estimate (bootstrap lo, hi)' for a per-parameter row."""
    return fmt(r[key], r[key + "_lo"], r[key + "_hi"], pct=pct)


def write_outputs(adh: dict, fid: dict) -> None:
    OUT.mkdir(exist_ok=True)
    adh["items"].to_csv(OUT / "adherence_items.csv", index=False)
    adh["per_param"].to_csv(OUT / "adherence_by_parameter.csv", index=False)
    rows = []
    for name, res in adh["pooled"].items():
        for k, v in res.items():
            if isinstance(v, tuple):
                rows.append({"subset": name, "metric": k, "estimate": v[0], "ci_lo": v[1], "ci_hi": v[2], "n_items": res["n_items"]})
    pd.DataFrame(rows).to_csv(OUT / "adherence_pooled.csv", index=False)
    fid["wide"].to_csv(OUT / "fidelity_claim_level.csv", index=False)
    fid["per_grader"].to_csv(OUT / "fidelity_by_grader.csv", index=False)
    fid["sensitivity"].to_csv(OUT / "fidelity_leave_one_grader_out.csv", index=False)
    fid["conv"].to_csv(OUT / "fidelity_conversation_level.csv", index=False)
    rows = [{"metric": m, "n_conversations": s["n"], "llm_mean": s["llm_mean"], "human_mean": s["human_mean"],
             "diff_llm_minus_human": s["diff"][0], "diff_lo": s["diff"][1], "diff_hi": s["diff"][2],
             "spearman": s["spearman"][0], "spearman_lo": s["spearman"][1], "spearman_hi": s["spearman"][2]} for m, s in fid["conv_stats"].items()]
    pd.DataFrame(rows).to_csv(OUT / "fidelity_conversation_summary.csv", index=False)
    # rating-level score summary (all claim ratings, not per conversation): input to the clinical-fidelity figure
    rows = []
    for scale, e in fid["scores"].items():
        for rater, dist, mean, rate in (("human", e["human_dist"], e["human_mean"], e["human_low_rate"]),
                                        ("llm", e["llm_dist"], e["llm_mean_ci"], (e["llm_low_rate"], np.nan, np.nan))):
            rows.append({"scale": scale, "rater": rater, "n_ratings": sum(dist.values()), "mean": mean[0], "mean_lo": mean[1], "mean_hi": mean[2],
                         "pct_rated_1_2": rate[0], **{f"n_score_{s}": dist[s] for s in (1, 2, 3, 4)}})
    pd.DataFrame(rows).to_csv(OUT / "fidelity_score_summary.csv", index=False)
    (OUT / "results_tables.md").write_text(markdown_tables(adh, fid))


def markdown_tables(adh: dict, fid: dict) -> str:
    L = ["# Experiment 1 human validation: tables", "", "Generated by analyze_exp1_validation.py. 95% CIs: percentile cluster bootstrap over records (B=2000);",
         "per-parameter proportions use Wilson intervals. LLM = Claude Opus 4.6 judge; human consensus = majority of 3 graders.", ""]
    L += ["## Table 1. Parameter adherence: LLM judge vs three human graders per record", "",
          "| Parameter | LLM picks activated | Human consensus picks activated | LLM = human consensus | Cohen's kappa, LLM vs consensus | "
          "LLM-human agreement | Cohen's kappa, LLM-human | Human-human agreement | Fleiss' kappa, human-human |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in adh["per_param"][adh["per_param"].n_items >= 20].iterrows():
        label = r.param if r.n_items == 20 else f"{r.param} (n = {int(r.n_items)})"
        L.append(f"| {label} | {count_cell(r, 'llm_acc_vs_design', r.llm_correct)} | {count_cell(r, 'human_consensus_acc_vs_design', r.human_consensus_correct)} | "
                 f"{count_cell(r, 'llm_vs_consensus_agree', r.llm_consensus_agree_k)} | {row_cell(r, 'llm_vs_consensus_kappa')} | "
                 f"{row_cell(r, 'llm_vs_human_pairwise_agree', True)} | {row_cell(r, 'llm_vs_human_pairwise_kappa')} | "
                 f"{row_cell(r, 'human_vs_human_pairwise_agree', True)} | {row_cell(r, 'human_fleiss_kappa')} |")
    a10 = adh["pooled"]["all 10 parameters (goal ties = no pick)"]
    L.append(f"| **All 10 parameters (n = {a10['n_items']})** | {ci(a10['llm_acc_vs_design'], True)} | {ci(a10['human_consensus_acc_vs_design'], True)} | "
             f"{ci(a10['llm_vs_consensus_agree'], True)} | {ci(a10['llm_vs_consensus_kappa'])} | {ci(a10['llm_vs_human_pairwise_agree'], True)} | "
             f"{ci(a10['llm_vs_human_pairwise_kappa'])} | {ci(a10['human_vs_human_pairwise_agree'], True)} | {ci(a10['human_fleiss_kappa'])} |")
    L += ["", "**Legend.**",
          "- *Unit.* One item is one forced choice between the two conversations of a record: one simulated with the parameter activated, one baseline "
          "(on-screen position A/B randomised). Each parameter setting has 20 records (20 items; Health literacy pools its low and high settings, 40 items), "
          "and each record was rated independently by 3 of the 6 graders. 'All 10 parameters' pools all items: 10 parameter settings (Recall and Health "
          "literacy each have a low and a high setting) x 20 records.",
          "- *Picks activated.* Proportion of items in which the rater chose the activated conversation; this is the concordance statistic (c-stat) of "
          "main-text Fig. 3a for a forced choice (chance = 50%). LLM = Claude Opus 4.6 judge. Human consensus = majority pick of the 3 graders; an item "
          f"without a majority pick (all three graders differ; {a10['n_no_majority']} items, only possible for Patient goal) counts as a tie and as a miss.",
          "- *LLM = human consensus.* Proportion of items where the LLM's pick equals the consensus pick.",
          "- *Agreement.* LLM-human: share of (LLM pick, grader pick) pairs that match, over the 3 pairs per item. Human-human: same over the 3 grader pairs per item.",
          "- *Kappa.* Cohen's kappa between the LLM and the consensus; LLM-human kappa is Cohen's kappa pooled over the 3 LLM-grader pairs per item; "
          "human-human kappa is Fleiss' kappa (3 raters per item). Picks are coded by on-screen position (A/B; T = tie), which is randomised independently "
          "of which conversation carries the setting, so chance agreement is about 50% and kappa is defined even when every rater picks the activated conversation.",
          f"- *95% CIs.* Counts (picks activated, LLM = consensus): Wilson score interval. Agreement and kappa, per parameter: percentile bootstrap over "
          f"records (B = {N_BOOT}, whole records resampled; left blank when kappa is undefined in more than 10% of resamples). With only 20 records the bootstrap "
          "collapses to a point when every record agrees (kappa = 1.00 (1.00, 1.00)); that is a limit of the method, not certainty, so read the Wilson interval in the "
          "'LLM = human consensus' column alongside it. 'All 10 parameters' row: percentile cluster bootstrap over records for every column (a record contributes "
          "10 items), which is wider than treating the 200 items as independent.",
          "- *Patient goal.* Graders judged, per conversation, whether the patient conveyed the goal (Yes/No); the pick is the conversation judged Yes when exactly "
          "one was, and Yes/Yes or No/No is a tie (T).", ""]
    p9 = adh["pooled"]["9 forced-choice parameters (excl. goal)"]
    L += [f"Excluding Patient goal (9 forced-choice parameters, n = {p9['n_items']}): LLM picks activated {ci(p9['llm_acc_vs_design'], True)}; "
          f"human consensus {ci(p9['human_consensus_acc_vs_design'], True)}; LLM = human consensus {ci(p9['llm_vs_consensus_agree'], True)}, "
          f"Cohen's kappa {ci(p9['llm_vs_consensus_kappa'])}; LLM-human agreement {ci(p9['llm_vs_human_pairwise_agree'], True)} (kappa {ci(p9['llm_vs_human_pairwise_kappa'])}); "
          f"human-human agreement {ci(p9['human_vs_human_pairwise_agree'], True)} (Fleiss' kappa {ci(p9['human_fleiss_kappa'])}, Krippendorff's alpha {ci(p9['human_alpha'])}); "
          f"LLM-human minus human-human pairwise agreement {ci(p9['pairwise_diff_llm_minus_hh'], True)}.", ""]
    g = adh["goal"]
    L += [f"Patient goal (Yes/No per conversation, {g['n_ratings']} ratings): the activated conversation was rated as conveying the goal in {g['active_yes']}/{g['n_ratings']}, "
          f"the baseline in {g['neutral_yes']}/{g['n_ratings']}; {g['tie_both_no']} No/No and {g['tie_both_yes']} Yes/Yes ratings gave no unique pick. "
          f"Human consensus: activated conversation conveys the goal in {g['consensus_active_yes']}/{g['n_records']}, baseline in {g['consensus_neutral_yes']}/{g['n_records']}. "
          f"Krippendorff's alpha on the per-conversation Yes/No = {g['yesno_alpha']:.2f}.", ""]
    part = adh["per_param"][adh["per_param"].n_items < 20]
    if len(part):
        L.append("Punctuation and capitalisation (rule-based; graders were later not asked): " + "; ".join(
            f"{r.param}: {int(r.n_items)} records with all 3 graders, LLM {r.llm_correct}, human consensus {r.human_consensus_correct}" for _, r in part.iterrows()) + ".")
        L.append("")
    tc = adh["typo_comments"]
    L += [f"Typos: {tc['n_with_comment']}/{tc['n_ratings']} typo ratings carry a comment, all saying no (or hardly any) typos could be seen, e.g. choosing at random; "
          "the forced choice had no 'cannot tell' option.", ""]

    c = fid["category"]
    L += ["## Table 2. Clinical fidelity: LLM judge vs human graders", "",
          "| Measure | LLM vs human | Human vs human |", "|---|---|---|",
          f"| Claim category, supported vs unsupported (n={c['n_claims']} claims): pairwise agreement | {ci(c['llm_vs_human_pairwise_agree'], True)} | {ci(c['human_vs_human_pairwise_agree'], True)} |",
          f"| &nbsp;&nbsp;Cohen's kappa, LLM-human pairs pooled / Fleiss' kappa, humans | {ci(c['llm_vs_human_pairwise_kappa'])} | {ci(c['human_fleiss_kappa'])} |",
          f"| &nbsp;&nbsp;LLM vs human-consensus agreement / Cohen's kappa | {ci(c['llm_vs_consensus_agree'], True)} / {ci(c['llm_vs_consensus_kappa'])} | Krippendorff's alpha {ci(c['human_alpha'])} |"]
    for scale, e in fid["scores"].items():
        L += [f"| {scale.capitalize()} score, exact agreement (pairs with same category; n={e['n_llm_human_pairs']} / {e['n_human_pairs']}) | {ci(e['lh_exact'], True)} | {ci(e['hh_exact'], True)} |",
              f"| &nbsp;&nbsp;within 1 point | {ci(e['lh_within1'], True)} | {ci(e['hh_within1'], True)} |",
              f"| &nbsp;&nbsp;quadratic-weighted kappa | {ci(e['lh_wkappa'])} | {ci(e['hh_wkappa'])} (human alpha, ordinal: {ci(e['human_alpha_ordinal'])}) |",
              f"| &nbsp;&nbsp;Kendall's tau-b (discrimination) | {ci(e['lh_kendall'])} | {ci(e['hh_kendall'])} |",
              f"| &nbsp;&nbsp;mean score difference, LLM minus human (calibration) | {ci(e['lh_mean_diff'])} | n/a (no direction) |"]
    L.append(f"| Supporting vignette claim, same claim chosen (n={fid['vclaim']['n']} ratings) | {ci(fid['vclaim']['match'], True)} | n/a |")
    cv = fid["coverage"]
    L += [f"| Vignette claim mentioned vs not (n={cv['n_items']} vignette claims): pairwise agreement | {ci(cv['llm_h'], True)} | {ci(cv['h_h'], True)} |",
          f"| &nbsp;&nbsp;Cohen's kappa, LLM-human pairs pooled / Fleiss' kappa, humans | {ci(cv['llm_h_kappa'])} | {ci(cv['h_h_kappa'])} |",
          f"| &nbsp;&nbsp;LLM vs human-consensus agreement / Cohen's kappa | {ci(cv['agree'], True)} / {ci(cv['kappa'])} | Krippendorff's alpha {ci(cv['alpha'])} |", ""]
    L += ["**Legend.**",
          "- *Claims.* Every claim in the simulated patient's conversation was labelled by the LLM and by each grader as supported (by the vignette) or unsupported, "
          "then scored 1-4: accuracy for supported claims, plausibility for unsupported claims. 20 conversations, 3 graders per conversation.",
          f"- *Sample.* Claim-category rows use the {c['n_claims']} claims rated by all 3 graders; {c['n_two_raters']} claims with only 2 graders are left out of the pairwise "
          "and consensus statistics but kept in Krippendorff's alpha. DS-04's F07 submission is excluded (stale duplicate, see KNOWN_BAD_FIDELITY). "
          "Human consensus on the category = majority of the 3 graders (always defined for two categories).",
          "- *Agreement and kappa.* Pairwise agreement: LLM-human = share of (LLM, grader) pairs with the same label, over the 3 pairs per claim; human-human = same over the 3 "
          "grader pairs. Cohen's kappa is pooled over the LLM-grader pairs; Fleiss' kappa is among the 3 graders; Krippendorff's alpha (nominal, or ordinal for scores) is among "
          "the graders only (for the claim category it uses every claim, including those with 2 graders).",
          "- *Scores.* Accuracy / plausibility rows use rating pairs where both raters gave the claim the same category (n = LLM-human pairs / human-human pairs). "
          "Exact = identical 1-4 score; within 1 = scores differ by at most 1; quadratic-weighted kappa treats the 1-4 scale as ordinal. "
          "Kendall's tau-b is the tie-corrected rank correlation of the paired scores (probability two claims are ordered the same way minus the probability they are ordered oppositely): it measures discrimination and, unlike exact agreement and kappa, ignores a constant shift between raters. "
          "The mean difference (LLM minus human) measures that shift (calibration); it has no direction for human-human pairs.",
          "- *Supporting vignette claim.* Share of supported-supported rating pairs in which the grader and the LLM point at the same vignette claim. "
          "*Mentioned vs not.* Per vignette claim, whether the conversation mentions it (conversations with all 3 graders' summaries); LLM vs the majority of the 3 graders.",
          f"- *95% CIs.* Percentile cluster bootstrap over records (B = {N_BOOT}; whole conversations resampled, because claims within a conversation are not independent).", ""]
    L += ["### Score distributions (all ratings)", "", "| Scale | Rater | 1 | 2 | 3 | 4 | Mean | Rated 1-2 |", "|---|---|---|---|---|---|---|---|"]
    for scale, e in fid["scores"].items():
        hd, ld = e["human_dist"], e["llm_dist"]
        L += [f"| {scale} | Humans (n={sum(hd.values())}) | {hd[1]} | {hd[2]} | {hd[3]} | {hd[4]} | {ci(e['human_mean'])} | {ci(e['human_low_rate'], True)} |",
              f"| {scale} | LLM (n={sum(ld.values())}) | {ld[1]} | {ld[2]} | {ld[3]} | {ld[4]} | {ci(e['llm_mean_ci'])} | {e['llm_low_rate']:.1%} |"]
    f = fid["flag"]
    L += ["", f"Claims rated 1-2 (distorted/contradicted or questionable/implausible) by at least 2 of 3 graders: {f['n_cons_flag']}/{f['n_claims']} "
          f"({ci(f['cons_flag_rate'], True)}); the LLM rated {f['n_llm_flag']} of the {f['n_claims']} claims 1-2. "
          f"{f['n_any_human_flag']} claims were rated 1-2 by at least one grader; none by all three. By LLM category the {f['n_cons_flag']} are {f['cons_flag_by_llm_cat']}.", ""]
    L += ["## Table 3. Conversation-level fidelity metrics (20 conversations; human = mean over that conversation's graders)", "",
          "| Metric | LLM mean | Human mean | LLM minus human (95% CI) | Spearman rho across conversations (95% CI) |", "|---|---|---|---|---|"]
    order = ["unsupported_rate", "mean_accuracy", "confirmed_rate", "contradiction_rate", "mean_plausibility", "coverage_rate"]
    for m in order:
        s = fid["conv_stats"][m]
        pct = m.endswith("rate")
        L.append(f"| {m} | {s['llm_mean']:.3f} | {s['human_mean']:.3f} | {ci(s['diff'], pct, 3)} | {ci(s['spearman'])} |")
    L += ["", "**Legend.**",
          "- *Per-conversation values.* The LLM value is computed over the conversation's claims; the human value is the mean over the conversation's graders (2-3) of "
          "each grader's value. unsupported_rate = share of claims labelled unsupported; mean_accuracy = mean 1-4 score of supported claims; confirmed_rate = share of "
          "supported claims scored 4; contradiction_rate = share scored 1; mean_plausibility = mean 1-4 score of unsupported claims; coverage_rate = share of vignette claims "
          "the conversation mentions. Means are over the 20 conversations (conversations without the relevant claims are dropped for that metric).",
          f"- *LLM minus human.* Mean paired difference over conversations; percentile bootstrap over conversations (B = {N_BOOT}). Spearman rho between the LLM and human "
          "per-conversation values with the same bootstrap; n/a when the LLM value is constant. Kappa is not defined for these continuous per-conversation metrics; "
          "item-level kappas are in Table 2.", ""]
    L += ["", "## Per-grader fidelity ratings", "", fid["per_grader"].round(3).to_markdown(index=False) if hasattr(pd.DataFrame, "to_markdown") and _has_tabulate() else fid["per_grader"].round(3).to_string(index=False), "",
          "## Leave-one-grader-out (point estimates)", "", fid["sensitivity"].round(3).to_string(index=False), ""]
    return "\n".join(L)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def print_report(adh: dict, fid: dict) -> None:
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
    print((OUT / "results_tables.md").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--split-health-literacy", action="store_true",
                        help="show Health literacy (low) and (high) as two rows instead of one pooled row")
    args = parser.parse_args()
    wb = load_workbook(args.workbook)
    adh = run_adherence(wb["parameter_adherence"], merge_hl=not args.split_health_literacy)
    fid = run_fidelity(wb["fidelity_claims"], wb["fidelity_summary"])
    write_outputs(adh, fid)
    print_report(adh, fid)


if __name__ == "__main__":
    main()
