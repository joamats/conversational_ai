"""Experiment 1 human-validation figures: Claude Opus 4.6 judge vs human graders.

Same palette and layout as Figure 3 of the paper, with the LLM and the humans side by side:

  (a) Parameter adherence: per parameter, human consensus (solid) and LLM judge (hatched) in the same colour,
      with 95% CIs and a pooled "All 10 parameters" row.
  (b) Clinical fidelity: supported (accuracy) and unsupported (plausibility) claims; LLM (hatched blue) and
      human (solid red) densities overlaid in each panel.

Numbers are read from the CSVs written by analyze_exp1_validation.py, not recomputed here.

    python exp1_validation_figures.py     # reads results/, writes results/figures/

Needs pandas, numpy, scipy, matplotlib.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# ── Palette: as in Fig. 3 of the paper ──────────────────────────────────────────────────────────────────────────
GROUP_ORDER = ["communication_params", "persona_params", "content_params", "strategy_params"]  # bottom -> top, as in Fig. 3a
GROUP_COLORS = {"communication_params": "#3498db", "persona_params": "#e74c3c",
                "content_params": "#57bb8a", "strategy_params": "#9b59b6"}
GROUP_LABELS = {"communication_params": "Communication", "persona_params": "Feelings",
                "content_params": "Content", "strategy_params": "Strategy"}
POOLED_COLOR = "#4d4d4d"
LLM_COLOR, HUMAN_COLOR = "#2e86c1", "#e74c3c"  # Claude Opus 4.6 / Human, as in Fig. 3c
INK, MUTED = "0.2", "0.3"
HATCH = "////"  # texture that marks the LLM everywhere (bars, legends, density fills); override with --hatch, e.g. "xxxx"

PARAM_GROUP = {
    "Anxiety": "persona_params", "Embarrassment": "persona_params",
    "Recall (low)": "content_params", "Recall (high)": "content_params", "Oversharing": "content_params",
    "Health literacy (low + high)": "communication_params", "Health literacy (low)": "communication_params",
    "Health literacy (high)": "communication_params", "English proficiency (non-proficient)": "communication_params",
    "Typos": "communication_params", "Patient goal": "strategy_params",
}
POOLED_SUBSET = "all 10 parameters (goal ties = no pick)"


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.fill(p, width) for p in text.split("\n"))


# ── (a) parameter adherence ───────────────────────────────────────────────────────────────────────────────────────

def load_adherence(results: Path) -> list[dict]:
    per_param = pd.read_csv(results / "adherence_by_parameter.csv")
    per_param = per_param[per_param.n_items >= 20]  # drops punctuation / capitalisation (graders were not asked later)
    rows = []
    for _, r in per_param.iterrows():
        rows.append({"label": r.param, "group": PARAM_GROUP[r.param], "n": int(r.n_items),
                     "human": r.human_consensus_acc_vs_design, "human_lo": r.human_consensus_acc_vs_design_lo, "human_hi": r.human_consensus_acc_vs_design_hi,
                     "llm": r.llm_acc_vs_design, "llm_lo": r.llm_acc_vs_design_lo, "llm_hi": r.llm_acc_vs_design_hi})
    rows.sort(key=lambda r: (GROUP_ORDER.index(r["group"]), r["human"], r["llm"]))  # ascending inside a group, as Fig. 3a
    pooled = pd.read_csv(results / "adherence_pooled.csv")
    pooled = pooled[pooled.subset == POOLED_SUBSET].set_index("metric")
    n = int(pooled.n_items.iloc[0])
    pooled_row = {"label": f"All 10 parameters (n = {n})", "group": "pooled", "n": n}
    for who, metric in (("human", "human_consensus_acc_vs_design"), ("llm", "llm_acc_vs_design")):
        pooled_row.update({who: pooled.loc[metric, "estimate"], f"{who}_lo": pooled.loc[metric, "ci_lo"], f"{who}_hi": pooled.loc[metric, "ci_hi"]})
    return rows + [pooled_row]


def draw_adherence(ax, rows: list[dict]) -> None:
    pooled = rows[-1]
    rows = rows[:-1]
    y, cur, prev = [], 0.0, None
    for r in rows:
        if prev is not None and r["group"] != prev:
            cur += 0.5  # gap between parameter types, as Fig. 3a
        y.append(cur)
        cur += 1
        prev = r["group"]
    y = np.array(y + [-1.7])  # pooled row sits below the first parameter row
    allrows = rows + [pooled]
    h = 0.38
    for yy, r in zip(y, allrows):
        color = POOLED_COLOR if r["group"] == "pooled" else GROUP_COLORS[r["group"]]
        for who, off, hatch in (("human", +h / 2 + 0.01, None), ("llm", -h / 2 - 0.01, HATCH)):
            v, lo, hi = r[who], r[f"{who}_lo"], r[f"{who}_hi"]
            if hatch:
                ax.barh(yy + off, v, height=h, facecolor=color, edgecolor="white", hatch=hatch, linewidth=0, zorder=3)
            else:
                ax.barh(yy + off, v, height=h, facecolor=color, edgecolor="none", zorder=3)
            ax.errorbar(v, yy + off, xerr=[[max(v - lo, 0)], [max(hi - v, 0)]], fmt="none", ecolor="black", elinewidth=0.8, capsize=2, zorder=4)
            ax.text(hi + 0.012, yy + off, f"{v:.2f}", va="center", ha="left", fontsize=7.5, color=MUTED)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, zorder=2)
    ax.text(0.5, -2.62, "Chance", fontsize=8, color="grey", ha="center", va="bottom", zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    ax.axhline(-0.85, color="0.8", linewidth=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] if r["n"] == 20 or r["group"] == "pooled" else f"{r['label']} (n = {r['n']})" for r in allrows], fontsize=9)
    ax.set_xlim(0.0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Concordance statistic (c-stat)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.set_ylim(-2.7, y[len(rows) - 1] + 2.3)  # headroom at the top holds the rater legend


def adherence_legends(ax) -> None:
    types = [Patch(facecolor=GROUP_COLORS[g], label=GROUP_LABELS[g]) for g in GROUP_ORDER] + [Patch(facecolor=POOLED_COLOR, label="Pooled")]
    leg = ax.legend(handles=types, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=5, framealpha=0.9,
                    title="Parameter Type", title_fontproperties={"size": 8.5})
    ax.add_artist(leg)
    raters = [Patch(facecolor="0.45", edgecolor="none", label="Human consensus (majority of 3 graders)"),
              Patch(facecolor="0.45", edgecolor="white", hatch=HATCH, linewidth=0, label="LLM judge (Claude Opus 4.6)")]
    ax.legend(handles=raters, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, framealpha=0.9,
              title="Rater", title_fontproperties={"size": 8.5}, borderaxespad=0.3)


ADHERENCE_NOTE = (
    "Bars: proportion of items in which the rater chose the conversation simulated with the parameter activated "
    "(the c-stat of Fig. 3a for a forced choice; chance = 0.5). Human consensus = majority pick of 3 graders per record "
    "(no majority counts as a miss). n = 20 records per parameter setting (Health literacy pools low and high: 40); "
    "'All 10 parameters' pools all 200 items. Whiskers: 95% CI, Wilson interval per parameter; percentile cluster bootstrap "
    "over records (B = 2000) for the pooled row. Patient goal: the pick is the conversation graders judged to convey the goal.")


def figure_adherence(results: Path) -> plt.Figure:
    fig = plt.figure(figsize=(8.6, 8.0))
    gs = gridspec.GridSpec(1, 1, left=0.34, right=0.97, top=0.88, bottom=0.25)
    ax = fig.add_subplot(gs[0])
    draw_adherence(ax, load_adherence(results))
    adherence_legends(ax)
    fig.text(0.02, 0.97, "(a) Parameter Adherence,\n     Human Graders vs Claude Opus 4.6 as a Judge", fontsize=12, va="top", ha="left")
    fig.text(0.02, 0.012, wrap(ADHERENCE_NOTE, 135), fontsize=7, color=MUTED, va="bottom", ha="left")
    return fig


# ── (b) clinical fidelity ─────────────────────────────────────────────────────────────────────────────────────────

PANELS = [  # metric in fidelity_conversation_level.csv, scale in fidelity_score_summary.csv, axis range, labels
    {"metric": "mean_accuracy", "scale": "accuracy", "xlim": (3.0, 4.0), "xlabel": "Accuracy (1–4)", "title": "Supported claims"},
    {"metric": "mean_plausibility", "scale": "plausibility", "xlim": (2.0, 4.0), "xlabel": "Plausibility (1–4)", "title": "Unsupported claims"},
]


def kde_pair(llm: np.ndarray, hum: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Densities with the same absolute bandwidth for both raters (0.3 x the pooled SD, the 0.3 of Fig. 3b), so that
    differences in shape are not an artefact of each rater's own spread."""
    bw = 0.3 * np.concatenate([llm, hum]).std(ddof=1)
    return tuple(gaussian_kde(v, bw_method=bw / v.std(ddof=1))(grid) for v in (llm, hum))


def draw_fidelity_panel(ax, spec: dict, conv: pd.DataFrame, summ: pd.DataFrame) -> None:
    d = conv[conv.metric == spec["metric"]]
    llm, hum = d.llm.dropna().to_numpy(), d.human_mean.dropna().to_numpy()
    grid = np.linspace(*spec["xlim"], 400)
    k_llm, k_hum = kde_pair(llm, hum, grid)
    s = summ[summ.scale == spec["scale"]].set_index("rater")
    top = max(k_llm.max(), k_hum.max())

    ax.fill_between(grid, k_hum, alpha=0.30, color=HUMAN_COLOR, linewidth=0, zorder=3)
    ax.plot(grid, k_hum, color=HUMAN_COLOR, linewidth=1.5, zorder=4)
    ax.fill_between(grid, k_llm, color=LLM_COLOR, alpha=0.10, linewidth=0, zorder=3)
    ax.fill_between(grid, k_llm, facecolor="none", edgecolor=(*to_rgb(LLM_COLOR), 0.65), hatch=HATCH, linewidth=0, zorder=3)
    ax.plot(grid, k_llm, color=LLM_COLOR, linewidth=1.5, zorder=4)

    line_top = 0.68  # axes fraction; the legend lives above this
    for who, color, style in (("human", HUMAN_COLOR, "-"), ("llm", LLM_COLOR, "--")):
        ax.axvline(s.loc[who, "mean"], ymax=line_top, color=color, linestyle=style, linewidth=1.5, zorder=5)
        ax.annotate("Human" if who == "human" else "LLM", xy=(s.loc[who, "mean"], line_top), xycoords=("data", "axes fraction"), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    tr = ax.get_xaxis_transform()  # rug of the individual conversations, inside the axes
    ax.plot(hum, np.full(len(hum), 0.025), "|", color=HUMAN_COLOR, ms=7, alpha=0.7, transform=tr, zorder=3)
    ax.plot(llm, np.full(len(llm), 0.075), "|", color=LLM_COLOR, ms=7, alpha=0.7, transform=tr, zorder=3)

    def lab(who: str, name: str, unit: str) -> str:
        r = s.loc[who]
        return f"{name} ({int(r.n_ratings)} {unit}): mean {r['mean']:.2f} ({r.mean_lo:.2f}, {r.mean_hi:.2f})"
    handles = [Patch(facecolor=(*to_rgb(HUMAN_COLOR), 0.30), edgecolor=HUMAN_COLOR, linewidth=1.2, label=lab("human", "Human graders", "ratings")),
               Patch(facecolor=(*to_rgb(LLM_COLOR), 0.10), edgecolor=LLM_COLOR, hatch=HATCH, linewidth=1.2, label=lab("llm", "LLM judge", "claims"))]
    ax.legend(handles=handles, fontsize=7.8, loc="upper left", frameon=False, borderaxespad=0.2, handlelength=1.8)

    ax.set_xlim(*spec["xlim"])
    ax.set_ylim(0, top * 1.9)  # headroom for the in-axes legend and mean labels
    ax.set_ylabel("Density", fontsize=9)
    ax.set_xlabel(spec["xlabel"], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_yticks([])
    ax.set_title(spec["title"], fontsize=10, loc="left", pad=8)


FIDELITY_NOTE = (
    "Curves: kernel density of each conversation's mean score (20 conversations; a conversation's human value is the mean of its 2–3 graders); "
    "same kernel bandwidth for LLM and humans; ticks mark the individual conversations. Vertical lines and legend: mean over all ratings with "
    "95% CI (percentile cluster bootstrap over conversations, B = 2000). Accuracy scores supported claims and plausibility scores unsupported "
    "claims (both 1–4); a claim is supported or unsupported according to each rater's own label. Densities are cut at the top of the scale (4.0).")


def figure_fidelity(results: Path) -> plt.Figure:
    conv = pd.read_csv(results / "fidelity_conversation_level.csv")
    summ = pd.read_csv(results / "fidelity_score_summary.csv")
    fig = plt.figure(figsize=(6.6, 7.6))
    gs = gridspec.GridSpec(2, 1, left=0.10, right=0.96, top=0.86, bottom=0.17, hspace=0.45)
    for i, spec in enumerate(PANELS):
        draw_fidelity_panel(fig.add_subplot(gs[i]), spec, conv, summ)
    fig.text(0.02, 0.97, "(b) Clinical Fidelity,\n     Human Graders vs Claude Opus 4.6 as a Judge", fontsize=12, va="top", ha="left")
    fig.text(0.02, 0.012, wrap(FIDELITY_NOTE, 105), fontsize=7, color=MUTED, va="bottom", ha="left")
    return fig


def main() -> None:
    global HATCH
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=RESULTS, help="results directory written by analyze_exp1_validation.py")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: <results>/figures)")
    parser.add_argument("--hatch", default=HATCH, help="matplotlib hatch pattern for the LLM (e.g. '////', 'xxxx', '----', '....')")
    args = parser.parse_args()
    HATCH = args.hatch
    out = args.out or args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for name, make in (("exp1_adherence_llm_vs_human", figure_adherence), ("exp1_fidelity_llm_vs_human", figure_fidelity)):
        fig = make(args.results)
        fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out / name}.png and .pdf")


if __name__ == "__main__":
    main()
