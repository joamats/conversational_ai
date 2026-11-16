# Experiment 2: clinical fidelity across personas

Re-evaluates conversations already produced by the persona fairness experiment (`python -m fairness.run`, doctor model Gemini 3.5 Flash) with the clinical-fidelity protocol of Experiment 1 (`eval/fidelity/`, Claude Opus 4.6 judge). No new conversations are simulated.

- Samples 150 source cases, preferring cases where the five personas' predicted triage disagreed.
- Evaluates all five persona conversations of each case (baseline and four personas) against the same source vignette.
- With `shared_vignette_claims` (on by default), each vignette's claims are extracted once and all five conversations of the case are classified against that list, so coverage has a common denominator within a case.
- Reports fidelity per persona, the within-case difference from baseline (paired t-test, Wilcoxon, Holm-adjusted p-values, 95% CIs), and the spread across personas within each case.

The script uses `eval/` and `fairness/` from this repository. `shared_claims.py` adds the shared-vignette-claims step (a vignette-only extraction and an `evaluate_conversation` that accepts a precomputed claim list) without changing `eval/`.

## Run

From the repository root:

```bash
python -m revisions.exp2.run_persona_experiment \
    --conversations-dir fairness/results/runs/<run_id>/conversations \
    --fairness-config fairness/configs/config_gemini.json

# resume or extend an existing run
python -m revisions.exp2.run_persona_experiment --evaluate-only --run-id <run_id>
```

`persona_config.json` holds the defaults (150 cases, seed 42, judge model, `shared_vignette_claims`); command-line options override it. Relative paths are resolved from the repository root.

Requirements:

- The conversations from a fairness run (`python -m fairness.run`); they are not included here.
- The input datasets referenced by the fairness config, under `data/` (used only to look up each case's vignette).
- Claude on Vertex AI via application-default credentials, as described in the repository README.

Output is written to `revisions/exp2/results/persona_runs/<run_id>/`: `run_meta.json`, `run.log`, `case_selection.json`, `vignette_claims.json` (shared claims), and `fidelity/` with one JSON per conversation, `metrics.json` and the `table_*.csv` files (within-case deltas with statistics, and the 1–4 rubric scores with bootstrap CIs).
