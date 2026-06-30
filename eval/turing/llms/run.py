"""CLI entry point for the realism equivalence trial.

Usage:
    # 1. Build JSONL data from an existing eval run
    python -m eval.turing.llms.run --build-data

    # 2. Launch Gradio UI for human raters
    python -m eval.turing.llms.run --ui

    # 3. Run LLM judge (no UI)
    python -m eval.turing.llms.run --llm-judge

    # 4. Extract simulation params from real conversations
    python -m eval.turing.llms.run --extract-params

    # 5. Normalize simulated conversations to turing format
    python -m eval.turing.llms.run --normalize

    # Options
    --n-simulated 5         Number of simulated conversations per test
    --n-real 5              Number of real conversations per test
    --model gpt-5.4-...     LLM judge model
    --seed 42               Random seed for sampling
    --run-dir <path>        Eval run directory for --build-data
    --simulated <path>      Path to simulated.jsonl
    --real <path>           Path to real.jsonl
    --data <path>           Path to real_clean.csv (for --extract-params / --build-real / --pipeline)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parents[3]
_EVAL_DIR = Path(__file__).parents[1]


def _find_latest_run() -> Path | None:
    results_base = _EVAL_DIR / "results" / "runs"
    if not results_base.exists():
        return None
    runs = sorted(results_base.iterdir(), reverse=True)
    for r in runs:
        if (r / "conversations").exists():
            return r
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Realism equivalence trial")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-data", action="store_true", help="Extract JSONL data from an eval run")
    mode.add_argument("--ui", action="store_true", help="Launch Gradio UI for human raters")
    mode.add_argument("--llm-judge", action="store_true", help="Run LLM judge (no UI)")
    mode.add_argument("--extract-params", action="store_true", help="Extract sim params from real conversations")
    mode.add_argument("--normalize", action="store_true", help="Normalize simulated conversations to turing format")
    mode.add_argument("--pipeline", action="store_true", help="Run preprocessing pipeline: build-real -> extract -> normalize")
    mode.add_argument("--build-real", action="store_true", help="Build real.jsonl from real_clean.csv")
    mode.add_argument("--turing-judge", action="store_true", help="Run 3-mode Turing test judge (absolute + comparative)")
    mode.add_argument("--multi-judge", action="store_true",
                      help="Run Turing judge with 3 models in parallel (GPT-5.5 + Claude Opus 4.7 + Gemini 3.1)")
    mode.add_argument("--build-splits", action="store_true", help="Build train/validation/test splits.json")

    parser.add_argument("--run-dir", type=str, default=None, help="Eval run directory (for --build-data)")
    parser.add_argument("--n-per-file", type=int, default=10, help="Conversations per JSONL file (for --build-data)")
    parser.add_argument("--n-simulated", type=int, default=5, help="Simulated conversations per test")
    parser.add_argument("--n-real", type=int, default=5, help="Real conversations per test")
    parser.add_argument("--n-pairs", type=int, default=10, help="Number of matched pairs for --turing-judge")
    parser.add_argument("--judge-mode", type=str, nargs="+", default=None,
                        choices=["absolute", "comparative_forced", "comparative_unsure", "all"],
                        help="Which judge mode(s) to run (default: all). E.g. --judge-mode absolute")
    parser.add_argument("--split", type=str, default=None,
                        help="Run only on session IDs in this split (from splits.json). Overrides --n-pairs.")
    parser.add_argument("--model", type=str, default="gpt-5.5-2026-04-23", help="LLM model")
    parser.add_argument("--reasoning-effort", type=str, default="xhigh")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers for LLM judge / extraction")
    parser.add_argument("--simulated", type=str, default=None, help="Path to simulated.jsonl")
    parser.add_argument("--real", type=str, default=None, help="Path to real.jsonl")
    parser.add_argument("--data", type=str, default=None, help="Path to real_clean.csv (for --extract-params / --build-real / --pipeline)")
    parser.add_argument("--skip-build-real", action="store_true", help="Skip building real.jsonl in --pipeline, reuse existing")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction in --pipeline, reuse existing")
    parser.add_argument("--judges", nargs="*", default=None,
                        help="Only run these judges (e.g. --judges claude-opus-4.6). Default: all in JUDGE_CONFIGS")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")

    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-8s %(message)s")

    if args.build_data:
        from eval.turing.preprocessing.data import build_initial_data

        run_dir = args.run_dir
        if not run_dir:
            latest = _find_latest_run()
            if not latest:
                print("No eval runs found. Provide --run-dir.", file=sys.stderr)
                sys.exit(1)
            run_dir = str(latest)
            print(f"Using latest eval run: {run_dir}")

        sim_path, real_path = build_initial_data(run_dir, n_per_file=args.n_per_file)
        print(f"Created {sim_path} and {real_path}")
        return

    sim_path = args.simulated
    real_path = args.real

    if args.ui:
        import gradio as gr
        from eval.turing.humans.app import (
            build_app,
            _get_root_path,
            _STATIC_DIR,
        )

        app = build_app()
        app.launch(
            share=args.share,
            root_path=_get_root_path(),
            ssr_mode=False,
            theme=gr.themes.Soft(),
            allowed_paths=[str(_STATIC_DIR)],
        )
        return

    if args.llm_judge:
        from eval.turing.preprocessing.data import (
            DEFAULT_REAL_PATH,
            DEFAULT_SIMULATED_PATH,
        )
        from eval.turing.llms.llm_judge import run_llm_judge

        session = run_llm_judge(
            simulated_path=sim_path or DEFAULT_SIMULATED_PATH,
            real_path=real_path or DEFAULT_REAL_PATH,
            n_simulated=args.n_simulated,
            n_real=args.n_real,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            seed=args.seed,
            max_workers=args.max_workers,
        )

        report = session["report"]
        c = report["confusion"]
        print(f"\nRealism Trial Results (judge: {session['rater']})")
        print(f"Accuracy: {report['accuracy']:.0%} ({report['correct']}/{report['total']})")
        print(f"Confusion: TP={c['tp']} TN={c['tn']} FP={c['fp']} FN={c['fn']}")
        print()
        for r in report["per_item"]:
            icon = "+" if r["correct"] else "x"
            print(f"  [{icon}] {r['conversation_id']}: truth={r['ground_truth']} "
                  f"classified={r['classification']} conf={r.get('confidence', '-')}")
            if r.get("feedback"):
                print(f"      {r['feedback'][:120]}")
        return

    if args.extract_params:
        from eval.turing.preprocessing.extract import (
            DEFAULT_CSV_PATH,
            DEFAULT_PARAMS_OUTPUT as DEFAULT_OUTPUT_PATH,
            run_extraction,
        )

        results = run_extraction(
            csv_path=args.data or DEFAULT_CSV_PATH,
            output_path=DEFAULT_OUTPUT_PATH,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_workers=args.max_workers,
        )
        n_ok = sum(1 for r in results if "error" not in r)
        print(f"\nExtracted params: {n_ok}/{len(results)} successful")
        print(f"Output: {DEFAULT_OUTPUT_PATH}")
        return

    if args.normalize:
        from eval.turing.preprocessing.normalize import (
            DEFAULT_OUTPUT_PATH as NORM_OUTPUT_PATH,
            DEFAULT_RAW_PATH,
            run_normalization,
        )

        results = run_normalization(
            raw_path=DEFAULT_RAW_PATH,
            output_path=NORM_OUTPUT_PATH,
        )
        print(f"\nNormalized: {len(results)} conversations")
        print(f"Output: {NORM_OUTPUT_PATH}")
        return

    if args.pipeline:
        from eval.turing.preprocessing.extract import (
            DEFAULT_CSV_PATH,
        )
        from eval.turing.preprocessing.pipeline import (
            run_pipeline,
        )

        summary = run_pipeline(
            csv_path=args.data or DEFAULT_CSV_PATH,
            extract_model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_extract_workers=args.max_workers,
            skip_build_real=args.skip_build_real,
            skip_extract=args.skip_extract,
        )
        print(f"\nPipeline complete in {summary['duration_seconds']}s")
        for stage in ("build_real", "extract", "normalize"):
            info = summary.get(stage, {})
            if info.get("skipped"):
                print(f"  {stage}: skipped")
            else:
                total = info.get("total", "?")
                success = info.get("success", total)
                path = info.get("path", "")
                if path:
                    print(f"  {stage}: {success}/{total} -> {path}")
                else:
                    print(f"  {stage}: {total}")
        return

    if args.turing_judge:
        from eval.turing.preprocessing.data import (
            DEFAULT_REAL_PATH,
            DEFAULT_SIMULATED_PATH,
        )
        from eval.turing.llms.llm_judge import VALID_MODES, run_turing_judge

        judge_modes = None
        if args.judge_mode and "all" not in args.judge_mode:
            judge_modes = set(args.judge_mode)

        session = run_turing_judge(
            simulated_path=sim_path or DEFAULT_SIMULATED_PATH,
            real_path=real_path or DEFAULT_REAL_PATH,
            n_pairs=args.n_pairs,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            seed=args.seed or 42,
            max_workers=args.max_workers,
            modes=judge_modes,
            split=args.split,
        )

        print(f"\n=== Turing Judge Results ({session['model']}) ===")

        if "absolute" in session:
            abs_r = session["absolute"]
            print(f"\nMode 1 — Absolute:")
            print(f"  Accuracy: {abs_r['accuracy']:.0%} ({abs_r['correct']}/{abs_r['total']})")
            c = abs_r["confusion"]
            print(f"  Confusion: TP={c['tp']} TN={c['tn']} FP={c['fp']} FN={c['fn']}")
            print(f"  Avg Likert (real): {abs_r['avg_likert_real']:.2f}  (sim): {abs_r['avg_likert_sim']:.2f}")

        if "comparative_forced" in session:
            cf_r = session["comparative_forced"]
            print(f"\nMode 2 — Comparative (forced):")
            print(f"  Accuracy: {cf_r['accuracy_decided']:.0%} ({cf_r['correct']}/{cf_r['decided']})")

        if "comparative_unsure" in session:
            cu_r = session["comparative_unsure"]
            print(f"\nMode 3 — Comparative (unsure allowed):")
            print(f"  Accuracy (decided): {cu_r['accuracy_decided']:.0%} ({cu_r['correct']}/{cu_r['decided']})")
            print(f"  Unsure: {cu_r['unsure']}/{cu_r['total']}")
        return

    if args.multi_judge:
        from eval.turing.preprocessing.data import (
            DEFAULT_REAL_PATH,
            DEFAULT_SIMULATED_PATH,
        )
        from eval.turing.llms.llm_judge import (
            VALID_MODES,
            run_multi_judge_turing,
        )

        judge_modes = None
        if args.judge_mode and "all" not in args.judge_mode:
            judge_modes = set(args.judge_mode)

        from eval.turing.llms.judge_clients import (
            JUDGE_CONFIGS,
        )

        judge_configs = JUDGE_CONFIGS
        if args.judges:
            judge_filter = set(args.judges)
            judge_configs = [c for c in JUDGE_CONFIGS if c["name"] in judge_filter]
            if not judge_configs:
                print(f"No matching judges for {args.judges}. "
                      f"Available: {[c['name'] for c in JUDGE_CONFIGS]}", file=sys.stderr)
                sys.exit(1)

        all_results = run_multi_judge_turing(
            simulated_path=sim_path or DEFAULT_SIMULATED_PATH,
            real_path=real_path or DEFAULT_REAL_PATH,
            n_pairs=args.n_pairs,
            reasoning_effort=args.reasoning_effort,
            seed=args.seed or 42,
            max_workers=args.max_workers,
            modes=judge_modes,
            split=args.split,
            judge_configs=judge_configs,
        )

        for judge_name, session in sorted(all_results.items()):
            print(f"\n{'=' * 60}")
            print(f"  Judge: {judge_name} ({session.get('model', '?')})")
            print(f"{'=' * 60}")

            if "error" in session:
                print(f"  ERROR: {session['error']}")
                continue

            if "absolute" in session:
                abs_r = session["absolute"]
                print(f"\n  Mode 1 — Absolute:")
                print(f"    Accuracy: {abs_r['accuracy']:.0%} ({abs_r['correct']}/{abs_r['total']})")
                c = abs_r["confusion"]
                print(f"    Confusion: TP={c['tp']} TN={c['tn']} FP={c['fp']} FN={c['fn']}")
                print(f"    Avg Likert (real): {abs_r['avg_likert_real']:.2f}  (sim): {abs_r['avg_likert_sim']:.2f}")

            if "comparative_forced" in session:
                cf_r = session["comparative_forced"]
                print(f"\n  Mode 2 — Comparative (forced):")
                print(f"    Accuracy: {cf_r['accuracy_decided']:.0%} ({cf_r['correct']}/{cf_r['decided']})")

            if "comparative_unsure" in session:
                cu_r = session["comparative_unsure"]
                print(f"\n  Mode 3 — Comparative (unsure allowed):")
                print(f"    Accuracy (decided): {cu_r['accuracy_decided']:.0%} ({cu_r['correct']}/{cu_r['decided']})")
                print(f"    Unsure: {cu_r['unsure']}/{cu_r['total']}")
        return

    if args.build_splits:
        from eval.turing.preprocessing.build_splits import (
            DEFAULT_SPLITS_PATH,
            build_splits,
        )
        from eval.turing.preprocessing.data import (
            DEFAULT_REAL_PATH,
        )

        result = build_splits(
            real_path=real_path or DEFAULT_REAL_PATH,
        )
        print(f"\nSplits ({result['total_pairs']} sessions from real.jsonl):")
        for name, count in result["split_counts"].items():
            print(f"  {name}: {count}")
        print(f"\nSaved to {DEFAULT_SPLITS_PATH}")
        return

    if args.build_real:
        from eval.turing.preprocessing.build_real import (
            DEFAULT_CSV_PATH,
            DEFAULT_REAL_OUTPUT as REAL_OUTPUT,
            build_real_jsonl,
        )

        results = build_real_jsonl(csv_path=args.data or DEFAULT_CSV_PATH)
        print(f"\nBuilt real.jsonl: {len(results)} conversations")
        print(f"Output: {REAL_OUTPUT}")
        return


if __name__ == "__main__":
    main()
