"""End-to-end preprocessing pipeline: build-real -> extract -> normalize.

"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from eval.turing.preprocessing.extract import (
    DEFAULT_CSV_PATH,
    DEFAULT_PARAMS_OUTPUT as EXTRACT_OUTPUT,
    run_extraction,
)
from eval.turing.preprocessing.normalize import (
    DEFAULT_OUTPUT_PATH as NORM_OUTPUT,
    DEFAULT_RAW_PATH as SIM_RAW_PATH,
    run_normalization,
)

logger = logging.getLogger(__name__)


def run_pipeline(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    extract_model: str = "gpt-5.5-2026-04-23",
    reasoning_effort: str = "medium",
    max_extract_workers: int = 4,
    skip_extract: bool = False,
    skip_build_real: bool = False,
) -> dict:
    """Run build-real -> extract -> normalize. Returns paths and counts per stage."""
    t0 = time.time()
    summary: dict = {}

    # -- Step 0: Build real.jsonl -------------------------------------------
    if skip_build_real:
        logger.info("Skipping build-real (--skip-build-real)")
        summary["build_real"] = {"skipped": True}
    else:
        logger.info("Step 1/3: Building real.jsonl from real_clean.csv")
        from eval.turing.preprocessing.build_real import (
            build_real_jsonl,
        )
        real_convos = build_real_jsonl(csv_path=csv_path)
        summary["build_real"] = {"total": len(real_convos)}
        logger.info("Built real.jsonl: %d conversations", len(real_convos))

    # -- Step 1: Extract params ---------------------------------------------
    if skip_extract:
        logger.info("Skipping extraction (--skip-extract)")
        summary["extract"] = {"skipped": True, "path": str(EXTRACT_OUTPUT)}
    else:
        logger.info("Step 2/3: Extracting params from real conversations")
        extracted = run_extraction(
            csv_path=csv_path,
            output_path=EXTRACT_OUTPUT,
            model=extract_model,
            reasoning_effort=reasoning_effort,
            max_workers=max_extract_workers,
        )
        n_ok = sum(1 for r in extracted if "error" not in r)
        summary["extract"] = {"total": len(extracted), "success": n_ok, "path": str(EXTRACT_OUTPUT)}
        logger.info("Extraction done: %d/%d", n_ok, len(extracted))

    # -- Step 2: Normalize --------------------------------------------------
    logger.info("Step 3/3: Normalizing simulated conversations to turing format")
    normalized = run_normalization(raw_path=SIM_RAW_PATH, output_path=NORM_OUTPUT)
    summary["normalize"] = {"total": len(normalized), "path": str(NORM_OUTPUT)}

    summary["duration_seconds"] = round(time.time() - t0, 1)
    logger.info("Pipeline complete in %.1fs", summary["duration_seconds"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turing preprocessing pipeline: build-real -> extract -> normalize"
    )
    parser.add_argument("--data", type=str, default=None, help=f"Path to real_clean.csv (default: {DEFAULT_CSV_PATH})")
    parser.add_argument("--model", type=str, default="gpt-5.5-2026-04-23", help="Extraction model")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--max-extract-workers", type=int, default=4, help="Parallel workers for param extraction")
    parser.add_argument("--skip-build-real", action="store_true", help="Skip building real.jsonl, reuse existing")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction, reuse existing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    summary = run_pipeline(
        csv_path=args.data or DEFAULT_CSV_PATH,
        extract_model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_extract_workers=args.max_extract_workers,
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
            print(f"  {stage}: {success}/{total} -> {path}" if path else f"  {stage}: {total}")


if __name__ == "__main__":
    main()
