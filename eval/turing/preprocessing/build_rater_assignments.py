"""Build deterministic rater assignments for the Turing test.

Partitions matched test pairs into 16 rater slots so that human rater i,
LLM rater i (opus), and LLM rater i (gpt) all see the exact same 10
conversations in the exact same order.

Algorithm:
  1. Take 82 matched pairs, discard 2 to get 80.
  2. Split 80 pairs into 8 groups of 10.
  3. Each group produces 2 complementary raters:
     - Rater 2k:   pairs 0-4 as real,  pairs 5-9 as simulated.
     - Rater 2k+1: pairs 0-4 as simulated, pairs 5-9 as real.
  4. Presentation order is randomised independently per rater.

Every conversation (each version of each pair) is seen by exactly one rater.

Usage:
    python -m eval.turing.preprocessing.build_rater_assignments
    python -m eval.turing.preprocessing.build_rater_assignments \
        --simulated path/to/simulated.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from eval.turing.llms.llm_judge import (
    build_matched_pairs,
)
from eval.turing.preprocessing.build_splits import (
    DEFAULT_SPLITS_PATH,
    load_splits,
)
from eval.turing.preprocessing.data import (
    DEFAULT_REAL_PATH,
    load_conversations,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_ASSIGNMENTS_PATH = DATA_DIR / "rater_assignments.json"

_RUNS_DIR = Path(__file__).parent.parent / "llms" / "results" / "runs"
_DEFAULT_RUN_DIR = _RUNS_DIR / "20260515_235747_test"

N_RATERS = 16
N_PER_RATER = 10
N_PAIRS_USED = (N_RATERS * N_PER_RATER) // 2  # 80 — each pair serves 2 raters


def _find_simulated_path(run_dir: str | Path | None = None) -> Path:
    if run_dir is not None:
        p = Path(run_dir) / "simulated.jsonl"
        if p.exists():
            return p
        raise FileNotFoundError(f"No simulated.jsonl in {run_dir}")
    if _DEFAULT_RUN_DIR.exists() and (_DEFAULT_RUN_DIR / "simulated.jsonl").exists():
        return _DEFAULT_RUN_DIR / "simulated.jsonl"
    raise FileNotFoundError("No default simulated.jsonl found")


def build_rater_assignments(
    real_path: str | Path = DEFAULT_REAL_PATH,
    simulated_path: str | Path | None = None,
    splits_path: str | Path = DEFAULT_SPLITS_PATH,
    output_path: str | Path = DEFAULT_ASSIGNMENTS_PATH,
    run_dir: str | Path | None = None,
    seed: int = 42,
) -> dict:
    if simulated_path is None:
        simulated_path = _find_simulated_path(run_dir)
    logger.info("Simulated data: %s", simulated_path)

    real = load_conversations(real_path)
    simulated = load_conversations(simulated_path)
    pairs = build_matched_pairs(real, simulated)

    assignments = load_splits(splits_path)
    test_ids = {sid for sid, s in assignments.items() if s == "test"}
    pairs = [p for p in pairs if p["session_id"] in test_ids]
    pairs.sort(key=lambda p: p["session_id"])

    logger.info("Test pairs available: %d", len(pairs))
    if len(pairs) < N_PAIRS_USED:
        raise ValueError(
            f"Need >= {N_PAIRS_USED} test pairs, got {len(pairs)}"
        )

    rng = random.Random(seed)
    rng.shuffle(pairs)
    used_pairs = pairs[:N_PAIRS_USED]
    discarded = [p["session_id"] for p in pairs[N_PAIRS_USED:]]

    groups = [
        used_pairs[i * 10 : (i + 1) * 10] for i in range(8)
    ]

    raters: dict[str, list[dict]] = {}

    for group_idx, group in enumerate(groups):
        group_rng = random.Random(seed + group_idx)
        group_rng.shuffle(group)

        real_indices = set(range(5))
        sim_indices = set(range(5, 10))

        for rater_offset in range(2):
            rater_idx = group_idx * 2 + rater_offset

            if rater_offset == 0:
                r_idx, s_idx = real_indices, sim_indices
            else:
                r_idx, s_idx = sim_indices, real_indices

            items = []
            for i, pair in enumerate(group):
                version = "real" if i in r_idx else "simulated"
                items.append({
                    "session_id": pair["session_id"],
                    "version": version,
                })

            order_rng = random.Random(seed + 100 + rater_idx)
            order_rng.shuffle(items)

            for order, item in enumerate(items, start=1):
                item["presentation_order"] = order

            raters[str(rater_idx)] = items

    _validate(raters, N_PAIRS_USED)

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "simulated_path": str(simulated_path),
        "n_raters": N_RATERS,
        "n_conversations_per_rater": N_PER_RATER,
        "n_pairs_used": N_PAIRS_USED,
        "discarded_pairs": discarded,
        "raters": raters,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Rater assignments saved to %s", output_path)
    return result


def _validate(raters: dict[str, list[dict]], n_pairs: int) -> None:
    assert len(raters) == N_RATERS, f"Expected {N_RATERS} raters, got {len(raters)}"

    all_slots: list[tuple[str, str]] = []

    for rater_key, items in raters.items():
        assert len(items) == N_PER_RATER, (
            f"Rater {rater_key}: expected {N_PER_RATER} items, got {len(items)}"
        )
        versions = [it["version"] for it in items]
        assert versions.count("real") == 5, f"Rater {rater_key}: not 5 real"
        assert versions.count("simulated") == 5, f"Rater {rater_key}: not 5 simulated"

        sids = [it["session_id"] for it in items]
        assert len(set(sids)) == N_PER_RATER, f"Rater {rater_key}: duplicate session_ids"

        for it in items:
            all_slots.append((it["session_id"], it["version"]))

    assert len(all_slots) == N_RATERS * N_PER_RATER
    assert len(set(all_slots)) == len(all_slots), "Duplicate (session_id, version) across raters"

    sid_versions: dict[str, set[str]] = {}
    for sid, ver in all_slots:
        sid_versions.setdefault(sid, set()).add(ver)
    for sid, vers in sid_versions.items():
        assert vers == {"real", "simulated"}, (
            f"Pair {sid} not used in both versions: {vers}"
        )
    assert len(sid_versions) == n_pairs


def load_rater_assignments(
    path: str | Path = DEFAULT_ASSIGNMENTS_PATH,
) -> dict:
    with open(path) as f:
        return json.load(f)


def load_rater_items(
    rater_index: int,
    real_by_id: dict[str, dict],
    sim_by_id: dict[str, dict],
    path: str | Path = DEFAULT_ASSIGNMENTS_PATH,
) -> list[dict]:
    """Load conversation items for a specific rater, resolved against data."""
    data = load_rater_assignments(path)
    entries = data["raters"][str(rater_index)]

    items = []
    for entry in entries:
        sid = entry["session_id"]
        if entry["version"] == "simulated":
            conv = sim_by_id[sid]
            gt = "simulated"
        else:
            conv = real_by_id[sid]
            gt = "real"
        items.append({
            "id": conv["id"],
            "transcript": conv["transcript"],
            "metadata": conv.get("metadata", {}),
            "_ground_truth": gt,
            "_pair_id": sid,
        })

    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic rater assignments for Turing test"
    )
    parser.add_argument("--real", type=str, default=None, help="Path to real.jsonl")
    parser.add_argument("--simulated", type=str, default=None, help="Path to simulated.jsonl")
    parser.add_argument("--run-dir", type=str, default=None, help="Run dir containing simulated.jsonl")
    parser.add_argument("--output", type=str, default=str(DEFAULT_ASSIGNMENTS_PATH))
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    result = build_rater_assignments(
        real_path=args.real or DEFAULT_REAL_PATH,
        simulated_path=args.simulated,
        run_dir=args.run_dir,
        output_path=args.output,
        seed=args.seed,
    )

    print(f"\nRater assignments ({result['n_raters']} raters):")
    print(f"  Pairs used: {result['n_pairs_used']}")
    print(f"  Discarded: {len(result['discarded_pairs'])}")
    print(f"  Conversations per rater: {result['n_conversations_per_rater']}")
    for rater_key in sorted(result["raters"], key=int):
        items = result["raters"][rater_key]
        n_real = sum(1 for it in items if it["version"] == "real")
        n_sim = sum(1 for it in items if it["version"] == "simulated")
        print(f"  Rater {rater_key:>2}: {n_real}R + {n_sim}S")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
