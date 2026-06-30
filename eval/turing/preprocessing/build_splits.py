"""Build train/validation/test splits for the Turing test evaluation.

Assigns each matched session_id to a split so that conversations used for
prompt optimisation are never used in the final test.

Default layout (can be changed via CLI):
  train_1    — first 10 pairs  (prompt optimisation split 1)
  train_2    — next 10 pairs   (prompt optimisation split 2)
  test       — remaining pairs (final held-out evaluation)

The split file is a JSON dict mapping session_id → split name, plus metadata.
The turing judge can then be pointed at a specific split via --split.

Usage:
    python -m eval.turing.preprocessing.build_splits
    python -m eval.turing.preprocessing.build_splits \
        --splits train_1:10 train_2:10 validation:10 test:rest
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from eval.turing.preprocessing.data import (
    DEFAULT_REAL_PATH,
    load_conversations,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_SPLITS_PATH = DATA_DIR / "splits.json"

DEFAULT_SPLIT_SPEC = [
    ("train_1", 10),
    ("train_2", 10),
    ("test", None),  # None = rest
]


def _parse_split_spec(specs: list[str]) -> list[tuple[str, int | None]]:
    result = []
    for spec in specs:
        name, size = spec.split(":")
        result.append((name, None if size == "rest" else int(size)))
    return result


def build_splits(
    real_path: str | Path = DEFAULT_REAL_PATH,
    output_path: str | Path = DEFAULT_SPLITS_PATH,
    split_spec: list[tuple[str, int | None]] | None = None,
    seed: int = 42,
) -> dict:
    """Build and save the split assignment file based on real.jsonl.

    Assigns every real session ID to a split, regardless of whether
    a simulated counterpart exists yet.

    Returns the full splits dict (also written to output_path).
    """
    if split_spec is None:
        split_spec = DEFAULT_SPLIT_SPEC

    real = load_conversations(real_path)
    session_ids = [r["id"] for r in real]
    total = len(session_ids)

    assignments: dict[str, str] = {}
    idx = 0
    for name, size in split_spec:
        if size is None:
            chunk = session_ids[idx:]
        else:
            chunk = session_ids[idx:idx + size]
        for sid in chunk:
            assignments[sid] = name
        idx += len(chunk)

    split_counts = {}
    for name in dict.fromkeys(v for v in assignments.values()):
        split_counts[name] = sum(1 for v in assignments.values() if v == name)

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_pairs": total,
        "assigned": len(assignments),
        "split_counts": split_counts,
        "split_spec": [[name, size if size is not None else "rest"] for name, size in split_spec],
        "assignments": assignments,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Splits saved to %s", output_path)
    for name, count in split_counts.items():
        logger.info("  %s: %d pairs", name, count)

    return result


def load_splits(path: str | Path = DEFAULT_SPLITS_PATH) -> dict[str, str]:
    """Load split assignments as {session_id: split_name}."""
    with open(path) as f:
        data = json.load(f)
    return data["assignments"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/validation/test splits")
    parser.add_argument("--real", type=str, default=None, help="Path to real.jsonl")
    parser.add_argument("--output", type=str, default=str(DEFAULT_SPLITS_PATH), help="Output path for splits.json")
    parser.add_argument("--splits", type=str, nargs="+", default=None,
                        help="Split specs as name:size pairs, e.g. train_1:10 train_2:10 test:rest")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    split_spec = _parse_split_spec(args.splits) if args.splits else None

    result = build_splits(
        real_path=args.real or DEFAULT_REAL_PATH,
        output_path=args.output,
        split_spec=split_spec,
    )

    print(f"\nSplits ({result['total_pairs']} sessions):")
    for name, count in result["split_counts"].items():
        print(f"  {name}: {count}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
