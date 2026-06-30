"""Data split manager — deterministic train/test vignette assignment.

Train = 50 vignettes sampled with seed=42 (identical to current pipeline).
Test  = 50 vignettes sampled from the remaining pool with seed=43.

Usage:
    python -m eval.splits
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_EVAL_DIR = Path(__file__).parent
_REPO_ROOT = Path(__file__).parents[1]
_DEFAULT_SPLITS_PATH = _EVAL_DIR / "splits.json"
_DEFAULT_DATA_PATH = "data/av/av_clean.csv"


def _load_eligible(tsv_path: str, exclude_urgent: bool = True) -> pd.DataFrame:
    sep = "\t" if tsv_path.endswith(".tsv") else ","
    df = pd.read_csv(tsv_path, sep=sep)
    if exclude_urgent and "urgency_status" in df.columns:
        df = df[~df["urgency_status"].str.contains("emergency|urgent", case=False, na=False)]
    return df


def create_splits(
    tsv_path: str,
    train_n: int = 50,
    test_n: int = 50,
    train_seed: int = 42,
    test_seed: int = 43,
    exclude_urgent: bool = True,
) -> dict:
    df = _load_eligible(tsv_path, exclude_urgent)
    total_eligible = len(df)

    train_df = df.sample(n=min(train_n, total_eligible), random_state=train_seed)
    train_case_ids = sorted(train_df["case"].tolist())

    remaining = df[~df["case"].isin(train_case_ids)]
    test_df = remaining.sample(n=min(test_n, len(remaining)), random_state=test_seed)
    test_case_ids = sorted(test_df["case"].tolist())

    return {
        "source": tsv_path,
        "total_eligible": total_eligible,
        "exclude_urgent": exclude_urgent,
        "train": {"seed": train_seed, "n": len(train_case_ids), "case_ids": train_case_ids},
        "test": {"seed": test_seed, "n": len(test_case_ids), "case_ids": test_case_ids},
    }


def save_splits(splits: dict, path: str | Path = _DEFAULT_SPLITS_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: str | Path = _DEFAULT_SPLITS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def load_split_vignettes(
    tsv_path: str,
    split: str,
    splits_path: str | Path = _DEFAULT_SPLITS_PATH,
) -> list[dict]:
    """Load vignettes for a specific split ('train' or 'test').

    Uses the source file recorded in splits.json (the file the splits were
    generated from) so that case IDs always match, regardless of what
    tsv_path the caller passes.

    Returns the same list[dict] format as load_vignettes():
      [{"vignette_id": int, "case_id": str, "vignette_text": str}, ...]
    """
    splits = load_splits(splits_path)
    case_ids = splits[split]["case_ids"]

    source = splits.get("source", tsv_path)
    sep = "\t" if source.endswith(".tsv") else ","
    df = pd.read_csv(source, sep=sep)
    df = df[df["case"].isin(case_ids)].reset_index(drop=True)

    vignettes = []
    for idx, row in df.iterrows():
        vignette_text = str(row.get("patient_summary", "")).strip()
        vignettes.append({
            "vignette_id": idx,
            "case_id": str(row.get("case", idx)),
            "vignette_text": vignette_text,
        })
    return vignettes


if __name__ == "__main__":
    from eval.adherence.dataset import load_vignettes

    abs_data_path = str(_REPO_ROOT / _DEFAULT_DATA_PATH)

    # Generate splits
    splits = create_splits(abs_data_path)
    save_splits(splits)
    print(f"Saved splits to {_DEFAULT_SPLITS_PATH}")
    print(f"  Total eligible: {splits['total_eligible']}")
    print(f"  Train: {splits['train']['n']} vignettes (seed={splits['train']['seed']})")
    print(f"  Test:  {splits['test']['n']} vignettes (seed={splits['test']['seed']})")

    # Verify train matches current load_vignettes(n=50, seed=42)
    current = load_vignettes(abs_data_path, n=50, seed=42)
    current_case_ids = sorted(int(v["case_id"]) for v in current)
    assert current_case_ids == splits["train"]["case_ids"], (
        f"Train split does not match current load_vignettes! "
        f"Expected {current_case_ids}, got {splits['train']['case_ids']}"
    )
    print("  Verified: train split matches current load_vignettes(n=50, seed=42)")

    # Verify no overlap
    overlap = set(splits["train"]["case_ids"]) & set(splits["test"]["case_ids"])
    assert not overlap, f"Train/test overlap: {overlap}"
    print("  Verified: no train/test overlap")
