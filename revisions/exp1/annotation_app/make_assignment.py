"""Assign the Experiment 1 review records to graders and write data/assignment.json.

Each record is reviewed by ``--votes`` distinct graders and every grader gets
the same number of records. When the number of records equals C(graders, votes)
— the default 20 records, 6 graders, 3 votes — every record gets a *different*
grader triple, which also makes the design pairwise balanced: each pair of
graders shares the same number of records (4), so agreement statistics are not
dominated by any one pair. Each grader's records are shown in their own random
order, so no record is always seen first or last.

The two tasks (parameter adherence, clinical fidelity) are assigned
independently with the same design.

    python make_assignment.py                      # DS-01 … DS-06, 3 votes, seed 42
    python make_assignment.py --graders AB CD EF GH IJ KL
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path

ROOT = Path(__file__).parent
TASK_FILES = {
    "parameter_adherence": ROOT / "data" / "parameter_adherence.json",
    "clinical_fidelity": ROOT / "data" / "clinical_fidelity.json",
}


def balanced_assignment(record_ids: list[str], graders: list[str], votes: int, rng: random.Random) -> dict[str, list[str]]:
    """Map each grader to their records (in presentation order)."""
    if votes > len(graders):
        raise ValueError(f"Need at least {votes} graders for {votes} votes per record.")
    if len(record_ids) * votes % len(graders):
        raise ValueError(f"{len(record_ids)} records x {votes} votes cannot be split evenly over {len(graders)} graders.")
    records = rng.sample(record_ids, len(record_ids))
    if len(records) == comb(len(graders), votes):
        groups = list(combinations(graders, votes))
        rng.shuffle(groups)
    else:
        # Equal load, but pair overlap is only approximately balanced.
        order = rng.sample(graders, len(graders))
        groups = [tuple(order[(i * votes + j) % len(order)] for j in range(votes)) for i in range(len(records))]
    per_grader: dict[str, list[str]] = {grader: [] for grader in graders}
    for record_id, group in zip(records, groups):
        for grader in group:
            per_grader[grader].append(record_id)
    for grader in graders:
        rng.shuffle(per_grader[grader])
    return per_grader


def check(per_grader: dict[str, list[str]], record_ids: list[str], votes: int) -> Counter:
    """Assert the design invariants; return the pairwise-overlap counts."""
    raters = Counter(record_id for records in per_grader.values() for record_id in records)
    assert set(raters) == set(record_ids) and set(raters.values()) == {votes}, "every record needs exactly `votes` graders"
    assert all(len(set(records)) == len(records) for records in per_grader.values()), "a grader got a record twice"
    assert len({len(records) for records in per_grader.values()}) == 1, "graders have unequal loads"
    return Counter({
        (a, b): len(set(per_grader[a]) & set(per_grader[b])) for a, b in combinations(sorted(per_grader), 2)
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graders", nargs="+", default=[f"DS-{i:02d}" for i in range(1, 7)])
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = {"seed": args.seed, "votes_per_record": args.votes, "graders": args.graders}
    for task, path in TASK_FILES.items():
        record_ids = [record["record_id"] for record in json.loads(path.read_text())]
        per_grader = balanced_assignment(record_ids, args.graders, args.votes, random.Random(f"{args.seed}:{task}"))
        overlap = check(per_grader, record_ids, args.votes)
        out[task] = per_grader
        print(f"{task}: {len(record_ids)} records x {args.votes} votes; "
              f"{len(per_grader[args.graders[0]])} records per grader; "
              f"pairwise overlap {min(overlap.values())}–{max(overlap.values())} records")
    path = ROOT / "data" / "assignment.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
