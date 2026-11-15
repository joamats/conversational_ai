"""Annotator progress for the grader app: who has finished what, and which records still lack votes.

Reads the workbook from a local file or a gs:// URI (read-only) and applies the app's own completion rules
(a parameter-adherence record is done when all its parameters are saved; a clinical-fidelity record when it
is final-submitted).

    python progress_report.py --workbook grader_annotations.xlsx
    python progress_report.py --gcs gs://<bucket>/<path>/grader_annotations.xlsx
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workbook", type=Path, help="local workbook")
    parser.add_argument("--gcs", help="gs:// URI of the workbook, downloaded read-only")
    args = parser.parse_args()
    if bool(args.workbook) == bool(args.gcs):
        parser.error("pass exactly one of --workbook or --gcs")

    if args.workbook:
        path = args.workbook
    else:
        path = Path(tempfile.mkdtemp()) / "grader_annotations.xlsx"
        os.environ["GRADER_WORKBOOK_GCS"] = args.gcs
    os.environ["GRADER_WORKBOOK"] = str(path)

    import grader_app  # reads GRADER_WORKBOOK at call time

    if not args.workbook:
        grader_app.restore_workbook()
        os.environ.pop("GRADER_WORKBOOK_GCS")  # read-only: never upload from here
        if not path.exists():
            raise SystemExit(f"No workbook at {args.gcs} yet.")

    graders = grader_app.grader_ids()
    if not graders:
        raise SystemExit("No data/assignment.json; run make_assignment.py first.")
    votes = grader_app.assignment().get("votes_per_record", 3)
    last_activity: dict[str, str] = {}
    for sheet in grader_app.SHEETS:
        for row in grader_app.read_rows(sheet):
            grader = row["grader_id"]
            last_activity[grader] = max(str(row["saved_at"]), last_activity.get(grader, ""))

    print(f"Workbook: {args.workbook or args.gcs}\n")
    print(f"{'grader':<8}  {'parameter adherence':>20}  {'clinical fidelity':>18}  last activity (UTC)")
    complete_votes: dict[str, Counter] = {task: Counter() for task in grader_app.TASKS}
    for grader in graders:
        cells = []
        for task in grader_app.TASKS:
            mine, done, upcoming = grader_app.record_status(task, grader)
            finished = [record["record_id"] for record in mine if record["record_id"] in done]
            complete_votes[task].update(finished)
            cells.append(f"{len(finished)}/{len(mine)}" + (" ✅" if upcoming is None else f" (next {upcoming})"))
        print(f"{grader:<8}  {cells[0]:>20}  {cells[1]:>18}  {last_activity.get(grader, '-')[:19]}")

    for task, (name, records, _) in grader_app.TASKS.items():
        short = [f"{record['record_id']} ({complete_votes[task][record['record_id']]}/{votes})"
                 for record in records() if complete_votes[task][record["record_id"]] < votes]
        print(f"\n{name}: {len(records()) - len(short)} of {len(records())} records have all {votes} votes"
              + (f"; still short: {', '.join(short)}" if short else " ✅"))

    unknown = sorted(set(last_activity) - set(graders))
    if unknown:
        print(f"\nRows from IDs outside the assignment (e.g. tests): {', '.join(unknown)}")


if __name__ == "__main__":
    main()
