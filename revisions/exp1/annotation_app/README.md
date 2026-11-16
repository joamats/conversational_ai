# Annotation app

Gradio app for the Experiment 1 human-validation study. Graders compare conversations for **parameter adherence** and rate patient claims for **clinical fidelity**, without seeing the LLM judge's output.

## Files

```
grader_app.py          the app
build_review_data.py   builds data/{parameter_adherence,clinical_fidelity}.json from pipeline runs
make_assignment.py     writes data/assignment.json (which grader reviews which records)
progress_report.py     per-grader completion and per-record vote counts
```

## Setup

Python 3.11+.

```bash
pip install -r requirements.txt
```

## Review data

The review data is not included. `data/parameter_adherence.json` and `data/clinical_fidelity.json` are built by `build_review_data.py` from a parameter-adherence run and a clinical-fidelity run of the pipeline (see its docstring). They also keep the LLM judge's output (`llm_selected_job_id`, `llm_classified_claims`, ...) and the ground truth (`activated_job_id`); the app never shows these to graders.

## Run

```bash
python grader_app.py          # http://127.0.0.1:7860
```

Answers go to an Excel workbook (`grader_annotations.xlsx` next to the app; set `GRADER_WORKBOOK` to change it).

## Assignment

`make_assignment.py` gives each record to 3 of 6 graders. With 20 records, C(6,3) = 20, so every record gets a different grader triple and every pair of graders shares exactly 4 records. Each grader sees their records in their own random order; the two tasks are assigned independently.

```bash
python make_assignment.py                                # DS-01 ... DS-06, 3 votes, seed 42
python make_assignment.py --graders AB CD EF GH IJ KL    # other grader IDs
```

With `data/assignment.json` present, graders pick their ID from a fixed list and see only their records. Without it, any ID can be typed and every record is shown. The server enforces the order, not only the UI:

- Records unlock one at a time in the grader's order; finished records stay editable.
- Clinical fidelity unlocks once every Parameter adherence record is done.

**Parameter adherence** is one page per record: all comparisons are on screen, answers are picked in the browser, and **Save record** writes them all. **Clinical fidelity** ratings are saved claim by claim, then the record is final-submitted. Punctuation and capitalisation are rule-based, so graders are not asked about them (`EXCLUDED_PARAMETERS` in `grader_app.py`).

## Workbook

Rows are appended and never edited; analysis keeps the latest row per key. The sheet headers must not change: the app moves a sheet with different headers aside as `legacy_*` and starts an empty one.

| Sheet | One row per | Key |
|---|---|---|
| `parameter_adherence` | (grader, record, parameter) judgement | `grader_id`, `task_id` |
| `fidelity_claims` | rated conversation claim | `grader_id`, `record_id`, `claim_number` |
| `fidelity_summary` | final submission with computed metrics | `grader_id`, `record_id` |

## Progress

```bash
python progress_report.py --workbook grader_annotations.xlsx    # or --gcs gs://<bucket>/<path>/grader_annotations.xlsx
```

## Hosting

Set `GRADER_WORKBOOK_GCS=gs://<bucket>/<path>/grader_annotations.xlsx` (needs `google-cloud-storage` and credentials) to mirror the workbook to Cloud Storage after every save and restore it on start. The download button is then hidden so graders cannot see each other's answers. When `PORT` is set the app is served through FastAPI, with a `/_ah/warmup` route.
