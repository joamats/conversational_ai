"""Gradio human-validation UI for the patient-simulator Experiment 1 review."""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import gradio as gr
from openpyxl import Workbook, load_workbook

ROOT = Path(__file__).parent
WORKBOOK_DEFAULT = ROOT / "grader_annotations.xlsx"
PARAMETER_DATA = ROOT / "data" / "parameter_adherence.json"
FIDELITY_DATA = ROOT / "data" / "clinical_fidelity.json"
# Written by make_assignment.py; when absent every grader sees every record.
ASSIGNMENT_DATA = ROOT / "data" / "assignment.json"

SHEETS = {
    "parameter_adherence": (
        "saved_at", "grader_id", "record_id", "vignette_id", "case_id", "task_id", "parameter_name",
        "parameter_value", "patient_goal", "conversation_a_job_id", "conversation_b_job_id",
        "selected_conversation", "selected_job_id", "goal_conveyed_a", "goal_conveyed_b", "comments",
    ),
    "fidelity_claims": (
        "saved_at", "grader_id", "record_id", "job_id", "case_id", "claim_number", "claim", "category",
        "score", "score_label", "supporting_vignette_claim_number", "supporting_vignette_claim",
    ),
    "fidelity_summary": (
        "saved_at", "grader_id", "record_id", "job_id", "case_id", "total_claims", "n_supported",
        "n_unsupported", "mean_accuracy", "mean_plausibility", "confirmed_rate", "contradiction_rate",
        "n_vignette_claims", "n_vignette_mentioned", "n_vignette_not_mentioned", "coverage_rate",
        "unsupported_rate", "accuracy_1", "accuracy_2", "accuracy_3", "accuracy_4", "plausibility_1",
        "plausibility_2", "plausibility_3", "plausibility_4", "not_mentioned_vignette_claims", "comments",
    ),
}
# Rule-based settings (the simulator edits the text mechanically) are not asked: hidden from the UI and ignored
# for completion. Answers already saved for them stay in the workbook.
EXCLUDED_PARAMETERS = frozenset({"punctuation_style", "capitalisation"})
THEME = gr.themes.Default(primary_hue="blue", neutral_hue="slate")
# Hide the Chatbot clear button (transcripts are read-only).
CSS = ".transcript button[aria-label='Clear'] { display: none; }"

ACCURACY = {
    4: ("Accurate", "Fully consistent with the vignette fact. No meaningful error."),
    3: ("Imprecise", "Broadly consistent, with vagueness or minor inaccuracy that would not change clinical reasoning."),
    2: ("Distorted", "Based on a real vignette fact but altered in a clinically meaningful way."),
    1: ("Contradicted", "Directly conflicts with a specific vignette fact."),
}
PLAUSIBILITY = {
    4: ("Expected", "A response any real patient with this clinical profile would likely give."),
    3: ("Plausible", "Realistic for this profile, though not directly implied by the vignette."),
    2: ("Questionable", "Unlikely for this profile; would raise clinical suspicion."),
    1: ("Implausible", "Contradicts or is clinically incompatible with the patient's known profile."),
}

HEADER_TEXT = """
# Patient simulator validation

A **patient simulator** turns a clinical vignette into a patient profile and holds a multi-turn conversation with an AI doctor.
Persona settings (recall, anxiety, health literacy, typos, goals, …) change how the patient communicates and what the doctor hears.
You are checking two things independently of the automated judge, whose output is not shown here:

1. **Parameter adherence:** each record is one page with all its comparisons: for every parameter, pick the conversation in which it is active, then press **Save record** once at the bottom.
2. **Clinical fidelity:** for each conversation, rate every patient claim against the source vignette, then confirm which vignette facts were never mentioned.

Select your grader ID first; you will only see the records assigned to you. Please complete **all** of them:
records open one at a time in your assigned order, and **Clinical fidelity unlocks once every Parameter adherence record is done**.
Every answer is saved as you go, so you can stop and come back later with the same grader ID.
"""

RUBRIC_TEXT = "\n".join([
    "**Supported claims** (a corresponding fact exists in the vignette): rate **accuracy**",
    "",
    *(f"- **{score} = {label}:** {text}" for score, (label, text) in ACCURACY.items()),
    "",
    "**Unsupported claims** (the patient made this up in response to a question the vignette does not cover): rate **plausibility** given everything known about the patient",
    "",
    *(f"- **{score} = {label}:** {text}" for score, (label, text) in PLAUSIBILITY.items()),
])


# ---------------------------------------------------------------------------
# Workbook
# ---------------------------------------------------------------------------

# Serialises workbook reads and writes: handlers run in parallel threads.
_WORKBOOK_LOCK = threading.RLock()


def workbook_path() -> Path:
    return Path(os.environ.get("GRADER_WORKBOOK", WORKBOOK_DEFAULT))


def workbook_gcs_uri() -> str:
    """Optional gs:// URI the workbook is mirrored to (for hosts with ephemeral disk)."""
    return os.environ.get("GRADER_WORKBOOK_GCS", "")


@cache
def _gcs_client() -> Any:
    from google.cloud import storage  # only needed when GRADER_WORKBOOK_GCS is set

    return storage.Client()


def _gcs_blob(uri: str) -> Any:
    bucket, _, name = uri.removeprefix("gs://").partition("/")
    return _gcs_client().bucket(bucket).blob(name)


def restore_workbook() -> None:
    """Download the mirrored workbook on a fresh instance (no-op without a GCS mirror)."""
    uri, path = workbook_gcs_uri(), workbook_path()
    with _WORKBOOK_LOCK:
        if uri and not path.exists() and (blob := _gcs_blob(uri)).exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(path))
            _backup_restored_workbook(uri, blob)


def _backup_restored_workbook(uri: str, blob: Any) -> None:
    """Snapshot the restored workbook under backup/ (add-only, best effort: never blocks startup)."""
    folder, _, filename = uri.rpartition("/")
    prefix = folder.removeprefix("gs://").partition("/")[2]
    name = f"{prefix}/backup/{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{filename}".lstrip("/")
    try:
        blob.bucket.copy_blob(blob, blob.bucket, name, if_generation_match=0)
    except Exception:
        logging.getLogger(__name__).exception("Could not snapshot the restored workbook to backup/%s", name)


# One upload at a time, so the last upload always holds every saved row.
_UPLOAD_LOCK = threading.Lock()
# Set when a mirror upload failed, so a retried save (which appends nothing) still re-uploads.
_MIRROR_STALE = threading.Event()


def _sync_to_gcs(path: Path, sheet_name: str, saved_at: str, rows: list[dict[str, Any]]) -> None:
    """Upload the workbook, plus an append-only JSON copy of the new rows as a backup (in parallel)."""
    uri = workbook_gcs_uri()
    if not uri:
        return
    log_uri = f"{uri.rsplit('/', 1)[0]}/log/{sheet_name}/{saved_at}_{uuid.uuid4().hex[:8]}.json"
    with _UPLOAD_LOCK:
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                uploads = [
                    pool.submit(_gcs_blob(uri).upload_from_filename, str(path)),
                    pool.submit(_gcs_blob(log_uri).upload_from_string, json.dumps(rows, default=str),
                                content_type="application/json"),
                ]
                for upload in uploads:
                    upload.result()  # re-raise: a failed mirror must not look like a successful save
        except Exception:
            _MIRROR_STALE.set()
            raise
        _MIRROR_STALE.clear()


def resync_if_stale() -> None:
    """Re-upload the workbook if the last mirror upload failed."""
    uri = workbook_gcs_uri()
    if uri and _MIRROR_STALE.is_set():
        with _UPLOAD_LOCK:
            _gcs_blob(uri).upload_from_filename(str(workbook_path()))
            _MIRROR_STALE.clear()


def _save_atomic(workbook: Workbook, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    workbook.save(tmp)
    os.replace(tmp, path)


def ensure_workbook(path: Path) -> None:
    """Create any missing sheet; a sheet with outdated headers is kept under a legacy_ name."""
    with _WORKBOOK_LOCK:
        if path.exists():
            workbook = load_workbook(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            workbook = Workbook()
            workbook.remove(workbook.active)
        changed = False
        for name, headers in SHEETS.items():
            if name in workbook.sheetnames:
                if tuple(next(workbook[name].iter_rows(max_row=1, values_only=True), ())) == headers:
                    continue
                legacy, suffix = f"legacy_{name}", 1
                while legacy in workbook.sheetnames:
                    suffix += 1
                    legacy = f"legacy_{name}_{suffix}"
                workbook[name].title = legacy
            sheet = workbook.create_sheet(name)
            sheet.append(list(headers))
            sheet.freeze_panes = "A2"
            changed = True
        if changed:
            _save_atomic(workbook, path)
        workbook.close()


# Parsed rows of every sheet, re-parsed only when the file on disk changes.
_CACHE: dict[str, Any] = {}


def _file_signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def _snapshot(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Rows of every sheet, creating the workbook if needed. Caller holds _WORKBOOK_LOCK."""
    if not path.exists() or _CACHE.get("signature") != _file_signature(path):
        ensure_workbook(path)
        workbook = load_workbook(path, read_only=True)
        rows = {}
        for name in SHEETS:
            sheet_rows = workbook[name].iter_rows(values_only=True)
            headers = next(sheet_rows)
            rows[name] = [dict(zip(headers, row)) for row in sheet_rows]
        workbook.close()
        _CACHE.update(signature=_file_signature(path), rows=rows)
    return _CACHE["rows"]


def append_rows(sheet_name: str, rows: list[dict[str, Any]]) -> None:
    path = workbook_path()
    saved_at = datetime.now(UTC).isoformat()
    headers = SHEETS[sheet_name]
    with _WORKBOOK_LOCK:
        cached = _snapshot(path)
        workbook = load_workbook(path)
        appended = [[saved_at, *(row.get(header, "") for header in headers[1:])] for row in rows]
        for values in appended:
            workbook[sheet_name].append(values)
        _save_atomic(workbook, path)
        workbook.close()
        # Keep the cache in step with the file; empty cells read back as None.
        cached[sheet_name].extend({h: None if v == "" else v for h, v in zip(headers, values)} for values in appended)
        _CACHE["signature"] = _file_signature(path)
    _sync_to_gcs(path, sheet_name, saved_at, rows)


def read_rows(sheet_name: str) -> list[dict[str, Any]]:
    path = workbook_path()
    with _WORKBOOK_LOCK:
        return list(_snapshot(path)[sheet_name])


def export_workbook() -> str:
    path = workbook_path()
    ensure_workbook(path)
    return str(path)


# ---------------------------------------------------------------------------
# Review data
# ---------------------------------------------------------------------------

@cache
def _load(path: Path) -> list[dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []


def parameter_records() -> list[dict[str, Any]]:
    return _load(PARAMETER_DATA)


def fidelity_records() -> list[dict[str, Any]]:
    return _load(FIDELITY_DATA)


def assignment() -> dict[str, Any]:
    """{"graders": [...], "parameter_adherence": {grader: [record_id, ...]}, "clinical_fidelity": {...}}."""
    data = _load(ASSIGNMENT_DATA)
    return data if isinstance(data, dict) else {}


def grader_ids() -> list[str]:
    return list(assignment().get("graders", []))


def _grader(grader_id: str | None) -> str:
    return (grader_id or "").strip()


def assigned_records(task: str, grader_id: str | None, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The grader's records in their assigned presentation order; every record when no assignment exists."""
    plan = assignment().get(task)
    if not plan:
        return records
    by_id = {record["record_id"]: record for record in records}
    return [by_id[record_id] for record_id in plan.get(_grader(grader_id), []) if record_id in by_id]


def assignment_error(task: str, grader_id: str, record_id: str, records: list[dict[str, Any]]) -> str:
    """Warning message if this grader may not save answers for this record, else ''.

    With an assignment, records unlock in order (finished ones stay editable) and
    clinical fidelity unlocks only after every parameter-adherence record is done.
    """
    if not grader_id:
        return "⚠️ Select your grader ID before saving."
    if assignment() and grader_id not in grader_ids():
        return f"⚠️ Unknown grader ID {grader_id!r}."
    if record_id not in {record["record_id"] for record in assigned_records(task, grader_id, records)}:
        return f"⚠️ Record {record_id} is not assigned to {grader_id}."
    if assignment():
        if task == "clinical_fidelity" and not task_complete("parameter_adherence", grader_id):
            return "⚠️ Finish all your Parameter adherence records before starting Clinical fidelity."
        _, done, upcoming = record_status(task, grader_id)
        if record_id not in done and record_id != upcoming:
            return f"⚠️ Records open in order: finish {upcoming} first."
    return ""


def _by_id(records: list[dict[str, Any]], record_id: str | None) -> dict[str, Any]:
    return next((record for record in records if record["record_id"] == record_id), {})


def review_tasks(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The comparisons annotators are asked about: every task except the rule-based parameters."""
    return [task for task in record.get("tasks", []) if task["param_name"] not in EXCLUDED_PARAMETERS]


def _record_choices(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(f"{record['record_id']} · case {record['case_id']}", record["record_id"]) for record in records]


def _completed_parameter_records(grader_id: str) -> set[str]:
    saved: dict[str, set[str]] = {}
    for row in read_rows("parameter_adherence"):
        if row["grader_id"] == grader_id:
            saved.setdefault(row["record_id"], set()).add(row["task_id"])
    return {
        record["record_id"] for record in parameter_records()
        if {task["task_id"] for task in review_tasks(record)} <= saved.get(record["record_id"], set())
    }


def _submitted_fidelity_records(grader_id: str) -> set[str]:
    return {row["record_id"] for row in read_rows("fidelity_summary") if row["grader_id"] == grader_id}


TASKS = {
    "parameter_adherence": ("Parameter adherence", parameter_records, _completed_parameter_records),
    "clinical_fidelity": ("Clinical fidelity", fidelity_records, _submitted_fidelity_records),
}


def record_status(task: str, grader_id: str | None) -> tuple[list[dict[str, Any]], set[str], str | None]:
    """(the grader's records in order, ids of finished ones, id of the next one to do or None if all done)."""
    _, records, done_fn = TASKS[task]
    mine = assigned_records(task, grader_id, records())
    done = done_fn(_grader(grader_id))
    upcoming = next((record["record_id"] for record in mine if record["record_id"] not in done), None)
    return mine, done, upcoming


def task_complete(task: str, grader_id: str | None) -> bool:
    mine, _, upcoming = record_status(task, grader_id)
    return bool(mine) and upcoming is None


def record_dropdown(task: str, grader_id: str | None) -> Any:
    """Finished records plus the next one, opened at the next one (every record when there is no assignment)."""
    mine, done, upcoming = record_status(task, grader_id)
    choices = [
        (f"{'✅' if record['record_id'] in done else '▶'} {number}/{len(mine)} · {record['record_id']} · case {record['case_id']}",
         record["record_id"])
        for number, record in enumerate(mine, start=1)
        if not assignment() or record["record_id"] in done or record["record_id"] == upcoming
    ]
    return gr.update(choices=choices, value=upcoming or (choices[-1][1] if choices else None))


def position_text(task: str, grader_id: str | None, record_id: str) -> str:
    if not assignment():
        return ""
    ids = [record["record_id"] for record in assigned_records(task, grader_id, TASKS[task][1]())]
    return f"Your record **{ids.index(record_id) + 1} of {len(ids)}** · " if record_id in ids else ""


def progress_message(grader_id: str | None) -> str:
    grader_id = _grader(grader_id)
    if not grader_id:
        return "Select your grader ID to see your assigned records."
    if all(task_complete(task, grader_id) for task in TASKS):
        return f"🎉 **{grader_id}**: every assigned record in both tasks is done. Thank you, you can close this page."
    parts = []
    for task, (name, _, _) in TASKS.items():
        mine, done, _ = record_status(task, grader_id)
        parts.append(f"{name}: **{len(done & {record['record_id'] for record in mine})} of {len(mine)}** done")
    if assignment() and not task_complete("parameter_adherence", grader_id):
        parts[-1] += " (unlocks after Parameter adherence)"
    return f"**{grader_id}** · " + " · ".join(parts)


def fidelity_tab_update(grader_id: str | None) -> Any:
    unlocked = not assignment() or task_complete("parameter_adherence", grader_id)
    return gr.update(interactive=unlocked, label="Clinical fidelity" if unlocked else "Clinical fidelity 🔒")


def _chat_messages(transcript: list[dict[str, str]]) -> list[dict[str, str]]:
    messages = []
    for turn in transcript:
        role = "assistant" if turn.get("role") == "doctor" else "user"
        speaker = "Doctor" if role == "assistant" else "Patient"
        messages.append({"role": role, "content": f"**{speaker}:** {turn.get('content', '')}"})
    return messages


def _transcript_view(label: str, height: int = 480) -> gr.Chatbot:
    """Read-only transcript: opens at the first turn, no clear/share/like controls."""
    return gr.Chatbot(label=label, height=height, autoscroll=False, buttons=[], feedback_options=None,
                      elem_classes=["transcript"])


def _value_text(task: dict[str, Any]) -> str:
    label, value = task["active_label"], task["active_value"].replace("_", " ")
    return label if label.lower() == value.lower() else f"{label} ({value})"


# ---------------------------------------------------------------------------
# Experiment 1: parameter adherence
# ---------------------------------------------------------------------------

# One page per record: all comparisons on screen, answers picked client-side, one Save writes them all.
# The comparison slots are built once and shown, filled or hidden per record.
SLOT_PARTS = 8  # outputs per slot: group, text, chat A, chat B, pick, goal A, goal B, comment
PARAMETER_CHAT_HEIGHT = 340
PARAMETER_INSTRUCTIONS = (
    "All comparisons for this record are below: read the two conversations, click your answer, then scroll to the "
    "next one. **Nothing is sent until you press Save record at the bottom**, so there is no waiting between "
    "questions. Press Save before switching to another record."
)


def _saved_parameter_answers(grader_id: str, record_id: str) -> dict[str, dict[str, Any]]:
    """Latest saved answer per task for this grader and record."""
    if not grader_id:
        return {}
    return {
        row["task_id"]: row for row in read_rows("parameter_adherence")
        if row["grader_id"] == grader_id and row["record_id"] == record_id
    }


def parameter_task_markdown(task: dict[str, Any], number: int, total: int) -> str:
    if task["param_name"] == "patient_goal":
        return (
            f"### {number} of {total} · Patient goal\n> {task['patient_goal']}\n\n"
            "For **each** conversation, decide whether the patient conveys this goal to the doctor."
        )
    return (
        f"### {number} of {total} · {task['param_label']}: {_value_text(task)}\n"
        f"{task['param_description']}\n\n"
        f"**What to look for:** {task['active_signals']}\n\n"
        f"One patient was configured as **{task['active_label']}** and the other as **{task['neutral_label']}**. "
        f"**Select the conversation that shows more evidence of \"{task['active_label']}\".**"
    )


def _slot_updates(task: dict[str, Any], number: int, total: int, previous: dict[str, Any]) -> tuple[Any, ...]:
    """Outputs for one comparison slot, pre-filled with the grader's latest saved answer."""
    conversation_a, conversation_b = task["conversations"]
    is_goal = task["param_name"] == "patient_goal"
    return (
        gr.update(visible=True), parameter_task_markdown(task, number, total),
        _chat_messages(conversation_a["transcript"]), _chat_messages(conversation_b["transcript"]),
        gr.update(visible=not is_goal, value=None if is_goal else previous.get("selected_conversation") or None),
        gr.update(visible=is_goal, value=previous.get("goal_conveyed_a") or None),
        gr.update(visible=is_goal, value=previous.get("goal_conveyed_b") or None),
        previous.get("comments") or "",
    )


def _empty_slot() -> tuple[Any, ...]:
    return (gr.update(visible=False), "", [], [], gr.update(value=None), gr.update(value=None), gr.update(value=None), "")


def parameter_header(grader_id: str | None, record: dict[str, Any]) -> str:
    tasks = review_tasks(record)
    saved = _saved_parameter_answers(_grader(grader_id), record["record_id"])
    return (
        f"{position_text('parameter_adherence', grader_id, record['record_id'])}"
        f"**Record {record['record_id']} · case {record['case_id']}**: "
        f"{len(tasks)} comparisons, {sum(task['task_id'] in saved for task in tasks)} saved so far"
    )


def parameter_page(grader_id: str | None, record_id: str | None, n_slots: int) -> tuple[Any, ...]:
    """Outputs: the header, then SLOT_PARTS values for each of the n_slots comparison slots."""
    record = _by_id(parameter_records(), record_id)
    tasks = review_tasks(record)
    if not tasks:
        message = ("Select your grader ID, then a record." if parameter_records()
                   else "No parameter-adherence data found. Run build_review_data.py first.")
        return message, *(part for _ in range(n_slots) for part in _empty_slot())
    saved = _saved_parameter_answers(_grader(grader_id), record["record_id"])
    filled = (_slot_updates(task, number, len(tasks), saved.get(task["task_id"], {}))
              for number, task in enumerate(tasks, start=1))
    return (
        parameter_header(grader_id, record),
        *(part for slot in filled for part in slot),
        *(part for _ in range(n_slots - len(tasks)) for part in _empty_slot()),
    )


def save_parameter_answers(
    grader_id: str, record_id: str, picks: list[str | None], goals_a: list[str | None], goals_b: list[str | None],
    comments: list[str | None],
) -> tuple[str, bool]:
    """Save every answered comparison of a record in one write; the lists follow review_tasks() order.

    Only answers that are new or changed are appended, so saving twice adds nothing. Returns
    (message, whether every comparison of the record now has an answer).
    """
    grader_id = _grader(grader_id)
    if error := assignment_error("parameter_adherence", grader_id, record_id, parameter_records()):
        return error, False
    record = _by_id(parameter_records(), record_id)
    tasks = review_tasks(record)
    saved = _saved_parameter_answers(grader_id, record_id)
    new_rows, unanswered = [], []
    for number, task in enumerate(tasks):
        conversation_a, conversation_b = task["conversations"]
        pick, goal_a, goal_b, note = (values[number] if number < len(values) else None
                                      for values in (picks, goals_a, goals_b, comments))
        if task["param_name"] == "patient_goal":
            answered = bool(goal_a and goal_b)
            selected = {("Yes", "No"): "A", ("No", "Yes"): "B"}.get((goal_a, goal_b), "")
        else:
            answered, selected, goal_a, goal_b = pick in ("A", "B"), pick if pick in ("A", "B") else "", "", ""
        note = (note or "").strip()
        previous = saved.get(task["task_id"])
        if not answered:
            if previous is None:
                unanswered.append(f"{number + 1} ({task['param_label']})")
            continue
        if previous and (
            previous.get("selected_conversation") or "", previous.get("goal_conveyed_a") or "",
            previous.get("goal_conveyed_b") or "", previous.get("comments") or "",
        ) == (selected, goal_a or "", goal_b or "", note):
            continue
        new_rows.append({
            "grader_id": grader_id, "record_id": record["record_id"], "vignette_id": record["vignette_id"],
            "case_id": record["case_id"], "task_id": task["task_id"], "parameter_name": task["param_name"],
            "parameter_value": task["active_value"], "patient_goal": task["patient_goal"],
            "conversation_a_job_id": conversation_a["job_id"], "conversation_b_job_id": conversation_b["job_id"],
            "selected_conversation": selected,
            "selected_job_id": {"A": conversation_a["job_id"], "B": conversation_b["job_id"]}.get(selected, ""),
            "goal_conveyed_a": goal_a or "", "goal_conveyed_b": goal_b or "", "comments": note,
        })
    if new_rows:
        append_rows("parameter_adherence", new_rows)
    else:
        resync_if_stale()
    if unanswered:
        return f"⚠️ Saved {len(new_rows)} answer(s), but still unanswered: {', '.join(unanswered)}.", False
    return f"✅ Record {record['record_id']} saved: all {len(tasks)} comparisons answered.", True


def _build_parameter_tab(grader: gr.Textbox, status: gr.Markdown) -> tuple[Any, list[Any], gr.Dropdown, Any]:
    """Build the tab; returns (load fn, its outputs, record dropdown, save event) so the app can wire the rest."""
    n_slots = max((len(review_tasks(record)) for record in parameter_records()), default=0)
    # With an assignment the list is filled in once the grader picks their ID.
    choices = [] if assignment() else _record_choices(parameter_records())
    record_id = gr.Dropdown(choices=choices, value=choices[0][1] if choices else None, label="Your records")
    header = gr.Markdown()
    gr.Markdown(PARAMETER_INSTRUCTIONS)
    slots = []
    for _ in range(n_slots):
        with gr.Group(visible=False) as group:
            task_text = gr.Markdown()
            with gr.Row(equal_height=True):
                chat_a = _transcript_view("Conversation A", PARAMETER_CHAT_HEIGHT)
                chat_b = _transcript_view("Conversation B", PARAMETER_CHAT_HEIGHT)
            pick = gr.Radio([("Conversation A", "A"), ("Conversation B", "B")], label="Your answer")
            goal_a = gr.Radio(["Yes", "No"], label="Is the goal conveyed in Conversation A?", visible=False)
            goal_b = gr.Radio(["Yes", "No"], label="Is the goal conveyed in Conversation B?", visible=False)
            comment = gr.Textbox(label="Comments (optional)", lines=1)
        slots.append((group, task_text, chat_a, chat_b, pick, goal_a, goal_b, comment))
    save = gr.Button("💾 Save record", variant="primary", size="lg")
    feedback = gr.Markdown()

    page_outputs = [header, *(part for slot in slots for part in slot)]
    answer_inputs = [slot[index] for index in (4, 5, 6, 7) for slot in slots]  # picks, goals A, goals B, comments

    def open_record(grader_id: str, selected_record: str) -> tuple[Any, ...]:
        return parameter_page(grader_id, selected_record, n_slots)

    def save_and_advance(grader_id, selected_record, *values):
        picks, goals_a, goals_b, notes = (list(values[part * n_slots:(part + 1) * n_slots]) for part in range(4))
        if error := assignment_error("parameter_adherence", _grader(grader_id), selected_record, parameter_records()):
            return error, gr.update(), gr.update()
        was_done = selected_record in _completed_parameter_records(_grader(grader_id))
        message, complete = save_parameter_answers(grader_id, selected_record, picks, goals_a, goals_b, notes)
        record = _by_id(parameter_records(), selected_record)
        dropdown = gr.update()
        if complete and not was_done:
            # Record just finished: unlock the next one; its change event opens it.
            dropdown = record_dropdown("parameter_adherence", grader_id)
            message += (" Opening your next record." if record_status("parameter_adherence", grader_id)[2]
                        else " All your Parameter adherence records are done: Clinical fidelity is now unlocked.")
        return message, dropdown, parameter_header(grader_id, record)

    # Saving the last question leaves the page scrolled to the bottom; start the next record at its top.
    record_id.change(fn=open_record, inputs=[grader, record_id], outputs=page_outputs, show_progress="minimal").then(
        fn=None, js="() => window.scrollTo({top: 0, behavior: 'smooth'})",
    )
    grader.change(fn=lambda g: record_dropdown("parameter_adherence", g), inputs=grader, outputs=record_id).then(
        fn=open_record, inputs=[grader, record_id], outputs=page_outputs, show_progress="minimal",
    )
    saved = save.click(
        fn=save_and_advance, inputs=[grader, record_id, *answer_inputs], outputs=[feedback, record_id, header],
        show_progress="minimal",
    ).then(fn=progress_message, inputs=grader, outputs=[status])
    return open_record, page_outputs, record_id, saved


# ---------------------------------------------------------------------------
# Experiment 1: clinical fidelity
# ---------------------------------------------------------------------------

def compute_fidelity_metrics(answers: list[dict[str, Any]], n_vignette_claims: int, n_not_mentioned: int) -> dict[str, Any]:
    """Conversation-level fidelity metrics (same definitions as the paper's pipeline)."""
    supported = [answer["score"] for answer in answers if answer["category"] == "supported"]
    unsupported = [answer["score"] for answer in answers if answer["category"] == "unsupported"]
    n_mentioned = n_vignette_claims - n_not_mentioned
    total = len(supported) + len(unsupported)
    metrics = {
        "total_claims": total,
        "n_supported": len(supported),
        "n_unsupported": len(unsupported),
        "mean_accuracy": round(sum(supported) / len(supported), 3) if supported else 0.0,
        "mean_plausibility": round(sum(unsupported) / len(unsupported), 3) if unsupported else 0.0,
        "confirmed_rate": round(supported.count(4) / len(supported), 4) if supported else 0.0,
        "contradiction_rate": round(supported.count(1) / len(supported), 4) if supported else 0.0,
        "n_vignette_claims": n_vignette_claims,
        "n_vignette_mentioned": n_mentioned,
        "n_vignette_not_mentioned": n_not_mentioned,
        "coverage_rate": round(n_mentioned / n_vignette_claims, 4) if n_vignette_claims else 0.0,
        "unsupported_rate": round(len(unsupported) / total, 4) if total else 0.0,
    }
    for score in (1, 2, 3, 4):
        metrics[f"accuracy_{score}"] = supported.count(score)
        metrics[f"plausibility_{score}"] = unsupported.count(score)
    return metrics


def fidelity_metrics_markdown(metrics: dict[str, Any]) -> str:
    def share(count: int, total: int) -> str:
        return f"{count} ({count / total:.1%})" if total else "0"

    n_sup, n_unsup = metrics["n_supported"], metrics["n_unsupported"]
    return "\n".join([
        f"### Clinical fidelity metrics ({metrics['total_claims']} conversation claims)",
        "",
        "| Supported | | Unsupported | | Coverage | |",
        "|---|---|---|---|---|---|",
        f"| Claims | **{n_sup}** | Claims | **{n_unsup}** | Vignette claims | **{metrics['n_vignette_claims']}** |",
        f"| Mean accuracy | **{metrics['mean_accuracy']:.3f}** / 4 | Mean plausibility | **{metrics['mean_plausibility']:.3f}** / 4 "
        f"| Mentioned | **{metrics['coverage_rate']:.1%}** |",
        f"| Confirmed rate | **{metrics['confirmed_rate']:.1%}** | Unsupported rate | **{metrics['unsupported_rate']:.1%}** "
        f"| Not mentioned | **{metrics['n_vignette_not_mentioned']}** |",
        f"| Contradiction rate | **{metrics['contradiction_rate']:.1%}** | | | | |",
        "",
        "| | Score 1 | Score 2 | Score 3 | Score 4 |",
        "|---|---|---|---|---|",
        "| Supported (accuracy) | " + " | ".join(share(metrics[f"accuracy_{s}"], n_sup) for s in (1, 2, 3, 4)) + " |",
        "| Unsupported (plausibility) | " + " | ".join(share(metrics[f"plausibility_{s}"], n_unsup) for s in (1, 2, 3, 4)) + " |",
    ])


def _claim_status(answer: dict[str, Any] | None) -> str:
    if not answer:
        return "⬜"
    label = (ACCURACY if answer["category"] == "supported" else PLAUSIBILITY)[answer["score"]][0]
    support = f" ← V{answer['vignette_claim'] + 1}" if answer.get("vignette_claim") is not None else ""
    return f"✅ *{answer['category']}, {answer['score']} {label}{support}*"


def claims_markdown(record: dict[str, Any], answers: dict[int, dict[str, Any]], current: int) -> tuple[str, str]:
    vignette = "\n".join(f"{index + 1}. {claim}" for index, claim in enumerate(record["vignette_claims"]))
    conversation = "\n".join(
        f"{index + 1}. {'**' + claim + '**' if index == current else claim} {_claim_status(answers.get(index))}"
        for index, claim in enumerate(record["conversation_claims"])
    )
    return f"#### Vignette claims (V)\n{vignette}", f"#### Conversation claims (C)\n{conversation}"


def _saved_claim_answers(grader_id: str, record_id: str) -> dict[int, dict[str, Any]]:
    if not grader_id:
        return {}
    answers: dict[int, dict[str, Any]] = {}
    for row in read_rows("fidelity_claims"):
        if row["grader_id"] == grader_id and row["record_id"] == record_id:
            vignette_claim = row["supporting_vignette_claim_number"]
            answers[int(row["claim_number"]) - 1] = {
                "category": row["category"], "score": int(row["score"]),
                "vignette_claim": int(vignette_claim) - 1 if vignette_claim not in (None, "") else None,
            }
    return answers


def _numbered(prefix: str, claims: list[str]) -> list[tuple[str, int]]:
    return [(f"{prefix}{index + 1}. {claim}", index) for index, claim in enumerate(claims)]


def show_claim(record_id: str, answers: dict[int, dict[str, Any]], index: int) -> tuple[Any, ...]:
    """Outputs for the claim-rating panel: index, both claim lists, picker, claim text, and the three score inputs."""
    record = _by_id(fidelity_records(), record_id)
    claims = record.get("conversation_claims", [])
    index = max(0, min(index, len(claims) - 1))
    answer = answers.get(index, {})
    category = answer.get("category")
    vignette_md, conversation_md = claims_markdown(record, answers, index) if record else ("", "")
    return (
        index, vignette_md, conversation_md,
        gr.update(choices=_numbered("C", claims), value=index if claims else None),
        f"**{len(answers)} of {len(claims)} claims rated**\n\n### C{index + 1}\n> {claims[index] if claims else ''}",
        gr.update(value=category),
        gr.update(visible=category == "supported", value=answer.get("score") if category == "supported" else None),
        gr.update(visible=category == "supported", choices=_numbered("V", record.get("vignette_claims", [])),
                  value=answer.get("vignette_claim")),
        gr.update(visible=category == "unsupported", value=answer.get("score") if category == "unsupported" else None),
    )


def open_fidelity_record(grader_id: str, record_id: str) -> tuple[Any, ...]:
    """Outputs: answers, vignette, conversation, metrics, final-step group, final feedback, then show_claim outputs."""
    record = _by_id(fidelity_records(), record_id)
    if not record:
        message = ("Select your grader ID, then a record." if fidelity_records()
                   else "No clinical-fidelity data found. Run build_review_data.py first.")
        return ({}, message, [], "", gr.update(visible=False), "", *show_claim("", {}, 0))
    answers = _saved_claim_answers(_grader(grader_id), record["record_id"])
    first_open = next((index for index in range(len(record["conversation_claims"])) if index not in answers), 0)
    return (
        answers,
        f"{position_text('clinical_fidelity', grader_id, record['record_id'])}"
        f"**Record {record['record_id']} · case {record['case_id']}**\n\n{record['vignette_text']}",
        _chat_messages(record["transcript"]),
        "", gr.update(visible=False), "",
        *show_claim(record["record_id"], answers, first_open),
    )


def save_claim(
    answers: dict[int, dict[str, Any]], index: int, category: str | None, accuracy: int | None,
    vignette_claim: int | None, plausibility: int | None,
) -> tuple[dict[int, dict[str, Any]], str]:
    if category == "supported" and accuracy:
        # `is None`, not falsiness: V1 is index 0.
        if vignette_claim is None:
            return answers, "⚠️ Select the supporting vignette claim for this supported claim."
        answer = {"category": category, "score": int(accuracy), "vignette_claim": int(vignette_claim)}
    elif category == "unsupported" and plausibility:
        answer = {"category": category, "score": int(plausibility), "vignette_claim": None}
    else:
        return answers, "⚠️ Choose Supported or Unsupported and give the matching score."
    return {**answers, index: answer}, f"✅ C{index + 1} rated."


def _claim_row(grader_id: str, record: dict[str, Any], index: int, answer: dict[str, Any]) -> dict[str, Any]:
    labels = ACCURACY if answer["category"] == "supported" else PLAUSIBILITY
    vignette_claim = answer.get("vignette_claim")
    return {
        "grader_id": grader_id, "record_id": record["record_id"], "job_id": record["job_id"],
        "case_id": record["case_id"], "claim_number": index + 1, "claim": record["conversation_claims"][index],
        "category": answer["category"], "score": answer["score"], "score_label": labels[answer["score"]][0],
        "supporting_vignette_claim_number": vignette_claim + 1 if vignette_claim is not None else "",
        "supporting_vignette_claim": record["vignette_claims"][vignette_claim] if vignette_claim is not None else "",
    }


def persist_claim(grader_id: str, record_id: str, index: int, answer: dict[str, Any]) -> str:
    """Save one claim rating right away, so a closed tab or restarted server does not lose it."""
    grader_id = _grader(grader_id)
    if error := assignment_error("clinical_fidelity", grader_id, record_id, fidelity_records()):
        return error
    append_rows("fidelity_claims", [_claim_row(grader_id, _by_id(fidelity_records(), record_id), index, answer)])
    return ""


def submit_claims(grader_id: str, record_id: str, answers: dict[int, dict[str, Any]]) -> tuple[str, Any]:
    """Intermediary submission: persist every claim rating and open the missing-vignette-claims step."""
    grader_id = _grader(grader_id)
    if error := assignment_error("clinical_fidelity", grader_id, record_id, fidelity_records()):
        return error, gr.update()
    record = _by_id(fidelity_records(), record_id)
    claims = record["conversation_claims"]
    missing = [f"C{index + 1}" for index in range(len(claims)) if index not in answers]
    if missing:
        return f"⚠️ Rate every claim first. Still open: {', '.join(missing)}", gr.update()
    no_support = [f"C{index + 1}" for index, answer in sorted(answers.items())
                  if answer["category"] == "supported" and answer.get("vignette_claim") is None]
    if no_support:
        return f"⚠️ Select the supporting vignette claim for: {', '.join(no_support)}", gr.update()
    rows = [_claim_row(grader_id, record, index, answers[index]) for index in range(len(claims))]
    append_rows("fidelity_claims", rows)
    referenced = {answer["vignette_claim"] for answer in answers.values() if answer.get("vignette_claim") is not None}
    not_mentioned = [index for index in range(len(record["vignette_claims"])) if index not in referenced]
    return (
        f"✅ {len(rows)} claim ratings saved. Now confirm which vignette claims were never mentioned.",
        gr.update(choices=_numbered("V", record["vignette_claims"]), value=not_mentioned),
    )


def submit_fidelity(
    grader_id: str, record_id: str, answers: dict[int, dict[str, Any]], not_mentioned: list[int], comments: str,
) -> tuple[str, str]:
    """Final submission: compute and save the conversation-level fidelity metrics."""
    grader_id = _grader(grader_id)
    if error := assignment_error("clinical_fidelity", grader_id, record_id, fidelity_records()):
        return error, ""
    record = _by_id(fidelity_records(), record_id)
    metrics = compute_fidelity_metrics(list(answers.values()), len(record["vignette_claims"]), len(not_mentioned))
    append_rows("fidelity_summary", [{
        "grader_id": grader_id, "record_id": record["record_id"], "job_id": record["job_id"], "case_id": record["case_id"],
        **metrics, "not_mentioned_vignette_claims": ", ".join(f"V{index + 1}" for index in sorted(not_mentioned)),
        "comments": comments.strip(),
    }])
    return f"✅ Record {record['record_id']} submitted.", fidelity_metrics_markdown(metrics)


def _build_fidelity_tab(grader: gr.Textbox, status: gr.Markdown) -> tuple[Any, list[Any], gr.Dropdown]:
    """Build the tab; returns (load fn, its outputs, record dropdown) so the app can populate it on page load."""
    choices = [] if assignment() else _record_choices(fidelity_records())
    answers = gr.State({})
    index = gr.State(0)
    record_id = gr.Dropdown(choices=choices, value=choices[0][1] if choices else None, label="Your records")
    with gr.Row(equal_height=True):
        vignette = gr.Markdown(label="Source vignette", show_label=True, container=True, height=480)
        conversation = _transcript_view("Conversation")
    with gr.Row():
        vignette_claims = gr.Markdown(container=True, max_height=420)
        conversation_claims = gr.Markdown(container=True, max_height=420)
    with gr.Accordion("Scoring rubric", open=False):
        gr.Markdown(RUBRIC_TEXT)

    gr.Markdown("## Step 1 · Rate each conversation claim")
    claim_picker = gr.Dropdown(label="Claim under evaluation")
    claim_text = gr.Markdown()
    category = gr.Radio(
        [("Supported: a corresponding fact exists in the vignette", "supported"),
         ("Unsupported: not covered by the vignette", "unsupported")],
        label="Classification",
    )
    accuracy = gr.Radio([(f"{score}: {label}. {text}", score) for score, (label, text) in ACCURACY.items()],
                        label="Accuracy (supported claims)", visible=False)
    support = gr.Dropdown(label="Supporting vignette claim (required)", visible=False)
    plausibility = gr.Radio([(f"{score}: {label}. {text}", score) for score, (label, text) in PLAUSIBILITY.items()],
                            label="Plausibility (unsupported claims)", visible=False)
    with gr.Row():
        previous = gr.Button("← Previous claim")
        save = gr.Button("💾 Save claim & next →", variant="primary")
    claim_feedback = gr.Markdown()
    submit = gr.Button("Submit claim ratings", variant="primary", size="lg")

    with gr.Group(visible=False) as final_step:
        gr.Markdown("## Step 2 · Vignette claims never mentioned in the conversation\n"
                    "Pre-selected: every vignette claim you did not link to a supported conversation claim. Adjust if needed.")
        not_mentioned = gr.CheckboxGroup(label="Not mentioned", show_label=False)
        comments = gr.Textbox(label="Comments (optional)", lines=2)
        final_submit = gr.Button("✅ Final submit & compute metrics", variant="primary", size="lg")
    final_feedback = gr.Markdown()
    metrics = gr.Markdown()

    claim_outputs = [index, vignette_claims, conversation_claims, claim_picker, claim_text, category, accuracy, support, plausibility]
    record_outputs = [answers, vignette, conversation, metrics, final_step, final_feedback, *claim_outputs]

    def set_category(choice: str | None) -> tuple[Any, Any, Any]:
        return (gr.update(visible=choice == "supported"), gr.update(visible=choice == "supported"),
                gr.update(visible=choice == "unsupported"))

    def save_and_advance(grader_id, selected_record, current_answers, current, choice, acc, vignette_claim, plaus):
        updated, message = save_claim(current_answers, current, choice, acc, vignette_claim, plaus)
        if message.startswith("✅"):
            message = persist_claim(grader_id, selected_record, current, updated[current]) or message
        if not message.startswith("✅"):
            return current_answers, message, *[gr.update() for _ in claim_outputs], gr.update()
        n_claims = len(_by_id(fidelity_records(), selected_record)["conversation_claims"])
        next_open = next((i for i in [*range(current + 1, n_claims), *range(current)] if i not in updated), current)
        return updated, message, *show_claim(selected_record, updated, next_open), gr.update(visible=False)

    def final_submit_and_advance(grader_id, selected_record, current_answers, unmentioned, note):
        was_done = selected_record in _submitted_fidelity_records(_grader(grader_id))
        message, metrics_md = submit_fidelity(grader_id, selected_record, current_answers, unmentioned, note)
        dropdown = gr.update()
        if message.startswith("✅") and not was_done:
            # Record just finished: unlock the next one; its change event opens it.
            dropdown = record_dropdown("clinical_fidelity", grader_id)
            message += (" Opening your next record." if record_status("clinical_fidelity", grader_id)[2]
                        else " That was your last record.")
        return message, metrics_md, dropdown

    record_id.change(fn=open_fidelity_record, inputs=[grader, record_id], outputs=record_outputs)
    grader.change(fn=lambda g: record_dropdown("clinical_fidelity", g), inputs=grader, outputs=record_id).then(
        fn=open_fidelity_record, inputs=[grader, record_id], outputs=record_outputs,
    )
    category.change(fn=set_category, inputs=category, outputs=[accuracy, support, plausibility])
    claim_picker.input(fn=show_claim, inputs=[record_id, answers, claim_picker], outputs=claim_outputs)
    previous.click(fn=lambda r, a, i: show_claim(r, a, i - 1), inputs=[record_id, answers, index], outputs=claim_outputs)
    save.click(
        fn=save_and_advance, inputs=[grader, record_id, answers, index, category, accuracy, support, plausibility],
        outputs=[answers, claim_feedback, *claim_outputs, final_step],
    )
    submit.click(fn=submit_claims, inputs=[grader, record_id, answers], outputs=[claim_feedback, not_mentioned]).then(
        fn=lambda message: gr.update(visible=message.startswith("✅")), inputs=claim_feedback, outputs=final_step,
    )
    final_submit.click(
        fn=final_submit_and_advance, inputs=[grader, record_id, answers, not_mentioned, comments],
        outputs=[final_feedback, metrics, record_id],
    ).then(fn=progress_message, inputs=grader, outputs=[status])
    return open_fidelity_record, record_outputs, record_id


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Patient simulator validation") as demo:
        gr.Markdown(HEADER_TEXT)
        with gr.Row():
            # Fixed IDs when an assignment exists (a typo would silently orphan answers); free text otherwise.
            grader = gr.Dropdown(choices=grader_ids(), value=None, label="Grader ID",
                                 allow_custom_value=not grader_ids())
            status = gr.Markdown(progress_message(None))
        with gr.Tabs():
            with gr.Tab("Parameter adherence"):
                open_parameter, parameter_outputs, parameter_record, parameter_saved = _build_parameter_tab(grader, status)
            # Locked (with an assignment) until every Parameter adherence record is done.
            with gr.Tab("Clinical fidelity 🔒" if assignment() else "Clinical fidelity",
                        interactive=not assignment()) as fidelity_tab:
                open_fidelity, fidelity_outputs, fidelity_record = _build_fidelity_tab(grader, status)
        # Graders must not see each other's answers, so the shared deployment has no download.
        if not workbook_gcs_uri():
            gr.DownloadButton("Download workbook", value=export_workbook)
        grader.change(fn=progress_message, inputs=grader, outputs=status)
        grader.change(fn=fidelity_tab_update, inputs=grader, outputs=fidelity_tab)
        parameter_saved.then(fn=fidelity_tab_update, inputs=grader, outputs=fidelity_tab)
        demo.load(fn=open_parameter, inputs=[grader, parameter_record], outputs=parameter_outputs)
        demo.load(fn=open_fidelity, inputs=[grader, fidelity_record], outputs=fidelity_outputs)
    # Gradio runs one call per handler at a time by default; reads are cached and writes are serialised
    # by _WORKBOOK_LOCK, so parallel calls are safe.
    demo.queue(default_concurrency_limit=8)
    return demo


def main() -> None:
    restore_workbook()
    read_rows("parameter_adherence")  # parse the workbook now, not on the first grader's click
    demo = build_ui()
    if "PORT" not in os.environ:
        demo.launch(server_name="127.0.0.1", server_port=7860, theme=THEME, css=CSS)
        return
    # Hosted mode (PORT set): serve through FastAPI and answer /_ah/warmup (App Engine keep-warm requests).
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI()
    app.get("/_ah/warmup")(lambda: {"status": "ok"})
    app = gr.mount_gradio_app(app, demo, path="/", theme=THEME, css=CSS)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ["PORT"]))


if __name__ == "__main__":
    main()
