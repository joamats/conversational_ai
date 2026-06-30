"""Gradio UI for the human Turing test.

Presents doctor-patient conversation transcripts one at a time. The rater
classifies the *patient* as "AI Patient" or "Real Patient", optionally
providing reasoning. After all conversations are rated, the session is saved
to results/. No performance feedback is shown — the rater never learns
whether they were correct.

Rater assignments are pre-built by build_rater_assignments.py so that
human rater i and LLM rater i see the exact same conversations in the
exact same order.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

from eval.turing.preprocessing.build_rater_assignments import (
    DEFAULT_ASSIGNMENTS_PATH,
    load_rater_assignments,
    load_rater_items,
)
from eval.turing.preprocessing.data import (
    DEFAULT_REAL_PATH,
    load_conversations,
)

HUMANS_DIR = Path(__file__).parent
TURING_DIR = HUMANS_DIR.parent
RESULTS_DIR = HUMANS_DIR / "results"
_STATIC_DIR = HUMANS_DIR / "static"
_DOCTOR_AVATAR = str(_STATIC_DIR / "avatar_doctor.svg")
_PATIENT_AVATAR = str(_STATIC_DIR / "avatar_patient.svg")

N_CONVERSATIONS = 10

_state: dict = {
    "real_by_id": {},
    "sim_by_source": {},
    "n_raters": 0,
}


def _init_data() -> None:
    """Load real and simulated conversations into module state."""
    if _state["real_by_id"]:
        return
    data = load_rater_assignments()
    sim_path = data["simulated_path"]
    _state["n_raters"] = data["n_raters"]

    real = load_conversations(DEFAULT_REAL_PATH)
    _state["real_by_id"] = {c["id"]: c for c in real}

    simulated = load_conversations(sim_path)
    sim_by_source = {}
    for s in simulated:
        src = s.get("metadata", {}).get("source_session_id", s["id"])
        sim_by_source[src] = s
    _state["sim_by_source"] = sim_by_source


def _get_root_path(port: int = 7860) -> str | None:
    proxy_uri = os.environ.get("VSCODE_PROXY_URI", "")
    if proxy_uri:
        return proxy_uri.replace("{{port}}", str(port)).rstrip("/")
    return None


def _get_used_rater_indices() -> set[int]:
    """Return rater indices already used in saved sessions."""
    used: set[int] = set()
    if not RESULTS_DIR.exists():
        return used
    for path in RESULTS_DIR.glob("session_*.json"):
        try:
            with open(path) as f:
                s = json.load(f)
            idx = s.get("rater_index")
            if idx is not None:
                used.add(int(idx))
        except Exception:
            pass
    return used


def _transcript_to_chatbot(transcript: list[dict]) -> list[dict]:
    messages = []
    for turn in transcript:
        if turn["role"] == "doctor":
            messages.append({"role": "user", "content": turn["content"]})
        else:
            messages.append({"role": "assistant", "content": turn["content"]})
    return messages


def _save_session(session: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"session_{ts}_{session['session_id'][:8]}.json"
    with open(path, "w") as f:
        json.dump(session, f, indent=2, default=str)
    return path


def build_app() -> gr.Blocks:
    _init_data()

    with gr.Blocks(title="Patient Realism Test") as app:
        # ── State ────────────────────────────────────────────────────────
        test_items = gr.State([])
        current_idx = gr.State(0)
        ratings = gr.State([])
        session_id = gr.State("")
        start_time = gr.State(0.0)
        rater_index_state = gr.State(-1)

        # ── Start screen ────────────────────────────────────────────────
        with gr.Column(visible=True) as start_screen:
            gr.Markdown(
                "# Patient Realism Test\n\n"
                "You will read **doctor-patient conversation transcripts**. "
                "The doctor is always an AI. Your task is to determine whether "
                "the **patient** is an **AI simulator** or a **real person**.\n\n"
                f"This test has **{N_CONVERSATIONS} conversations**.\n\n"
                "For each conversation, classify the patient and optionally "
                "explain your reasoning."
            )
            rater_name = gr.Textbox(label="Your name", placeholder="Enter your name")
            rater_index_input = gr.Number(
                label="Rater index (0-15)",
                value=None,
                precision=0,
                minimum=0,
                maximum=_state["n_raters"] - 1,
            )
            name_validation_md = gr.Markdown("", visible=False)
            start_btn = gr.Button("Start Test", variant="primary", size="lg")

        # ── Test screen ─────────────────────────────────────────────────
        with gr.Column(visible=False) as test_screen:
            progress_md = gr.Markdown("")
            chatbot = gr.Chatbot(
                value=[],
                label="Conversation transcript",
                height=900,
                avatar_images=(_DOCTOR_AVATAR, _PATIENT_AVATAR),
            )
            classification_radio = gr.Radio(
                choices=["AI Patient", "Real Patient"],
                label="Is the patient an AI or a real person?",
                value=None,
            )
            feedback_box = gr.Textbox(
                label="Reasoning (optional)",
                placeholder="What made you choose this classification?",
                lines=2,
            )
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary", size="lg", scale=3)
                finish_early_btn = gr.Button("Finish Early", variant="stop", size="lg", scale=1)
            validation_md = gr.Markdown("", visible=False)

        # ── Done screen ─────────────────────────────────────────────────
        with gr.Column(visible=False) as done_screen:
            done_md = gr.Markdown("")
            saved_path_md = gr.Markdown("")
            restart_btn = gr.Button("Back to Start", variant="secondary", size="lg")

        # ── Callbacks ───────────────────────────────────────────────────

        def _error_return(msg):
            return (
                gr.update(), gr.update(), gr.update(),
                [], 0, [], "", 0.0, -1,
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(visible=False),
                gr.update(visible=True, value=msg),
            )

        def on_start(name, rater_idx):
            name_clean = name.strip() if name else ""
            if not name_clean:
                return _error_return("**Please enter your name.**")

            if rater_idx is None or rater_idx < 0 or rater_idx >= _state["n_raters"]:
                return _error_return(
                    f"**Please enter a rater index between 0 and {_state['n_raters'] - 1}.**"
                )

            rater_idx = int(rater_idx)
            used = _get_used_rater_indices()
            if rater_idx in used:
                return _error_return(
                    f"**Rater index {rater_idx} has already been used. Please choose another.**"
                )

            items = load_rater_items(
                rater_idx,
                _state["real_by_id"],
                _state["sim_by_source"],
            )
            sid = str(uuid.uuid4())
            chat_messages = _transcript_to_chatbot(items[0]["transcript"])
            progress = f"**Conversation 1 / {len(items)}**"
            return (
                gr.update(visible=False),           # start_screen
                gr.update(visible=True),            # test_screen
                gr.update(visible=False),           # done_screen
                items, 0, [], sid, time.time(),     # state
                rater_idx,                          # rater_index_state
                progress,                           # progress_md
                chat_messages,                      # chatbot
                None,                               # classification_radio
                "",                                 # feedback_box
                gr.update(visible=False, value=""), # validation_md
                gr.update(visible=False, value=""), # name_validation_md
            )

        def on_submit(choice, items, idx, current_ratings, sid, t0, feedback, name, r_idx):
            if not choice:
                return (
                    gr.update(), gr.update(),
                    idx, current_ratings, t0,
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(visible=True, value="**Please select AI Patient or Real Patient before submitting.**"),
                    gr.update(), gr.update(),
                )

            classification = "simulated" if choice == "AI Patient" else "real"
            item = items[idx]

            current_ratings = list(current_ratings)
            ground_truth = item["_ground_truth"]
            correct = classification == ground_truth
            current_ratings.append({
                "conversation_id": item["id"],
                "pair_id": item.get("_pair_id", ""),
                "ground_truth": ground_truth,
                "classification": classification,
                "classified_simulated": classification == "simulated",
                "correct": correct,
                "feedback": feedback.strip() if feedback else "",
                "time_seconds": round(time.time() - t0, 1),
                "presentation_order": idx + 1,
                "rater_index": int(r_idx) if r_idx is not None else None,
            })

            next_idx = idx + 1

            if next_idx >= len(items):
                done_text, saved = _finalize(current_ratings, sid, t0, name, r_idx)
                return (
                    gr.update(visible=False),           # test_screen
                    gr.update(visible=True),            # done_screen
                    next_idx, current_ratings, time.time(),
                    "",                                 # progress_md
                    [],                                 # chatbot
                    None,                               # classification_radio
                    "",                                 # feedback_box
                    gr.update(visible=False, value=""), # validation_md
                    done_text,
                    f"Session saved to `{saved}`",
                )

            next_item = items[next_idx]
            chat_messages = _transcript_to_chatbot(next_item["transcript"])
            progress = f"**Conversation {next_idx + 1} / {len(items)}**"
            return (
                gr.update(visible=True),            # test_screen
                gr.update(visible=False),           # done_screen
                next_idx, current_ratings, time.time(),
                progress,
                chat_messages,
                None,
                "",
                gr.update(visible=False, value=""),
                "",
                "",
            )

        def _finalize(current_ratings, sid, t0, name, r_idx):
            n = len(current_ratings)
            correct = sum(1 for r in current_ratings if r["correct"])
            accuracy = correct / n if n else 0

            tp = sum(1 for r in current_ratings if r["ground_truth"] == "simulated" and r["classification"] == "simulated")
            tn = sum(1 for r in current_ratings if r["ground_truth"] == "real" and r["classification"] == "real")
            fp = sum(1 for r in current_ratings if r["ground_truth"] == "real" and r["classification"] == "simulated")
            fn = sum(1 for r in current_ratings if r["ground_truth"] == "simulated" and r["classification"] == "real")

            report = {
                "total": n,
                "correct": correct,
                "accuracy": accuracy,
                "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            }
            session = {
                "session_id": sid,
                "rater": name.strip() if name else "anonymous",
                "rater_index": int(r_idx) if r_idx is not None else None,
                "started_at": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "judge_type": "human",
                "report": report,
                "ratings": current_ratings,
            }
            saved = _save_session(session)

            done_text = (
                f"## Thank you!\n\n"
                f"You rated **{n}** conversations.\n\n"
                f"### Results\n\n"
                f"**Accuracy: {accuracy:.0%}** ({correct}/{n})\n\n"
                f"If accuracy is near 50%, the simulator is indistinguishable from real patients.\n\n"
                f"### Confusion Matrix\n\n"
                f"|  | Classified AI | Classified Real |\n"
                f"|---|---|---|\n"
                f"| **Actually AI** | {tp} | {fn} |\n"
                f"| **Actually Real** | {fp} | {tn} |"
            )
            return done_text, saved

        def on_finish_early(current_ratings, sid, t0, name, r_idx):
            if not current_ratings:
                return (
                    gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(visible=True, value="**Rate at least one conversation before finishing.**"),
                    gr.update(), gr.update(),
                )

            done_text, saved = _finalize(current_ratings, sid, t0, name, r_idx)
            return (
                gr.update(visible=False),           # test_screen
                gr.update(visible=True),            # done_screen
                "",                                 # progress_md
                [],                                 # chatbot
                None,                               # classification_radio
                "",                                 # feedback_box
                gr.update(visible=False, value=""), # validation_md
                done_text,
                f"Session saved to `{saved}`",
            )

        def on_restart():
            return (
                gr.update(visible=True),            # start_screen
                gr.update(visible=False),           # test_screen
                gr.update(visible=False),           # done_screen
                "",                                 # rater_name cleared
                gr.update(visible=False, value=""), # name_validation_md
            )

        # ── Wire callbacks (queue=False to avoid SSE) ───────────────────

        start_outputs = [
            start_screen, test_screen, done_screen,
            test_items, current_idx, ratings, session_id, start_time,
            rater_index_state,
            progress_md, chatbot, classification_radio, feedback_box,
            validation_md, name_validation_md,
        ]
        start_btn.click(
            on_start,
            inputs=[rater_name, rater_index_input],
            outputs=start_outputs,
            queue=False,
        )

        submit_outputs = [
            test_screen, done_screen,
            current_idx, ratings, start_time,
            progress_md, chatbot, classification_radio, feedback_box, validation_md,
            done_md, saved_path_md,
        ]
        submit_inputs = [
            classification_radio,
            test_items, current_idx, ratings, session_id, start_time,
            feedback_box, rater_name, rater_index_state,
        ]
        submit_btn.click(on_submit, inputs=submit_inputs, outputs=submit_outputs, queue=False)

        finish_early_outputs = [
            test_screen, done_screen,
            progress_md, chatbot, classification_radio, feedback_box, validation_md,
            done_md, saved_path_md,
        ]
        finish_early_inputs = [ratings, session_id, start_time, rater_name, rater_index_state]
        finish_early_btn.click(on_finish_early, inputs=finish_early_inputs, outputs=finish_early_outputs, queue=False)

        restart_outputs = [
            start_screen, test_screen, done_screen,
            rater_name, name_validation_md,
        ]
        restart_btn.click(on_restart, inputs=[], outputs=restart_outputs, queue=False)

    return app


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    data = load_rater_assignments()
    print(f"Using simulated data from: {data['simulated_path']}")
    print(f"Rater slots available: 0-{data['n_raters'] - 1}")

    root_path = _get_root_path(args.port)
    app = build_app()
    app.launch(
        server_port=args.port,
        share=args.share,
        root_path=root_path,
        ssr_mode=False,
        theme=gr.themes.Soft(),
        allowed_paths=[str(_STATIC_DIR)],
    )


if __name__ == "__main__":
    main()
