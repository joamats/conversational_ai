"""Terminal demo: play as a doctor and chat with a patient simulator.

Usage:
    python -m src.terminal_demo
    python -m src.terminal_demo --config path/to/config.json
    python -m src.terminal_demo --config path/to/config.json --verbose
    python -m src.terminal_demo --config path/to/config.json --show-nodes
    python -m src.terminal_demo --log-file session.log

Requires an OpenAI API key: set the OPENAI_API_KEY environment variable
(otherwise the demo falls back to Google Cloud Secret Manager).

Config JSON format: see demo_config.json in this directory.

Log file contents (when --log-file is set):
  - Per-node: inputs, model used, token counts (including reasoning tokens), latency, full output
  - Per-turn: complete intermediate state (content/feelings/strategy signals,
    orchestrator raw_intent, final_response)
  - Init: full patient_profile
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

_DIR = Path(__file__).parent


class _FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes to disk after every record so the log is readable in real time."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def _setup_logging(log_file: str | None, verbose: bool) -> None:
    """Configure logging handlers.

    - If --verbose: INFO-level logs also appear on stderr (separate from the conversation).
    - If --log-file: ALL logs flush to disk immediately after every record.
    - DEBUG level (full structured outputs) only goes to the file.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # let handlers decide their own level

    fmt_detailed = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fmt_simple = logging.Formatter("%(asctime)s %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")

    if verbose:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt_simple)
        root.addHandler(ch)

    if log_file:
        fh = _FlushingFileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt_detailed)
        root.addHandler(fh)


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def _set_openai_api_key_from_secret_manager() -> None:
    """Fetch the OpenAI API key from Secret Manager and set it as OPENAI_API_KEY env var."""
    from google.cloud import secretmanager  # type: ignore[import]

    project_id = os.environ.get("GCP_PROJECT_ID", "your-gcp-project")
    secret_name = os.environ.get("OPENAI_API_KEY_SECRET", "openai-api-key")
    name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    secret_client = secretmanager.SecretManagerServiceClient()
    openai_key = secret_client.access_secret_version(name=name).payload.data.decode("UTF-8")
    os.environ["OPENAI_API_KEY"] = openai_key


def _load_model_map(path: str) -> dict:
    """Load a model_map JSON file. Supports both new dict format and legacy [model_id, reasoning_effort] lists."""
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for node, v in raw.items():
        if isinstance(v, dict):
            result[node] = v
        else:
            result[node] = {"model_id": v[0], "reasoning_effort": v[1], "prompt_id": "default"}
    return result


def _flatten_config(config: dict) -> dict:
    """Flatten nested demo_config.json groups into a single dict for SimulatorConfig."""
    flat: dict = {}
    for group in [
        "input", "persona_params", "content_params", "communication_params",
        "strategy_params", "orchestrator_params", "technical",
    ]:
        flat.update(config.get(group, {}))
    return flat


def _build_simulator(config: dict, model_map: dict | None, verbose: bool):
    from src.config_wrapper import SimulatorConfig
    from src.patient_simulator import PatientSimulator

    flat = _flatten_config(config)

    sim_config = SimulatorConfig(
        model_map=model_map,
        verbose=verbose,
        thread_id=flat.get("thread_id"),
        **{k: v for k, v in flat.items() if k not in ("thread_id", "model_map_file")},
    )

    try:
        sim_config.validate()
    except ValueError as exc:
        print()
        print("Configuration error:")
        print(str(exc))
        print()
        sys.exit(1)

    return PatientSimulator(**sim_config.to_simulator_kwargs())


def _print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


def _print_init_summary(sim) -> None:
    profile = sim.patient_profile

    print()
    _print_separator("═")
    print("  PATIENT SIMULATOR — ready")
    _print_separator("═")
    print(f"  Chief complaint      : {profile.get('chief_complaint', '—')}")
    print(f"  Age / sex            : {profile.get('age', '—')} / {profile.get('sex', '—')}")
    print(f"  Positive symptoms    : {', '.join(profile.get('present_illness_positive', [])) or '—'}")
    print(f"  Negative symptoms    : {', '.join(profile.get('present_illness_negative', [])) or '—'}")
    _print_separator("═")
    print()


def _print_node_outputs(node_outputs: dict) -> None:
    _print_separator("·")
    print("  NODE OUTPUTS")
    _print_separator("·")
    for node_name, envelope in node_outputs.items():
        tok = envelope.get("token_usage", {})
        lat = envelope.get("latency_ms", 0)
        total = tok.get("total_tokens", 0)
        print(f"  {node_name:<35}  {total:>5} tokens  {lat:>8.0f} ms")
    _print_separator("·")
    print()



def _wrap(text: str, width: int = 70, indent: str = "  ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _append_turn_to_state_file(path: str, entry: dict[str, Any]) -> None:
    """Append a turn entry as a single JSON line to the JSONL state file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Terminal demo: chat with a patient simulator as a doctor."
    )
    parser.add_argument(
        "--config",
        default=str(_DIR / "demo_config.json"),
        help="Path to config JSON file (default: demo_config.json in this directory)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print INFO-level logs to stderr during the session",
    )
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        default=str(_DIR / "session.log"),
        help=(
            "Save all logs to this file (default: session.log in this directory). "
            "Captures per-node inputs/outputs, token counts (including reasoning tokens), "
            "latency, and full intermediate state after every turn."
        ),
    )
    parser.add_argument(
        "--show-nodes",
        action="store_true",
        help="Print per-node token usage and latency after each turn",
    )
    parser.add_argument(
        "--state-file",
        metavar="PATH",
        default=str(_DIR / "state.jsonl"),
        help=(
            "Save full per-turn state to this JSONL file after every turn "
            "(default: state.jsonl in this directory). "
            "One JSON object per line, each containing patient_profile, "
            "node outputs, and final system prompts for that turn."
        ),
    )
    args = parser.parse_args()

    _setup_logging(log_file=args.log_file, verbose=args.verbose)

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print("Create a config JSON file or specify one with --config.")
        sys.exit(1)

    config = _load_config(args.config)

    if args.log_file:
        print(f"Logging full session to: {args.log_file}")

    # Resolve the OpenAI API key: prefer OPENAI_API_KEY; else fall back to Secret Manager.
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — fetching from Secret Manager...")
        _set_openai_api_key_from_secret_manager()

    # Load model map from file if specified in config
    model_map: dict | None = None
    model_map_file = config.get("technical", {}).get("model_map_file")
    if model_map_file:
        print(f"Loading model map from: {model_map_file}")
        model_map = _load_model_map(model_map_file)

    print()
    print("Initialising patient simulator...")
    sim = _build_simulator(config, model_map=model_map, verbose=args.verbose)
    _print_init_summary(sim)

    # Reset state file and write init entry (contains ingest system prompt)
    open(args.state_file, "w").close()
    with open(args.state_file, "a", encoding="utf-8") as _sf:
        _sf.write(json.dumps({
            "turn": 0,
            "phase": "init",
            "patient_profile": sim.patient_profile,
            "node_outputs": sim.init_node_outputs,
        }, default=str) + "\n")
    print(f"State saved to: {args.state_file}")
    print()

    print("Type your message as the doctor. Type 'quit' or press Ctrl-C to exit.")
    print()

    turn = 0
    while True:
        try:
            user_input = input("Doctor: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Session ended.")
            break

        turn += 1
        print()

        result_box: list = []
        exc_box: list = []

        def _run():
            try:
                result_box.append(sim.chat(user_input))
            except Exception as e:
                exc_box.append(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        start = time.perf_counter()
        while t.is_alive():
            elapsed = time.perf_counter() - start
            print(f"\r  [thinking...  {elapsed:.1f}s]", end="", flush=True)
            time.sleep(0.1)
        elapsed = time.perf_counter() - start
        print(f"\r  [done in {elapsed:.1f}s]          ")

        if exc_box:
            raise exc_box[0]
        result = result_box[0]

        print()
        _print_separator()
        # Strip the stop token from display if present
        display_response = result["response"].replace("<<CONVERSATION_END>>", "").strip()
        print(_wrap(display_response))
        _print_separator()
        print()

        if args.show_nodes:
            _print_node_outputs(result["node_outputs"])

        _append_turn_to_state_file(args.state_file, {
            "turn": turn,
            "doctor": user_input,
            "patient": display_response,
            "stopped": result.get("stopped", False),
            "patient_profile": result.get("patient_profile", {}),
            "node_outputs": result["node_outputs"],
        })

        if result.get("stopped"):
            print("  [Patient ended the conversation.]")
            print()
            break


if __name__ == "__main__":
    main()
