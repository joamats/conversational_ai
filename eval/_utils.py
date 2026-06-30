"""Shared utilities for the eval package."""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import types
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_json_response(raw: str) -> dict:
    """Parse a JSON object from an LLM response, with fallback extraction.

    Handles cases where the model wraps JSON in markdown code fences or
    includes surrounding text.
    """
    raw = raw.strip()
    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip(), strict=False)
        except json.JSONDecodeError:
            pass
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end], strict=False)
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class _FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes after every record (no buffering lag)."""
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(log_path: Path, *, console: bool = True) -> None:
    """Configure root logger with file + optional console output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [_FlushingFileHandler(log_path)]
    if console:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    for noisy in ("httpx", "httpcore", "openai", "langgraph", "langchain"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def split_config(config: dict) -> tuple[dict, dict, dict]:
    """Split merged config into (eval_config, doctor_config, judge_config)."""
    eval_config = {k: v for k, v in config.items() if k not in ("doctor", "judge")}
    return eval_config, config["doctor"], config["judge"]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def atomic_write_json(data: dict, path: Path) -> None:
    """Write JSON atomically via a temp file to avoid partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def format_transcript(transcript: list[dict]) -> str:
    """Format a doctor-patient transcript into numbered turns."""
    lines = []
    turn = 1
    i = 0
    while i < len(transcript):
        entry = transcript[i]
        if entry["role"] == "doctor":
            block = f"[Turn {turn}]\nDoctor: {entry['content']}"
            if i + 1 < len(transcript) and transcript[i + 1]["role"] == "patient":
                block += f"\nPatient: {transcript[i + 1]['content']}"
                i += 2
            else:
                i += 1
            lines.append(block)
            turn += 1
        else:
            i += 1
    return "\n\n".join(lines) if lines else "(empty transcript)"


def override_src_package(src_path: str) -> None:
    """Redirect ``src`` to *src_path*.

    Must be called **before** any module imports from that package.
    """
    src_dir = Path(src_path).resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"src path does not exist: {src_dir}")
    pkg_name = "src"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(src_dir)]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg
    logger.info("src package overridden to %s", src_dir)


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.5-2026-04-23":      {"input": 5.00, "cached": 0.50, "output": 30.00},
    "gpt-5.4-mini-2026-03-17":  {"input": 0.75, "cached": 0.075, "output": 4.50},
    "gpt-5.4-2026-03-25":       {"input": 2.50, "cached": 0.25, "output": 15.00},
    "claude-opus-4-6":          {"input": 5.00, "cached": 0.50, "output": 25.00},
    "gemini-3.5-flash":         {"input": 1.50, "cached": 0.00, "output": 9.00},
}
_DEFAULT_PRICING = {"input": 5.00, "cached": 0.50, "output": 30.00}


def compute_cost_from_results(results: list[dict]) -> dict:
    """Aggregate token usage and compute dollar cost from conversation results.

    Each result should have ``init_node_outputs`` and
    ``patient_node_outputs_per_turn`` as saved by ``run_single_job``.
    """
    totals: dict[str, dict[str, int]] = {}

    def _accum(node_name: str, envelope: dict) -> None:
        tok = envelope.get("token_usage", {})
        if not tok:
            return
        model_id = envelope.get("model_id", "unknown")
        key = f"{node_name}|{model_id}"
        if key not in totals:
            totals[key] = {
                "model_id": model_id,
                "node": node_name,
                "input_tokens": 0,
                "cached_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "calls": 0,
            }
        t = totals[key]
        t["input_tokens"] += tok.get("input_tokens", 0)
        t["cached_tokens"] += tok.get("cached_tokens", 0)
        t["output_tokens"] += tok.get("output_tokens", 0)
        t["reasoning_tokens"] += tok.get("reasoning_tokens", 0)
        t["calls"] += 1

    for r in results:
        if r.get("error"):
            continue
        for node_name, env in (r.get("init_node_outputs") or {}).items():
            _accum(node_name, env)
        for turn_outputs in (r.get("patient_node_outputs_per_turn") or []):
            for node_name, env in turn_outputs.items():
                _accum(node_name, env)
        for entry in (r.get("doctor_token_usage") or []):
            _accum(entry.get("node", "doctor"), entry)
        for entry in (r.get("judge_token_usage") or []):
            _accum(entry.get("node", "judge"), entry)

    rows = []
    grand = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
             "reasoning_tokens": 0, "calls": 0, "cost_usd": 0.0}

    for key in sorted(totals):
        t = totals[key]
        pricing = _PRICING.get(t["model_id"], _DEFAULT_PRICING)
        uncached = t["input_tokens"] - t["cached_tokens"]
        cost = (
            uncached * pricing["input"] / 1_000_000
            + t["cached_tokens"] * pricing["cached"] / 1_000_000
            + t["output_tokens"] * pricing["output"] / 1_000_000
        )
        row = {**t, "cost_usd": round(cost, 4)}
        rows.append(row)
        for k in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "calls"):
            grand[k] += t[k]
        grand["cost_usd"] += cost

    grand["cost_usd"] = round(grand["cost_usd"], 4)
    cache_rate = (
        grand["cached_tokens"] / grand["input_tokens"] * 100
        if grand["input_tokens"] > 0 else 0.0
    )

    return {
        "per_node": rows,
        "total": grand,
        "cache_hit_rate_pct": round(cache_rate, 1),
    }


def print_cost_summary(cost: dict) -> None:
    """Print a human-readable cost summary to stdout."""
    print("\n=== Cost Summary ===")
    print(f"{'Node':<30} {'Model':<25} {'Calls':>6} {'Input':>10} {'Cached':>10} {'Output':>10} {'Cost':>10}")
    print("-" * 105)
    for row in cost["per_node"]:
        print(
            f"{row['node']:<30} {row['model_id']:<25} {row['calls']:>6} "
            f"{row['input_tokens']:>10,} {row['cached_tokens']:>10,} "
            f"{row['output_tokens']:>10,} ${row['cost_usd']:>9.4f}"
        )
    t = cost["total"]
    print("-" * 105)
    print(
        f"{'TOTAL':<30} {'':<25} {t['calls']:>6} "
        f"{t['input_tokens']:>10,} {t['cached_tokens']:>10,} "
        f"{t['output_tokens']:>10,} ${t['cost_usd']:>9.4f}"
    )
    print(f"\nCache hit rate: {cost['cache_hit_rate_pct']:.1f}%")
    print(f"Total cost:     ${t['cost_usd']:.2f}")


def get_openai_api_key() -> str:
    """Return the OpenAI API key, fetching from Secret Manager if needed.

    Priority:
        1. OPENAI_API_KEY env var (already set)
        2. Google Cloud Secret Manager

    Sets OPENAI_API_KEY in the environment so subsequent OpenAI() clients
    pick it up automatically.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    try:
        from google.cloud import secretmanager  # type: ignore[import]

        project_id = os.environ.get("GCP_PROJECT_ID", "your-gcp-project")
        secret_name = os.environ.get("OPENAI_API_KEY_SECRET", "openai-api-key")
        name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        client = secretmanager.SecretManagerServiceClient()
        key = client.access_secret_version(name=name).payload.data.decode("UTF-8")
        os.environ["OPENAI_API_KEY"] = key
        logger.info("OpenAI API key loaded from Secret Manager.")
        return key
    except Exception as exc:
        logger.warning("Could not fetch API key from Secret Manager: %s", exc)

    raise RuntimeError(
        "No OpenAI API key found. Set the OPENAI_API_KEY environment variable "
        "or ensure Secret Manager access is available."
    )
