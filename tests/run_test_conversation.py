"""Scripted doctor-patient conversation for testing the PatientSimulator."""
import os
import sys
from pathlib import Path

# Ensure the conversational_ai root is on sys.path when run as a script.
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# Resolve the OpenAI API key: prefer the OPENAI_API_KEY env var; otherwise fall
# back to Google Cloud Secret Manager (configure the project/secret below).
if not os.environ.get("OPENAI_API_KEY"):
    from google.cloud import secretmanager

    project_id = os.environ.get("GCP_PROJECT_ID", "your-gcp-project")
    secret_name = os.environ.get("OPENAI_API_KEY_SECRET", "openai-api-key")
    name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    secret_client = secretmanager.SecretManagerServiceClient()
    openai_key = secret_client.access_secret_version(name=name).payload.data.decode("UTF-8")
    os.environ["OPENAI_API_KEY"] = openai_key

import json
import logging

_DIR = Path(__file__).parent
_LOG_FILE = _DIR / "session.log"
_STATE_FILE = _DIR / "results" / "state.jsonl"


class _FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


# Set up logging — flushes to disk after every record so you can tail -f session.log
_root = logging.getLogger()
_root.setLevel(logging.DEBUG)
_fh = _FlushingFileHandler(_LOG_FILE, mode="w", encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
))
_root.addHandler(_fh)
print(f"Logging to: {_LOG_FILE}")


def _load_model_map(path):
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for node, v in raw.items():
        if isinstance(v, dict):
            result[node] = v
        else:
            result[node] = {"model_id": v[0], "reasoning_effort": v[1], "prompt_id": "default"}
    return result

# Load config + model map
with open(_DIR.parent / "src" / "demo_config.json") as f:
    config = json.load(f)

model_map = None

from src.patient_simulator import PatientSimulator

inp = config["input"]
persona = config.get("persona_params", {})
content = config.get("content_params", {})
comm = config.get("communication_params", {})
strategy = config.get("strategy_params", {})
orch = config.get("orchestrator_params", {})
tech = config["technical"]

print("Initialising patient simulator...")
sim = PatientSimulator(
    vignette=inp.get("vignette"),
    ehr_json=inp.get("ehr_json"),
    prior_conversation=inp.get("prior_conversation"),
    environment=inp.get("environment"),
    setting=inp.get("setting", "general"),
    task=inp.get("task", "general"),
    socioeconomic_context=persona.get("socioeconomic_context"),
    cultural_context=persona.get("cultural_context"),
    recall_level=content.get("recall_level", "normal"),
    oversharing_tendency=content.get("oversharing_tendency", "no"),
    topic_sensitivity=content.get("topic_sensitivity", []),
    health_literacy=comm.get("health_literacy", "medium"),
    verbosity=comm.get("verbosity", "moderate"),
    communication_style=comm.get("communication_style", "formal"),
    english_proficiency=comm.get("english_proficiency", "native"),
    profanity_level=comm.get("profanity_level", "no"),
    typo_rate=comm.get("typo_rate", "no"),
    abbreviation_tendency=comm.get("abbreviation_tendency", "no"),
    punctuation_style=comm.get("punctuation_style", "standard"),
    capitalisation=comm.get("capitalisation", "standard"),
    gives_up=strategy.get("gives_up", False),
    max_turns=strategy.get("max_turns"),
    enable_feelings=orch.get("enable_feelings", True),
    enable_strategy=orch.get("enable_strategy", True),
    model_map=model_map,
)

print(f"Chief complaint: {sim.patient_profile.get('chief_complaint')}")
print()

# Reset state file and write init entry (contains ingest system prompt)
_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
_STATE_FILE.write_text("")
with open(_STATE_FILE, "a", encoding="utf-8") as _sf:
    _sf.write(json.dumps({
        "turn": 0,
        "phase": "init",
        "patient_profile": sim.patient_profile,
        "node_outputs": sim.init_node_outputs,
    }, default=str) + "\n")
print(f"State saved to: {_STATE_FILE}")
print()

DOCTOR_TURNS = [
    "Hi there, I can see you've reached out tonight. What's going on?",
    "Where exactly is the pain — can you point to it? And does it shoot anywhere, like down into your legs or buttocks?",
    "Any numbness, tingling, or weakness in your legs at all? And have you had any issues with your bladder or bowels?",
    "When did this come on — was it sudden or did it build up through the day? And have you ever had pain like this before?",
    "What do you do for work, and what were you doing today when it started?",
    "Are you on any medications at the moment? Anything for blood pressure or diabetes perhaps?",
    "Have you taken anything for the pain tonight — ibuprofen or paracetamol?",
    "Based on everything you've told me, this sounds very much like mechanical lower back pain — muscle strain from the physical demands of the day. There are no red flags here: no leg weakness, no numbness, no bladder or bowel problems. You should absolutely be able to go to work tomorrow. I'd recommend taking ibuprofen tonight with food, applying a heat pack to your lower back, and trying to keep moving gently rather than resting in bed. Does that make sense?",
]

SEP = "─" * 70

for i, doctor_msg in enumerate(DOCTOR_TURNS, 1):
    print(f"{SEP}")
    print(f"[Turn {i}] DOCTOR: {doctor_msg}")
    print(f"{SEP}")

    result = sim.chat(doctor_msg)
    response = result["response"]
    stopped = result.get("stopped", False)
    node_outputs = result["node_outputs"]

    display_response = response.replace("<<CONVERSATION_END>>", "").strip()
    print(f"PATIENT: {display_response}")
    print()

    # Token summary
    total_tokens = sum(
        n.get("token_usage", {}).get("total_tokens", 0) for n in node_outputs.values()
    )
    total_reasoning = sum(
        n.get("token_usage", {}).get("reasoning_tokens", 0) for n in node_outputs.values()
    )
    slowest = max(node_outputs.items(), key=lambda x: x[1].get("latency_ms", 0))
    print(f"  tokens: {total_tokens} total ({total_reasoning} reasoning)  |  slowest node: {slowest[0]} ({slowest[1]['latency_ms']:.0f}ms)")
    print()

    # Append turn state to JSONL file
    turn_entry = {
        "turn": i,
        "doctor": doctor_msg,
        "patient": display_response,
        "stopped": stopped,
        "patient_profile": result.get("patient_profile", {}),
        "node_outputs": node_outputs,
    }
    with open(_STATE_FILE, "a", encoding="utf-8") as _sf:
        _sf.write(json.dumps(turn_entry, default=str) + "\n")

    if stopped:
        print(f"{SEP}")
        print("Patient ended the conversation.")
        break

print(f"{SEP}")
print("Conversation complete.")
print(f"Full log: {_DIR / 'session.log'}")
