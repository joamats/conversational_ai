# `src/` — Patient simulator

A lightweight, single-file implementation of the patient simulator (which is proprietary and cannot be released). It exposes the same public interface and parameter set as the version used in the paper, generating each patient turn with a single GPT-5.5 call, so the rest of the
codebase (`eval/`, `fairness/`, `doctor/`) runs against it unchanged.

Set an OpenAI key via the `OPENAI_API_KEY` environment variable.

## Files

| File | Role |
|------|------|
| `patient_simulator.py` | `PatientSimulator`, `SimulatorConfig`, the schema loader, and public constants. |
| `param_schema.json` | The full parameter schema (names, categories, defaults). |
| `terminal_demo.py` | Interactive demo — chat with the patient simulator in the terminal. |
| `demo_config.json` | Example patient config for the demo. |
