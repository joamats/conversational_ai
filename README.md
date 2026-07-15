# The complexities of patient-centred conversational artificial intelligence

Code accompanying the paper *The complexities of patient-centred conversational artificial intelligence*.

Paper: [arXiv:2607.08625](https://arxiv.org/abs/2607.08625)

This repository accompanies a study on patient-centred conversational AI for symptom assessment. A patient simulator is assessed for parameter adherence (whether its behaviour changes as configured), clinical fidelity (whether it stays faithful to the underlying case), and realism (a Turing-style test in which human and LLM graders try to tell simulated conversations from real ones). Using a set of patient personae across clinician-graded cases, we also measure how a patient's communication style affects the urgency (triage) assessments produced by LLMs.

## Setup

Requires Python ≥ 3.10. From this directory (`conversational_ai/`, the project root):

```bash
python -m venv .venv
source .venv/bin/activate    
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set your OpenAI key (used by the patient simulator and some judges):

```bash
export OPENAI_API_KEY="sk-..."
```

The LLM judges (Gemini / Claude) run on Vertex AI via application-default credentials. Run `gcloud auth application-default login` if you use them.

## Hardware requirements

Nothing special. Everything runs on a standard CPU machine (e.g. a laptop). No GPU is required, since all models are accessed through AI provider APIs. A few GB of RAM and disk for run outputs are sufficient. Runtime is dominated by API latency, so more CPU cores mainly help the parallel workers (`--max-workers`).

## Running

Run all commands **from this directory** so the modules resolve:

```bash
# Interactive demo — play the clinician, chat with the patient simulator
python -m src.terminal_demo

# Parameter-adherence eval
python -m eval.adherence.run

# Clinical-fidelity eval
python -m eval.fidelity.run

# Fairness experiment (personas)
python -m fairness.run --config fairness/configs/config_gpt_54_mini.json

# Turing test (human UI / LLM judge)
python -m eval.turing.llms.run --ui
python -m eval.turing.llms.run --llm-judge
```

Each folder's `README.md` lists the full options. Data-dependent runs expect the input data under `data/` (see the per-folder configs).

## Tests

Clinical Vignette data can be obtained by contacting authors from original publications.

Small synthetic sample datasets (5 rows each, matching the real schema) live under `tests/sample_synthetic_data/`. Tests confirm that every module works:

```bash
python -m pytest tests/test_data_pipeline.py     # or: python tests/test_data_pipeline.py
```

Regenerate the samples with `python tests/generate_sample_data.py`.


## Repository layout

| Folder | What's here |
|--------|-------------|
| `eda/` | Exploratory data analysis of the datasets. |
| `real/` | Analyses based on real conversations. |
| `eval/` | Evaluation suites of the patient simulator: `adherence/` (parameter adherence), `fidelity/` (clinical fidelity), `turing/` (human & LLM discrimination studies), `report/` (metric aggregation).|
| `doctor/` | The AI clinician / triage agent that converses with simulated patients. |
| `fairness/` | Fairness and subgroup analyses run over simulated personas, to assess the impact of communication style on urgency assessment. |
| `figures/` | Figures used in the paper. |
| `src/` | The patient simulator and its parameter schema (see `src/README.md`). |
| `tests/` | Smoke tests and synthetic sample data (`sample_synthetic_data/`). |

## Notes

Some scripts and inputs are derived from real user data and/or need proprietary patient simulator (for which only a synthetic sample or dummy has been shared)

## Citation


```bibtex
@article{patient_centred_conversational_ai,
  title   = {The complexities of patient-centred conversational artificial intelligence},
  author  = {João Matos, Olivia Buege, Donny Cheung, Gary S. Collins, Paula Dhiman, Nan Li, Bingyu Mao, Benjamin W. Nelson, Michail Ouroutzoglou, Paul Varghese, Jonathan Amar},
  year    = {2026},
  journal = {arXiv preprint arXiv:2607.08625},
  note    = {Preprint.},
  url     = {https://arxiv.org/abs/2607.08625}
}
```
