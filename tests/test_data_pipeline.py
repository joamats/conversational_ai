"""Smoke tests that the data loaders, dataset builders, and the patient
simulator work end-to-end on the synthetic sample data in tests/sample_synthetic_data/.

These exercise the real functions used in the paper's runs (loaders in
eval/ and fairness/, plus the PatientSimulator construction) without needing
the real (not public) data, an API key, or network access.

Run:
    python -m pytest tests/test_data_pipeline.py
    python tests/test_data_pipeline.py        # also works as a plain script
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the conversational_ai root importable regardless of how this is invoked.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DATA = Path(__file__).parent / "sample_synthetic_data"
_AV = str(_DATA / "av" / "av_clean.csv")
_HV = str(_DATA / "hv" / "hv_clean.csv")
_REAL = str(_DATA / "real" / "real_clean.csv")
_SIM_RAW = str(_DATA / "turing" / "simulated_raw.jsonl")

_URGENCY = {"self-care", "non-urgent", "urgent"}

# Entry modules a user runs via `python -m ...` — these should all wire up.
_ENTRY_MODULES = [
    "src.patient_simulator", "src.config_wrapper", "src.nodes", "src.param_prompts",
    "src.terminal_demo", "eval._utils", "eval.runner", "eval.splits",
    "eval.adherence.run", "eval.adherence.dataset", "eval.adherence.metrics", "eval.adherence.judge",
    "eval.fidelity.run", "eval.fidelity.extractor",
    "fairness.run", "fairness.simulation", "fairness.dataset", "doctor.doctor",
    "eval.turing.llms.run", "eval.turing.llms.llm_judge", "eval.turing.llms.judge_clients",
    "eval.turing.llms.sequential_judge", "eval.turing.preprocessing.pipeline",
    "eval.turing.preprocessing.extract", "eval.turing.preprocessing.build_real",
    "eval.turing.preprocessing.build_splits", "eval.turing.preprocessing.build_rater_assignments",
    "eval.turing.preprocessing.normalize", "eval.turing.preprocessing.data",
    "eval.turing.humans.app",
]
# Third-party deps that aren't required for the core pipeline.
_OPTIONAL_DEPS = {"gradio"}


def _ensure_samples() -> None:
    if not all(Path(p).exists() for p in (_AV, _HV, _REAL, _SIM_RAW)):
        from tests.generate_sample_data import main as gen
        gen()


def test_entry_modules_import():
    """Every runnable module imports cleanly (the whole package wires up)."""
    import importlib

    for m in _ENTRY_MODULES:
        try:
            importlib.import_module(m)
        except ModuleNotFoundError as e:
            if (e.name or "").split(".")[0] in _OPTIONAL_DEPS:
                continue  # optional dep (e.g. gradio) not installed
            raise


def test_load_vignettes():
    from eval.adherence.dataset import load_vignettes

    vignettes = load_vignettes(_AV, n=5, seed=42)
    assert len(vignettes) == 5
    for v in vignettes:
        assert v["case_id"]
        assert v["vignette_text"].strip()


def test_load_av_cases():
    from fairness.dataset import load_av_cases

    cases = load_av_cases(_AV)
    assert len(cases) == 5
    for c in cases:
        assert c["dataset_name"] == "av"
        assert c["gold_triage"] in _URGENCY
        assert "demographics" in c["ehr_json"] and c["ehr_json"]["chief_complaint"]
        assert isinstance(c["prior_conversation"], list) and c["prior_conversation"]


def test_load_hv_cases():
    from fairness.dataset import load_hv_cases

    cases = load_hv_cases(_HV)
    assert len(cases) == 5
    for c in cases:
        assert c["dataset_name"] == "hv"
        assert c["gold_triage"] in _URGENCY
        assert c["ehr_json"]["chief_complaint"]


def test_adherence_build_dataset():
    from eval.adherence.dataset import build_dataset
    from src.param_prompts import SCHEMA

    cfg = {
        "data_path": _AV,
        "n_vignettes": 5,
        "seed": 42,
        "comparative_params_to_eval": ["anxiety", "trust"],
        "verbosity_params_to_eval": [],
        "max_turns_params_to_eval": [],
        "probing_params_to_eval": [],
        "patient_goal_eval": False,
        "sensitive_topics_eval": False,
    }
    dataset = build_dataset(cfg, SCHEMA)
    assert dataset["stats"]["vignettes_count"] == 5
    assert dataset["stats"]["total_jobs"] > 0
    # every job carries the params the simulator expects
    assert all("all_params" in j and "vignette_text" in j for j in dataset["jobs"])


def test_fairness_build_jobs():
    from fairness.dataset import DEFAULT_URGENCY_LEVELS, build_jobs, load_all_cases, load_personas

    datasets = [
        {"name": "av", "type": "av", "path": _AV, "urgency_levels": DEFAULT_URGENCY_LEVELS},
        {"name": "hv", "type": "hv", "path": _HV, "urgency_levels": DEFAULT_URGENCY_LEVELS},
    ]
    cases = load_all_cases(datasets)
    assert len(cases) == 10  # 5 av + 5 hv

    conditions = load_personas(str(_ROOT / "fairness" / "personas.json"))
    jobs = build_jobs(cases, conditions, {"eval": {"n_repetitions": 1, "seed": 42}})
    assert len(jobs) == len(cases) * len(conditions)
    for j in jobs:
        assert j["gold_triage"] in _URGENCY
        assert "sim_params" in j


def test_simulator_accepts_sample_params():
    """PatientSimulator should construct from a sample vignette (no API call)."""
    from fairness.dataset import load_av_cases
    from src.config_wrapper import SimulatorConfig
    from src.patient_simulator import PatientSimulator

    case = load_av_cases(_AV)[0]
    cfg = SimulatorConfig(
        vignette=case["patient_vignette"],
        ehr_json=case["ehr_json"],
        anxiety="anxious",
        trust="sceptical",
    )
    cfg.validate()
    sim = PatientSimulator(**cfg.to_simulator_kwargs())
    assert sim.patient_profile.get("chief_complaint")


def test_fidelity_build_dataset():
    from eval.fidelity.run import _build_fidelity_dataset

    ds = _build_fidelity_dataset({"data_path": _AV, "n_vignettes": 5, "seed": 42})
    assert ds["stats"]["vignettes_count"] == 5
    assert ds["stats"]["total_jobs"] == 5
    assert all("all_params" in j and "vignette_text" in j for j in ds["jobs"])


def test_turing_build_real():
    import tempfile
    from eval.turing.preprocessing.build_real import build_real_jsonl

    out = Path(tempfile.mkdtemp()) / "real.jsonl"
    convos = build_real_jsonl(csv_path=_REAL, output_path=out)
    assert out.exists() and len(convos) >= 1
    for c in convos:
        assert c["id"] and isinstance(c["transcript"], list) and c["transcript"]


def test_turing_normalize():
    import tempfile
    from eval.turing.preprocessing.normalize import run_normalization

    out = Path(tempfile.mkdtemp()) / "simulated.jsonl"
    convos = run_normalization(raw_path=_SIM_RAW, output_path=out)
    assert out.exists() and len(convos) >= 1
    for c in convos:
        assert c["id"] and c["transcript"]


def test_turing_build_splits():
    import tempfile
    from eval.turing.preprocessing.build_real import build_real_jsonl
    from eval.turing.preprocessing.build_splits import build_splits

    d = Path(tempfile.mkdtemp())
    real = d / "real.jsonl"
    build_real_jsonl(csv_path=_REAL, output_path=real)
    splits = build_splits(
        real_path=real, output_path=d / "splits.json",
        split_spec=[("train", 2), ("test", None)],
    )
    assert (d / "splits.json").exists()
    assert isinstance(splits, dict)


def _main() -> None:
    _ensure_samples()
    tests = [
        test_entry_modules_import,
        test_load_vignettes, test_load_av_cases, test_load_hv_cases,
        test_adherence_build_dataset, test_fidelity_build_dataset,
        test_fairness_build_jobs, test_simulator_accepts_sample_params,
        test_turing_build_real, test_turing_normalize, test_turing_build_splits,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
