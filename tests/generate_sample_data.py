"""Generate small synthetic sample datasets for tests.

These are hand-authored, entirely fictional records that mirror the column
schema of the real AV (synthetic clinical vignettes) and HV (de-identified EHR)
datasets used in the paper, so the downstream loaders / dataset builders /
simulator can be exercised without the real (not public) data.

Writes 5-row CSVs to:
    tests/sample_synthetic_data/av/av_clean.csv
    tests/sample_synthetic_data/hv/hv_clean.csv

Run:  python tests/generate_sample_data.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

_DIR = Path(__file__).parent

# Triage vocabularies (must match AV_TRIAGE_MAP / GOLD_LABEL_MAP in the loaders)
_AV_SELF = "pursue self-care and monitoring"
_AV_NONURG = "seek non-urgent follow-up care"
_AV_URG = "seek immediate medical care"
_HV_SELF = "Self-care"
_HV_NONURG = "Follow up with PCP"
_HV_URG = "Urgent care / emergency"

# ---------------------------------------------------------------------------
# AV — synthetic clinical vignettes
# ---------------------------------------------------------------------------

AV_COLUMNS = [
    "case", "sex", "age", "chief_symptom", "chronic_conditions",
    "last_encounter_notes", "patient_summary", "labs", "previous_dx",
    "ICD9_10 code", "grader_triage", "grader_1_triage", "grader_2_triage",
    "grader_3_triage", "grader_4_triage", "grader_5_triage", "conversation",
]


def _av_row(case, sex, age, chief, chronic, notes, summary, labs, dx, icd, triage):
    return {
        "case": case, "sex": sex, "age": age, "chief_symptom": chief,
        "chronic_conditions": chronic, "last_encounter_notes": notes,
        "patient_summary": summary, "labs": labs, "previous_dx": dx,
        "ICD9_10 code": icd, "grader_triage": triage,
        "grader_1_triage": triage, "grader_2_triage": triage,
        "grader_3_triage": triage, "grader_4_triage": triage,
        "grader_5_triage": triage,
        "conversation": (
            "Patient: Hi, I wanted to ask about my symptoms.\n"
            "Agent: Of course — can you tell me what's been going on?\n"
            "Patient: " + chief.rstrip(".") + ".\n"
            "Agent: Thanks, that helps. How long has this been happening?"
        ),
    }


AV_ROWS = [
    _av_row(1, "Female", 28, "Sore throat and mild cough for 2 days",
            "None", "No significant past medical history.",
            "28-year-old woman with a 2-day history of sore throat, mild cough, and "
            "low-grade fever. No difficulty breathing or swallowing. Eating and "
            "drinking normally. No known sick contacts.",
            "", "None", "J02.9", _AV_SELF),
    _av_row(2, "Male", 54, "Right knee pain worse over 3 weeks",
            "Osteoarthritis, Hypertension",
            "Seen 6 months ago for knee osteoarthritis; advised on exercises.",
            "54-year-old man with a 3-week history of gradually worsening right knee "
            "pain, worse with stairs and prolonged standing. Mild swelling, no "
            "redness or warmth. No trauma. Over-the-counter ibuprofen helps partly.",
            "", "Knee osteoarthritis", "M17.11", _AV_NONURG),
    _av_row(3, "Male", 67, "Chest pain and shortness of breath",
            "Hypertension, Type 2 diabetes, Hyperlipidemia",
            "Last seen for routine diabetes review 2 months ago.",
            "67-year-old man with 1 hour of central crushing chest pain radiating to "
            "the left arm, with sweating and shortness of breath. Pain not relieved "
            "by rest. History of hypertension and diabetes.",
            "", "Hypertension; Type 2 diabetes", "I20.0", _AV_URG),
    _av_row(4, "Female", 41, "Recurrent headaches",
            "Migraine",
            "Migraine diagnosed 5 years ago; uses sumatriptan as needed.",
            "41-year-old woman with a 4-day history of throbbing one-sided headaches "
            "with light sensitivity and nausea, similar to prior migraines but more "
            "frequent this month. No fever, neck stiffness, or weakness.",
            "", "Migraine without aura", "G43.909", _AV_NONURG),
    _av_row(5, "Male", 35, "Lower back pain after lifting",
            "None", "No prior back problems documented.",
            "35-year-old man with 3 days of lower back pain that started after "
            "lifting boxes. Pain is localised, no radiation to the legs, no numbness, "
            "weakness, or bladder/bowel changes. Improves with rest and paracetamol.",
            "", "None", "M54.5", _AV_SELF),
]

# ---------------------------------------------------------------------------
# HV — de-identified EHR-style encounters
# ---------------------------------------------------------------------------

HV_COLUMNS = [
    "encounter_id", "patient_id", "encounter_date", "gender", "age",
    "chief_complaint", "history_of_present_illness", "review_of_systems",
    "chronic_conditions", "current_medications", "allergies", "immunizations",
    "recent_labs", "last_encounter_notes", "social_history", "family_history",
    "patient_vignette", "encounter_purpose", "chief_complaint_category",
    "condition_severity", "symptom_onset_timing", "conversation",
    "annotator_1_selected_diagnoses", "annotator_1_additional_diagnoses",
    "annotator_1_triage_recommendation", "annotator_2_selected_diagnoses",
    "annotator_2_additional_diagnoses", "annotator_2_triage_recommendation",
    "annotator_3_selected_diagnoses", "annotator_3_additional_diagnoses",
    "annotator_3_triage_recommendation", "triage_label",
]


def _hv_row(eid, pid, gender, age, cc, hpi, ros, chronic, meds, allergies,
            imm, labs, notes, social, family, vignette, purpose, category,
            severity, onset, dx, triage):
    return {
        "encounter_id": eid, "patient_id": pid, "encounter_date": "2026-01-15",
        "gender": gender, "age": age, "chief_complaint": cc,
        "history_of_present_illness": hpi, "review_of_systems": ros,
        "chronic_conditions": chronic, "current_medications": meds,
        "allergies": allergies, "immunizations": imm, "recent_labs": labs,
        "last_encounter_notes": notes, "social_history": social,
        "family_history": family, "patient_vignette": vignette,
        "encounter_purpose": purpose, "chief_complaint_category": category,
        "condition_severity": severity, "symptom_onset_timing": onset,
        "conversation": (
            "Patient: Hello, I'm not feeling well.\n"
            "Doctor: I'm sorry to hear that. What's bothering you most?\n"
            "Patient: " + cc.rstrip(".") + ".\n"
            "Doctor: Understood. Let me ask a few questions to help."
        ),
        "annotator_1_selected_diagnoses": dx,
        "annotator_1_additional_diagnoses": "",
        "annotator_1_triage_recommendation": triage,
        "annotator_2_selected_diagnoses": dx,
        "annotator_2_additional_diagnoses": "",
        "annotator_2_triage_recommendation": triage,
        "annotator_3_selected_diagnoses": dx,
        "annotator_3_additional_diagnoses": "",
        "annotator_3_triage_recommendation": triage,
        "triage_label": triage,
    }


HV_ROWS = [
    _hv_row("ENC0001", "PT001", "Female", 33,
            "Painful urination for 2 days",
            "2-day history of dysuria and urinary frequency. No fever, flank pain, "
            "or vaginal discharge. No prior UTIs this year.",
            "Denies fever, nausea, back pain.", "None", "None", "None documented",
            "Up to date", "Urinalysis pending", "Healthy; last seen 1 year ago.",
            "Non-smoker, occasional alcohol.", "Non-contributory.",
            "33-year-old woman with 2 days of painful, frequent urination and no "
            "systemic symptoms.",
            "acute", "genitourinary", "mild", "days",
            "Uncomplicated urinary tract infection", _HV_NONURG),
    _hv_row("ENC0002", "PT002", "Male", 72,
            "Sudden weakness on one side",
            "1-hour history of sudden left-sided arm and face weakness with slurred "
            "speech. No headache or trauma.",
            "Positive for facial droop and arm drift.",
            "Atrial fibrillation, Hypertension",
            "Apixaban, Amlodipine", "None", "Up to date",
            "INR pending", "Seen 3 months ago for AF review.",
            "Ex-smoker.", "Father had stroke.",
            "72-year-old man with sudden left-sided weakness and slurred speech "
            "starting one hour ago.",
            "acute", "neurological", "severe", "hours",
            "Acute stroke", _HV_URG),
    _hv_row("ENC0003", "PT003", "Female", 45,
            "Seasonal nasal congestion",
            "2-week history of nasal congestion, sneezing, and itchy eyes during "
            "pollen season. No fever or facial pain.",
            "Denies fever, cough, shortness of breath.",
            "Allergic rhinitis", "Loratadine as needed", "Penicillin",
            "Up to date", "None", "Seen last spring for similar symptoms.",
            "Non-smoker.", "Mother has hay fever.",
            "45-year-old woman with recurrent seasonal allergic rhinitis symptoms.",
            "routine", "ear_nose_throat", "mild", "weeks",
            "Allergic rhinitis", _HV_SELF),
    _hv_row("ENC0004", "PT004", "Male", 58,
            "Worsening cough and breathlessness",
            "5-day history of productive cough, fever, and breathlessness on "
            "exertion. Background of COPD.",
            "Positive for fever and wheeze.",
            "COPD, Hypertension",
            "Tiotropium inhaler, Ramipril", "None", "Influenza vaccine given",
            "CRP elevated", "Seen 4 months ago for COPD review.",
            "Ex-smoker, 30 pack-years.", "Non-contributory.",
            "58-year-old man with a COPD background and 5 days of productive cough, "
            "fever, and worsening breathlessness.",
            "acute", "respiratory", "moderate", "days",
            "COPD exacerbation", _HV_NONURG),
    _hv_row("ENC0005", "PT005", "Female", 26,
            "Mild ankle sprain",
            "1-day history of left ankle pain after rolling it while running. Able "
            "to bear weight with mild discomfort. No deformity.",
            "Denies numbness or severe swelling.", "None", "None", "None",
            "Up to date", "None", "No prior injuries.",
            "Non-smoker, active runner.", "Non-contributory.",
            "26-year-old woman with a mild left ankle sprain after running, able to "
            "bear weight.",
            "acute", "musculoskeletal", "mild", "days",
            "Ankle sprain", _HV_SELF),
]


# ---------------------------------------------------------------------------
# Turing samples: real_clean.csv (JSON conversations) + simulated_raw.jsonl
# ---------------------------------------------------------------------------

_CHIEFS = [
    "I've had a sore throat and mild cough",
    "my right knee has been aching for weeks",
    "I keep getting bad headaches",
    "I've got some lower back pain after lifting",
    "I've had a burning feeling when I pee",
]

REAL_COLUMNS = [
    "session_id", "batch", "patient_id", "age", "gender", "location",
    "pha_triage", "urgency_reasoning", "care_recommendations",
    "when_to_escalate", "most_probable_diagnosis", "labs_tests_needed",
    "labeller_triage", "conversation",
]


def _real_conversation(chief: str) -> str:
    msgs = [
        {"role": "agent", "content": "Hi there, what's brought you to the app today?"},
        {"role": "patient", "content": f"{chief}."},
        {"role": "agent", "content": "How long has this been going on?"},
        {"role": "patient", "content": "A couple of days now."},
        {"role": "agent", "content": "Any other symptoms alongside it?"},
        {"role": "patient", "content": "Not really, just the main issue."},
        {"role": "agent", "content": "Have you tried anything for it so far?"},
        {"role": "patient", "content": "Some over-the-counter remedies, a little relief."},
    ]
    return json.dumps(msgs)


def _build_real_rows() -> list[dict]:
    rows = []
    for i, chief in enumerate(_CHIEFS, 1):
        rows.append({
            "session_id": f"sess_{i:03d}", "batch": "sample", "patient_id": f"PT{i:03d}",
            "age": 30 + i * 5, "gender": "Female" if i % 2 else "Male",
            "location": "Region A", "pha_triage": "non-urgent",
            "urgency_reasoning": "Symptoms appear mild and non-emergent.",
            "care_recommendations": "Self-care and monitoring.",
            "when_to_escalate": "If symptoms worsen or new red flags appear.",
            "most_probable_diagnosis": "Self-limiting condition",
            "labs_tests_needed": "None", "labeller_triage": "non-urgent",
            "conversation": _real_conversation(chief),
        })
    return rows


def _sim_raw_record(i: int, chief: str) -> dict:
    transcript = [
        {"role": "doctor", "content": "Hi, Alex: what's going on today?"},
        {"role": "patient", "content": "Yes, that's me."},
        {"role": "doctor", "content": "Can you tell me more about your symptoms?"},
        {"role": "patient", "content": f"{chief}."},
        {"role": "doctor", "content": "How long has this lasted?"},
        {"role": "patient", "content": "A few days."},
    ]
    return {
        "session_id": f"sim_{i:03d}", "age": 30 + i * 5,
        "gender": "Female" if i % 2 else "Male", "turn_count": 3,
        "urgency_status": "non-urgent", "pha_urgency_label": "non-urgent",
        "urgency_reasoning": "", "care_recommendations": "",
        "when_to_escalate": "", "most_probable_diagnosis": "",
        "labs_tests_needed": "", "sim_params": {}, "transcript": transcript,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} rows -> {path}")


def _write(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}")


def main() -> None:
    base = _DIR / "sample_synthetic_data"
    _write(base / "av" / "av_clean.csv", AV_COLUMNS, AV_ROWS)
    _write(base / "hv" / "hv_clean.csv", HV_COLUMNS, HV_ROWS)
    _write(base / "real" / "real_clean.csv", REAL_COLUMNS, _build_real_rows())
    _write_jsonl(base / "turing" / "simulated_raw.jsonl",
                 [_sim_raw_record(i, c) for i, c in enumerate(_CHIEFS, 1)])


if __name__ == "__main__":
    main()
