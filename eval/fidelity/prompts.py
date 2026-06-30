"""LLM prompts for clinical fidelity evaluation.

Two-phase pipeline:
  Phase 1 — extract atomic clinical claims from vignette and conversation.
  Phase 2 — classify each patient claim against vignette ground truth.
"""

EXTRACT_CLAIMS_PROMPT = """\
You are a clinical information extraction specialist.

You will be given:
- PATIENT VIGNETTE: A structured patient profile (ground truth).
- SIMULATED CONVERSATION: A doctor–patient conversation transcript.

Your task is to independently enumerate the factual claims from each source.

Produce a JSON object with exactly two keys:

1. "vignette_claims"
   A flat list of all distinct, atomic factual claims derivable from the vignette.
   Each entry should be a single self-contained statement (e.g. "Patient is 84 years old",
   "Patient has atrial fibrillation", "Patient takes atorvastatin 20mg daily",
   "Patient denies alcohol use"). Cover all clinically meaningful content: demographics,
   social history, medical history, current visit details, medications, allergies,
   symptoms, and diagnosis.

2. "conversation_patient_claims"
   A flat list of all atomic factual claims made by the patient (not the doctor) in the
   conversation. Paraphrase naturally (e.g. "Patient reports shortness of breath worsening
   over several days", "Patient denies fever"). Exclude doctor statements, questions, and
   conversational filler. Focus on clinically meaningful assertions only.

Each list must be deduplicated: if the same fact is stated more than once, include it only
once using the clearest and most complete formulation. Two claims cover the same topic if
they assert the same underlying fact — minor wording differences do not make them distinct
entries.

Output valid JSON only. No preamble, no explanation outside the JSON.

---

PATIENT VIGNETTE:
{vignette}

SIMULATED CONVERSATION:
{conversation}
"""


CLASSIFY_CLAIMS_PROMPT = """\
You are an expert clinical annotator evaluating the factual fidelity of a simulated patient.

CONTEXT: A patient simulator receives a clinical vignette as input and role-plays a patient
in a doctor–patient conversation. The doctor may ask questions about topics not covered in
the original vignette — in those cases the simulator must fabricate plausible answers. Your
job is to evaluate the PATIENT's claims only (ignore doctor statements).

You will be given:
- VIGNETTE_CLAIMS: All factual claims from the patient's vignette (ground truth).
- CONVERSATION_PATIENT_CLAIMS: All factual claims made by the patient in the conversation.

For each claim in CONVERSATION_PATIENT_CLAIMS, determine:
1. Is it SUPPORTED (a corresponding fact exists in the vignette) or UNSUPPORTED (the patient
   fabricated this in response to a question not covered by the vignette)?
2. Rate it on a 1–4 scale using the appropriate rubric below.

SUPPORTED CLAIMS — Accuracy Scale:
  4 = Accurate:     Fully consistent with the vignette fact. No meaningful error.
  3 = Imprecise:    Broadly consistent but with vagueness or minor inaccuracy that would
                    not change clinical reasoning (e.g. "about two weeks" when the vignette
                    says "10 days").
  2 = Distorted:    Based on a real vignette fact but significantly altered in a clinically
                    meaningful way (e.g. wrong dose, wrong severity, different body region).
  1 = Contradicted: Directly conflicts with a specific vignette fact (e.g. denying a
                    condition the vignette lists, stating the wrong medication).

UNSUPPORTED CLAIMS — Plausibility Scale:
  These are claims the simulated patient fabricated in response to questions not covered
  by the input vignette. Rate how realistic the patient's answer is given everything
  known about this patient from the vignette:
  4 = Expected:     A response any real patient with this clinical profile would likely
                    give (e.g. a patient with anaemia reporting fatigue when asked about
                    energy levels, or denying a symptom that would be inconsistent with
                    their known condition).
  3 = Plausible:    A realistic response that could naturally come from a patient with
                    this profile, even though it is not directly implied by the vignette.
  2 = Questionable: An unlikely response for this patient's profile; would raise clinical
                    suspicion or seem inconsistent with their known history.
  1 = Implausible:  A response that contradicts or is clinically incompatible with the
                    patient's known profile (e.g. claiming to exercise daily when the
                    vignette describes severe immobility).

Also identify which vignette claims were never raised in the conversation (coverage gaps).

Produce a JSON object with exactly three keys:

1. "classified_claims"
   A list of objects, one per conversation claim:
   {{
     "claim": "<the conversation patient claim>",
     "category": "supported" or "unsupported",
     "score": 1 | 2 | 3 | 4,
     "label": "<scale label>",
     "rationale": "<brief explanation of the rating>",
     "matched_vignette_claim": "<the corresponding vignette claim, or null if unsupported>"
   }}

   Scale labels for reference:
     Supported:   1=contradicted, 2=distorted, 3=imprecise, 4=accurate
     Unsupported: 1=implausible, 2=questionable, 3=plausible, 4=expected

2. "vignette_claims_not_mentioned"
   A list of vignette claims that were never raised by the patient.
   Each entry: {{"claim": "<vignette claim>"}}

3. "summary"
   A brief (2-3 sentence) overall assessment of the simulator's factual fidelity.

Output valid JSON only. No preamble, no explanation outside the JSON.

---

VIGNETTE_CLAIMS:
{vignette_claims}

CONVERSATION_PATIENT_CLAIMS:
{conversation_patient_claims}
"""
