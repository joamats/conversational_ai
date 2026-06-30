# Turing Test Evaluation

Evaluates whether simulated patient conversations are distinguishable from real ones, using both human raters (Gradio UI) and an LLM judge.

## Pipeline overview

Real and simulated conversations are normalized to a common format, then judged.

```
real_clean.csv
      │
      ├──> build-real ──────> data/real.jsonl
      │
      └──> extract-params ──> data/extracted_params.jsonl

   data/simulated_raw.jsonl ──> normalize ──> data/simulated.jsonl
                                                    │
            data/real.jsonl ──────────────────────>│
                                                    v
                                       turing-judge / llm-judge / ui
```

All intermediate and output files are written to `data/`.

> **Note:** `data/simulated_raw.jsonl` (the simulated-side conversations) is produced separately; `normalize` consumes it if present.

## Step 1: Build real.jsonl

Converts the source `real_clean.csv` into the turing format.

```bash
python -m eval.turing.llms.run \
    --build-real --data path/to/real_clean.csv
```

Output: `data/real.jsonl`.

## Step 2: Extract simulation parameters from real conversations

Uses an LLM to reverse-engineer patient-simulator parameters from each real conversation:

- **Vignette** — built only from facts the patient explicitly stated
- **Communication parameters** — verbosity, health literacy, style, typos, etc.
- **Persona parameters** — anxiety, frustration, patience, trust, embarrassment
- **EHR data** — mapped directly from source columns

```bash
python -m eval.turing.llms.run \
    --extract-params --data path/to/real_clean.csv \
    --model gpt-5.5-2026-04-23 --reasoning-effort medium --max-workers 4
```

Output: `data/extracted_params.jsonl`.

## Step 3: Normalize simulated conversations to turing format

Post-processes `data/simulated_raw.jsonl` to match `real.jsonl` (strips intros, redacts names, truncates so the patient speaks last).

```bash
python -m eval.turing.llms.run --normalize
```

Output: `data/simulated.jsonl`.

## Evaluate: LLM / Turing judge

```bash
python -m eval.turing.llms.run \
    --llm-judge --n-simulated 5 --n-real 5 --model gpt-5.5-2026-04-23 --reasoning-effort xhigh

python -m eval.turing.llms.run \
    --turing-judge --n-pairs 10
```

Override data paths with `--simulated <path>` and `--real <path>`.

## Evaluate: human raters (Gradio UI)

```bash
python -m eval.turing.llms.run --ui
```

Add `--share` to create a public link.

## Preprocessing pipeline (one command)

Runs build-real → extract → normalize in sequence:

```bash
python -m eval.turing.llms.run \
    --pipeline --data path/to/real_clean.csv --model gpt-5.5-2026-04-23 --max-workers 4
```

Use `--skip-build-real` or `--skip-extract` to reuse existing intermediate files.
