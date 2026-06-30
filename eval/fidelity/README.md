# Clinical Fidelity Eval

Evaluates whether the patient simulator accurately reproduces clinical facts from the input vignette (extract claims, then classify as supported/unsupported).

## Run

```bash
python -m eval.fidelity.run
python -m eval.fidelity.run --config path/to/config.json
python -m eval.fidelity.run --source-run-id <id>
```
