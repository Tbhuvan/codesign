# Contributing

Research-artifact contribution model: correctness, reproducibility,
clarity > features.

## Setup

```bash
git clone https://github.com/Tbhuvan/codesign.git
cd codesign
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebook]"
pytest -m "not slow"
```

## Adding a mutator

1. Implement the `Mutator` protocol in `src/codesign/attacker.py`.
2. Mutator must satisfy: AST validity for any AST-valid input, and
   DFG isomorphism (for renaming-only mutators) or a documented
   departure (e.g., dead-code insertion alters the CFG by design).
3. Add tests under `tests/test_attacker.py`: idempotence, AST safety,
   empty-input behaviour.
4. Update the comparison table in the README if it's a default.

## Adding a CWE sample

Drop `<id>.py` and `<id>.json` in `data/cwe_samples/`. The schema is
in `data/README.md`. Picked up automatically by `codesign benchmark`
and by `tests/test_dataset.py`.

## Adding an experiment

Place a runnable script under `experiments/expNN_<name>.py`. Persist
results to `experiments/results/expNN_<name>.json`. Append a one-line
record to `experiments.json` (see `exp01` for the pattern). Document
in `experiments/README.md`.

## Style

- `ruff check src tests experiments` must pass.
- Type hints on public APIs.
- Public functions: docstring only when the why is non-obvious.
- Per the project's CLAUDE.md: validate inputs at module boundaries,
  never log sensitive data, cite arXiv IDs when making ML claims.
