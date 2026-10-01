# Contributing

## Setup

```bash
pip install -e ".[dev,segmentation,timm]"
pre-commit install
```

## Checks (same as CI)

```bash
ruff check src tests && ruff format --check src tests
mypy src tests
bandit -c pyproject.toml -r src
pytest
```

## Adding a dataset adapter

Each task package (`classification/`, `detection/`, `segmentation/semantic/`)
has the same shape: `base.py` (spec + abstract adapter), `registry.py`,
`datasets/<name>.py` (the only place that knows the raw format), and a
`trainer.py`. To add a dataset, write one adapter and register it with
`@register`; trainers and CLIs don't change.

Rules every adapter follows (see `docs/official-toolkit-review.md`):

- Read released masks/labels directly; never round-trip ids through colour
  channels or floats.
- Fail loudly: an empty `train`/`test` split raises
  (`bdd100k_toolkit.utils.checks.require_nonempty`); anything dropped is counted
  and printed.
- Tests use small synthetic fixtures (`tests/conftest.py`) and, where
  licensing allows, real fixtures.
- The official `bdd100k/bdd100k` toolkit is a read-only spec, never imported.

Commits follow Conventional Commits (enforced by pre-commit).
