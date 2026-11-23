# Contributing

Thank you for improving the project. Keep changes small, reproducible, and consistent with the
single-package architecture.

## Development setup

```bash
conda env create -f environment.yml
conda activate speech-repr
pre-commit install
```

## Project rules

- Extend the `speech-repr` CLI instead of adding standalone experiment scripts.
- Put canonical settings in `configs/project.yaml`; do not introduce overlapping configuration
  files.
- Preserve group-disjoint evaluation and validation-only model selection.
- Pin external model revisions to immutable commits.
- Treat source fingerprints as part of the public artifact contract. Regenerate incompatible
  features/results rather than bypassing a provenance failure.
- Add tests for behavior and failure modes, especially around splits, IDs, shapes, and leakage.
- Do not commit corpora, model weights, HDF5 caches, downloaded archives, or build products.

## Acceptance checks

```bash
ruff check .
ruff format --check .
mypy src
pytest tests/unit --cov=speech_representations --cov-report=term-missing
```

Changes to model extraction should also pass:

```bash
pytest tests/integration -m "integration and gpu"
```

Changes to data or experimental methodology must run `speech-repr validate`, regenerate affected
artifacts, and update `docs/methodology.md` plus the README result summary when conclusions change.
