# Reproducibility

## Reference environment

The recorded study ran on:

- Windows with Python 3.11.18
- NVIDIA GeForce RTX 2070 Super (8 GB)
- PyTorch 2.6.0 + CUDA 12.4
- Transformers 4.57.6

`environment.yml` creates the reference CUDA environment and pins PyTorch/Torchaudio to their
CUDA 12.4 builds. `pyproject.toml` defines the supported package range for Python 3.10–3.12.

```bash
conda env create -f environment.yml
conda activate speech-repr
speech-repr doctor
```

The environment declares `CUBLAS_WORKSPACE_CONFIG=:4096:8`, which is required for deterministic
CUDA matrix multiplication when PyTorch deterministic algorithms are enabled.

## Reproducing the study

The one-command path is:

```bash
speech-repr reproduce
```

The first run downloads LibriSpeech `train-clean-100`, MUSAN, CREMA-D audio through Git LFS, and
three large model checkpoints. It then prepares manifests, generates the matched-noise corpus,
extracts features, trains probes, computes CKA, runs the downstream study, and generates figures.
This requires substantial network transfer, disk space, and GPU time.

For inspection or recovery, run the stages separately:

```bash
speech-repr prepare --tasks all
speech-repr validate
speech-repr extract --models all --datasets all
speech-repr probe --models all --tasks all
speech-repr cka --models all
speech-repr downstream
speech-repr figures
```

Extraction is resumable at the individual-row level. A `.partial.h5` file is never treated as a
finished feature store. Once every row is complete, it is atomically renamed to `.h5` and paired
with a JSON manifest.

## Provenance model

Feature-cache identity covers:

- schema version,
- pinned model configuration and commit,
- ordered sample IDs, paths, durations, and metadata,
- dataset name, pooling rule, and sample rate,
- canonical project configuration, package metadata, and executable Python source.

Probe results add the task manifest, training configuration, feature fingerprint, and full task
records. Result provenance also records Python, package, CUDA, and GPU versions.

`git_revision` records the repository HEAD that existed when an artifact was generated. Because an
experiment may be run before its changes are committed, `source_fingerprint` is the authoritative
identifier for the exact executable source, including uncommitted code. A cache with incompatible
identity is rejected unless replacement is explicitly requested with `--overwrite`.

## Determinism and leakage checks

- Cohorts, split assignment, MUSAN choices/crops, and probe initialization use explicit seeds.
- PyTorch deterministic algorithms and deterministic cuBLAS workspaces are enabled.
- Test data are not used for checkpoint or layer selection.
- Dataset validation rejects duplicate IDs, missing files, invalid splits, absent classes,
  non-finite regression targets, and group leakage.
- Feature stores reject duplicate/missing IDs, incorrect layer indices, incompatible shapes, and
  mismatched fingerprints.
- CKA explicitly intersects and aligns sample IDs.

Exact floating-point agreement can still depend on GPU architecture, CUDA/cuDNN, PyTorch, and
Transformers versions. Reproducing on a different supported environment should preserve the study
protocol but may not be bit-identical.

## Verification commands

Fast CPU-safe checks:

```bash
ruff check .
ruff format --check .
mypy src
pytest tests/unit --cov=speech_representations --cov-report=term-missing
speech-repr validate
```

Real-checkpoint GPU integration:

```bash
pytest tests/integration -m "integration and gpu"
```

Build the package outside the repository tree to avoid leaving distribution archives:

```bash
python -m build --outdir /path/to/a/temporary/directory
```

## Artifact policy

Tracked:

- source, configuration, tests, CI, and documentation;
- compact probing/downstream JSON;
- compressed CKA matrices;
- generated PNG figures.

Ignored:

- raw and processed corpora;
- downloaded model weights;
- HDF5 feature caches and temporary files;
- local environments, tool caches, coverage, and build products.

Dataset archives are deleted after checksum validation and successful extraction. The project does
not keep duplicate legacy scripts or alternate configurations; the package and CLI are canonical.

## Known limitations

- Speaker identification is closed-set across unseen chapters, not open-set speaker verification.
- Pitch uses an utterance-level median and does not model contours or frame-level voicing.
- The acoustic-condition task uses one SNR (10 dB) and MUSAN-derived corruption categories.
- CREMA-D is acted English speech; results should not be generalized to spontaneous, multilingual,
  or in-the-wild emotion recognition without further evaluation.
- Three seeds quantify initialization variance but do not characterize uncertainty across dataset
  resamples.
- Mean pooling intentionally discards temporal structure.
- Probing measures linear accessibility, not whether a model internally uses the attribute.
