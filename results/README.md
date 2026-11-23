# Result artifacts

This directory contains compact outputs from the corrected study. Raw data, checkpoints, and
HDF5 feature caches are intentionally excluded.

| Path | Schema |
|---|---|
| `probing/<task>/<model>/result.json` | per-layer, per-seed validation/test metrics; selected layer; split counts; provenance |
| `cka/<model_a>__<model_b>.npz` | CKA matrix, aligned sample IDs, model names, and feature fingerprints |
| `downstream/emotion/wavlm_large/result.json` | five representation strategies with per-seed metrics and provenance |
| `figures/probing_summary.png` | four-task layer curves with uncertainty and selected-layer markers |
| `figures/cka_*.png` | within-model and cross-model CKA heatmaps |

JSON and NPZ files are the source of truth for numbers in the README. Figures are generated with
`speech-repr figures`; do not edit them manually.
