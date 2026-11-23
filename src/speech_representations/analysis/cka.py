"""Efficient linear CKA with explicit sample-ID alignment."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from speech_representations.config import ProjectConfig
from speech_representations.features.store import FeatureStore


def linear_cka(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Compute biased linear CKA after centering samples."""
    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError("CKA inputs must be two-dimensional")
    if features_a.shape[0] != features_b.shape[0]:
        raise ValueError("CKA inputs must describe the same number of samples")
    a = features_a.astype(np.float64, copy=False) - features_a.mean(axis=0, keepdims=True)
    b = features_b.astype(np.float64, copy=False) - features_b.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(a.T @ b, ord="fro") ** 2
    self_a = np.linalg.norm(a.T @ a, ord="fro")
    self_b = np.linalg.norm(b.T @ b, ord="fro")
    denominator = self_a * self_b
    return float(cross / denominator) if denominator > 1e-12 else 0.0


def _normalized_grams(store: FeatureStore, ids: Sequence[str]) -> np.ndarray:
    grams: list[np.ndarray] = []
    for layer in range(store.num_layers):
        features = store.load_layer(layer, ids).astype(np.float32)
        features -= features.mean(axis=0, keepdims=True)
        gram = features @ features.T
        norm = np.linalg.norm(gram)
        grams.append((gram / max(float(norm), 1e-12)).reshape(-1))
    return np.stack(grams)


def aligned_sample_ids(
    stores: Sequence[FeatureStore], maximum_samples: int, seed: int
) -> list[str]:
    common = set(stores[0].ids)
    for store in stores[1:]:
        common.intersection_update(store.ids)
    if not common:
        raise ValueError("Feature caches do not share any sample IDs")
    ordered = np.asarray(sorted(common), dtype=object)
    if len(ordered) > maximum_samples:
        rng = np.random.default_rng(seed)
        ordered = ordered[np.sort(rng.choice(len(ordered), maximum_samples, replace=False))]
    return ordered.tolist()


def run_cka(
    config: ProjectConfig,
    model_a: str,
    model_b: str,
    dataset: str = "librispeech",
    maximum_samples: int | None = None,
) -> Path:
    first = FeatureStore(config.paths.features / model_a / f"{dataset}.h5")
    second = FeatureStore(config.paths.features / model_b / f"{dataset}.h5")
    sample_limit = maximum_samples or config.analysis.cka_max_samples
    ids = aligned_sample_ids([first, second], sample_limit, config.runtime.seed)
    grams_a = _normalized_grams(first, ids)
    grams_b = grams_a if first.path == second.path else _normalized_grams(second, ids)
    matrix = np.clip(grams_a @ grams_b.T, 0.0, 1.0)
    output_dir = config.paths.runs / "cka"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_a}__{model_b}.npz"
    np.savez_compressed(
        output_path,
        matrix=matrix.astype(np.float32),
        sample_ids=np.asarray(ids),
        model_a=model_a,
        model_b=model_b,
        dataset=dataset,
        feature_fingerprint_a=first.fingerprint,
        feature_fingerprint_b=second.fingerprint,
    )
    return output_path
