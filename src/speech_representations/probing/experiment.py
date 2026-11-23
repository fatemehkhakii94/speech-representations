"""Canonical layer-wise probing workflow."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from speech_representations.config import ProjectConfig
from speech_representations.data.records import read_task_records
from speech_representations.features.store import FeatureStore
from speech_representations.probing.metrics import summarize_runs
from speech_representations.probing.models import create_probe
from speech_representations.probing.trainer import train_probe
from speech_representations.utils import (
    atomic_write_json,
    canonical_hash,
    environment_snapshot,
    git_revision,
    resolve_device,
    seed_everything,
    source_fingerprint,
)

RESULT_SCHEMA_VERSION = 1


def _select_layer(layer_results: list[dict[str, Any]], metric: str, mode: str) -> int:
    values = [result["validation"][metric]["mean"] for result in layer_results]
    return int(np.argmin(values) if mode == "min" else np.argmax(values))


def run_probing(
    config: ProjectConfig,
    model_name: str,
    task_name: str,
    overwrite: bool = False,
) -> Path:
    """Train one probe per representation layer over all configured seeds."""
    task = config.task(task_name)
    model_config = config.model(model_name)
    task_path = config.paths.data / "manifests" / "tasks" / f"{task_name}.jsonl"
    if not task_path.is_file():
        raise FileNotFoundError(
            f"Task manifest not found: {task_path}. Run `speech-repr prepare` first."
        )
    records = read_task_records(task_path)
    feature_path = config.paths.features / model_name / f"{task.dataset}.h5"
    store = FeatureStore(feature_path)
    ids = [record.id for record in records]
    labels = np.asarray(
        [record.target for record in records],
        dtype=np.int64 if task.kind == "classification" else np.float32,
    )
    splits = np.asarray([record.split for record in records])
    masks = {split: splits == split for split in ("train", "val", "test")}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError(f"Task {task_name} has an empty split")
    output_size = int(labels.max()) + 1 if task.kind == "classification" else 1
    result_dir = config.paths.runs / "probing" / task_name / model_name
    output_path = result_dir / "result.json"
    fingerprint = canonical_hash(
        {
            "schema": RESULT_SCHEMA_VERSION,
            "model": asdict(model_config),
            "task": asdict(task),
            "training": asdict(config.training),
            "feature_fingerprint": store.fingerprint,
            "records": [asdict(record) for record in records],
            "source_fingerprint": source_fingerprint(config.root),
        }
    )
    if output_path.exists() and not overwrite:
        import json

        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint:
            return output_path
        raise RuntimeError(
            f"Existing result is incompatible: {output_path}. Use --overwrite deliberately."
        )

    device = resolve_device(config.runtime.device)
    layer_results: list[dict[str, Any]] = []
    for layer in range(store.num_layers):
        features = store.load_layer(layer, ids)
        runs: list[dict[str, Any]] = []
        for seed in config.training.seeds:
            seed_everything(seed, config.runtime.deterministic)
            probe = create_probe(
                config.training.probe,
                store.hidden_size,
                output_size,
                config.training.mlp_hidden_size,
                config.training.dropout,
            )
            trained = train_probe(
                model=probe,
                kind=task.kind,
                train_features=features[masks["train"]],
                train_labels=labels[masks["train"]],
                validation_features=features[masks["val"]],
                validation_labels=labels[masks["val"]],
                test_features=features[masks["test"]],
                test_labels=labels[masks["test"]],
                device=device,
                learning_rate=config.training.learning_rate,
                batch_size=config.runtime.probe_batch_size,
                max_epochs=config.training.max_epochs,
                patience=config.training.patience,
            )
            runs.append(
                {
                    "seed": seed,
                    "validation": trained.validation,
                    "test": trained.test,
                    "best_epoch": trained.best_epoch,
                    "epochs": trained.epochs,
                }
            )
        layer_results.append(
            {
                "layer": layer,
                "validation": summarize_runs([run["validation"] for run in runs]),
                "test": summarize_runs([run["test"] for run in runs]),
                "runs": runs,
            }
        )

    selected_layer = _select_layer(layer_results, task.primary_metric, task.selection_mode)
    class_counts = None
    majority_baseline = None
    if task.kind == "classification":
        test_labels = labels[masks["test"]].astype(int)
        class_counts = np.bincount(test_labels, minlength=output_size).tolist()
        majority_baseline = float(max(class_counts) / sum(class_counts))
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "model": model_name,
        "task": task_name,
        "kind": task.kind,
        "probe": config.training.probe,
        "selection": {
            "split": "validation",
            "metric": task.primary_metric,
            "mode": task.selection_mode,
            "layer": selected_layer,
        },
        "splits": {split: int(mask.sum()) for split, mask in masks.items()},
        "num_classes": output_size if task.kind == "classification" else None,
        "test_class_counts": class_counts,
        "majority_baseline": majority_baseline,
        "layers": layer_results,
        "provenance": {
            "feature_cache": str(feature_path.relative_to(config.root)),
            "feature_fingerprint": store.fingerprint,
            "task_manifest": str(task_path.relative_to(config.root)),
            "git_revision": git_revision(config.root),
            "model_revision": model_config.revision,
            "source_fingerprint": source_fingerprint(config.root),
            "environment": environment_snapshot(("torch", "numpy", "scikit-learn")),
        },
    }
    atomic_write_json(output_path, result)
    return output_path
