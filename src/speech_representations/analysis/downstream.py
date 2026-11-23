"""Unbiased downstream validation of probing-guided emotion representations."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from speech_representations.config import ProjectConfig
from speech_representations.data.records import read_task_records
from speech_representations.features.store import FeatureStore
from speech_representations.probing.metrics import summarize_runs
from speech_representations.probing.trainer import train_probe
from speech_representations.utils import (
    atomic_write_json,
    environment_snapshot,
    git_revision,
    resolve_device,
    seed_everything,
    source_fingerprint,
)


class EmotionClassifier(nn.Module):
    def __init__(self, input_size: int, classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class WeightedLayerClassifier(nn.Module):
    def __init__(self, layers: int, hidden_size: int, classes: int) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(layers))
        self.classifier = EmotionClassifier(hidden_size, classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = self.layer_logits.softmax(dim=0)
        return self.classifier(torch.einsum("bld,l->bd", features, weights))


def _parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_downstream_emotion(
    config: ProjectConfig, model_name: str | None = None, overwrite: bool = False
) -> Path:
    model_name = model_name or config.analysis.downstream_model
    probe_path = config.paths.runs / "probing" / "emotion" / model_name / "result.json"
    if not probe_path.is_file():
        raise FileNotFoundError(f"Emotion probing result not found: {probe_path}")
    probe_result = json.loads(probe_path.read_text(encoding="utf-8"))
    if probe_result["selection"]["split"] != "validation":
        raise RuntimeError("Downstream selection requires validation-selected probing results")
    layers = probe_result["layers"]
    metric = probe_result["selection"]["metric"]
    mode = probe_result["selection"]["mode"]
    ranked = sorted(
        range(len(layers)),
        key=lambda layer: layers[layer]["validation"][metric]["mean"],
        reverse=mode == "max",
    )
    selected = ranked[0]
    top_three = ranked[:3]
    final = len(layers) - 1
    early_control = min(2, final)

    task_records = read_task_records(config.paths.data / "manifests" / "tasks" / "emotion.jsonl")
    ids = [record.id for record in task_records]
    labels = np.asarray([record.target for record in task_records], dtype=np.int64)
    splits = np.asarray([record.split for record in task_records])
    masks = {split: splits == split for split in ("train", "val", "test")}
    classes = int(labels.max()) + 1
    store = FeatureStore(config.paths.features / model_name / "cremad.h5")
    device = resolve_device(config.runtime.device)

    strategies: dict[str, tuple[list[int], str]] = {
        "final_layer": ([final], "single"),
        "probing_guided": ([selected], "single"),
        "probing_top3": (top_three, "concatenate"),
        "learned_weighted_sum": (list(range(store.num_layers)), "weighted"),
        "early_layer_control": ([early_control], "single"),
    }
    results: dict[str, Any] = {}
    for strategy, (selected_layers, method) in strategies.items():
        stacked = store.load_layers(selected_layers, ids)
        features = (
            stacked[:, 0, :]
            if method == "single"
            else (stacked.reshape(len(ids), -1) if method == "concatenate" else stacked)
        )
        runs: list[dict[str, Any]] = []
        for seed in config.training.seeds:
            seed_everything(seed, config.runtime.deterministic)
            model: nn.Module
            if method == "weighted":
                model = WeightedLayerClassifier(len(selected_layers), store.hidden_size, classes)
            else:
                model = EmotionClassifier(features.shape[-1], classes)
            parameter_count = _parameters(model)
            trained = train_probe(
                model=model,
                kind="classification",
                train_features=features[masks["train"]],
                train_labels=labels[masks["train"]],
                validation_features=features[masks["val"]],
                validation_labels=labels[masks["val"]],
                test_features=features[masks["test"]],
                test_labels=labels[masks["test"]],
                device=device,
                learning_rate=5e-4 if method == "weighted" else 1e-3,
                batch_size=128,
                max_epochs=100,
                patience=10,
            )
            run: dict[str, Any] = {
                "seed": seed,
                "validation": trained.validation,
                "test": trained.test,
                "best_epoch": trained.best_epoch,
            }
            if isinstance(model, WeightedLayerClassifier):
                run["layer_weights"] = model.layer_logits.detach().cpu().softmax(0).tolist()
            runs.append(run)
        results[strategy] = {
            "layers": selected_layers,
            "combination": method,
            "trainable_parameters": parameter_count,
            "validation": summarize_runs([run["validation"] for run in runs]),
            "test": summarize_runs([run["test"] for run in runs]),
            "runs": runs,
        }

    output_path = config.paths.runs / "downstream" / "emotion" / model_name / "result.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Downstream result exists: {output_path}; pass --overwrite to replace it"
        )
    atomic_write_json(
        output_path,
        {
            "schema_version": 1,
            "model": model_name,
            "task": "emotion",
            "selection_source": str(probe_path.relative_to(config.root)),
            "selection_metric": metric,
            "selection_split": "validation",
            "selected_layer": selected,
            "top_three_layers": top_three,
            "splits": {split: int(mask.sum()) for split, mask in masks.items()},
            "strategies": results,
            "provenance": {
                "config": asdict(config.training),
                "git_revision": git_revision(config.root),
                "source_fingerprint": source_fingerprint(config.root),
                "environment": environment_snapshot(("torch", "numpy", "scikit-learn")),
            },
        },
    )
    return output_path
