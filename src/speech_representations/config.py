"""Typed project configuration and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


class ConfigError(ValueError):
    """Raised when project configuration is invalid."""


@dataclass(frozen=True)
class PathsConfig:
    data: Path
    features: Path
    runs: Path
    reports: Path


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    precision: Literal["fp32", "fp16", "bf16"]
    seed: int
    extraction_batch_size: int
    probe_batch_size: int
    num_workers: int
    deterministic: bool


@dataclass(frozen=True)
class AnalysisConfig:
    cka_max_samples: int
    downstream_model: str


@dataclass(frozen=True)
class ModelConfig:
    name: str
    display_name: str
    hub_id: str
    revision: str
    architecture: Literal["wav2vec2", "whisper"]
    transformer_layers: int
    hidden_size: int

    @property
    def representation_layers(self) -> int:
        return self.transformer_layers + 1


@dataclass(frozen=True)
class TaskConfig:
    name: str
    display_name: str
    kind: Literal["classification", "regression"]
    dataset: Literal["librispeech", "cremad", "noise"]
    primary_metric: str
    selection_mode: Literal["min", "max"]
    options: Mapping[str, Any]


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    max_epochs: int
    patience: int
    probe: Literal["linear", "mlp"]
    mlp_hidden_size: int
    dropout: float
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class ProjectConfig:
    schema_version: int
    root: Path
    paths: PathsConfig
    runtime: RuntimeConfig
    analysis: AnalysisConfig
    models: Mapping[str, ModelConfig]
    tasks: Mapping[str, TaskConfig]
    training: TrainingConfig

    def model(self, name: str) -> ModelConfig:
        try:
            return self.models[name]
        except KeyError as exc:
            raise ConfigError(f"Unknown model {name!r}; choose from {sorted(self.models)}") from exc

    def task(self, name: str) -> TaskConfig:
        try:
            return self.tasks[name]
        except KeyError as exc:
            raise ConfigError(f"Unknown task {name!r}; choose from {sorted(self.tasks)}") from exc


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing {context}.{key}")
    return mapping[key]


def load_config(path: str | Path = "configs/project.yaml") -> ProjectConfig:
    """Load and validate the single canonical project configuration."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Project configuration must be a mapping")

    root = config_path.parent.parent
    raw_paths = _required(raw, "paths", "project")
    paths = PathsConfig(**{key: (root / value).resolve() for key, value in raw_paths.items()})

    runtime = RuntimeConfig(**_required(raw, "runtime", "project"))
    if runtime.extraction_batch_size < 1 or runtime.probe_batch_size < 1:
        raise ConfigError("Batch sizes must be positive")
    analysis = AnalysisConfig(**_required(raw, "analysis", "project"))
    if analysis.cka_max_samples < 2:
        raise ConfigError("analysis.cka_max_samples must be at least two")

    models: dict[str, ModelConfig] = {}
    for name, values in _required(raw, "models", "project").items():
        model = ModelConfig(name=name, **values)
        if len(model.revision) != 40:
            raise ConfigError(f"models.{name}.revision must be a 40-character commit hash")
        models[name] = model

    tasks: dict[str, TaskConfig] = {}
    required_task_keys = {"display_name", "kind", "dataset", "primary_metric", "selection_mode"}
    for name, values in _required(raw, "tasks", "project").items():
        missing = required_task_keys - values.keys()
        if missing:
            raise ConfigError(f"tasks.{name} is missing {sorted(missing)}")
        core = {key: values[key] for key in required_task_keys}
        options = {key: value for key, value in values.items() if key not in required_task_keys}
        tasks[name] = TaskConfig(name=name, options=options, **core)

    training_values = dict(_required(raw, "training", "project"))
    training_values["seeds"] = tuple(training_values["seeds"])
    training = TrainingConfig(**training_values)
    if not training.seeds:
        raise ConfigError("training.seeds cannot be empty")

    return ProjectConfig(
        schema_version=int(_required(raw, "schema_version", "project")),
        root=root,
        paths=paths,
        runtime=runtime,
        analysis=analysis,
        models=models,
        tasks=tasks,
        training=training,
    )
