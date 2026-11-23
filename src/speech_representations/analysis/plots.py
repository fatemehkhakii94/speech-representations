"""Generate all portfolio figures directly from validated result artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from speech_representations.config import ProjectConfig

COLORS = {"wavlm_large": "#2563EB", "hubert_large": "#F59E0B", "whisper_large_v3": "#16A34A"}


def _probe_result(config: ProjectConfig, task: str, model: str) -> dict[str, Any] | None:
    path = config.paths.runs / "probing" / task / model / "result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _plot_probe_task(config: ProjectConfig, task_name: str, axis: plt.Axes) -> bool:
    task = config.task(task_name)
    found = False
    for model_name, model in config.models.items():
        result = _probe_result(config, task_name, model_name)
        if result is None:
            continue
        found = True
        layers = result["layers"]
        means = np.asarray([layer["test"][task.primary_metric]["mean"] for layer in layers])
        stds = np.asarray([layer["test"][task.primary_metric]["std"] for layer in layers])
        x = np.arange(len(layers))
        color = COLORS[model_name]
        axis.plot(x, means, color=color, label=model.display_name, linewidth=2)
        axis.fill_between(x, means - stds, means + stds, color=color, alpha=0.15)
        selected = int(result["selection"]["layer"])
        axis.scatter(selected, means[selected], color=color, edgecolor="white", s=55, zorder=3)
    if found:
        ylabel = (
            "MAE (semitones; lower is better)"
            if task.selection_mode == "min"
            else task.primary_metric.upper()
        )
        axis.set(title=task.display_name, xlabel="Representation layer", ylabel=ylabel)
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.25)
    return found


def generate_figures(config: ProjectConfig) -> list[Path]:
    output = config.paths.reports
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    available = [
        task
        for task in config.tasks
        if any(_probe_result(config, task, model) for model in config.models)
    ]
    if available:
        figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        for axis, task in zip(axes.flat, available, strict=False):
            _plot_probe_task(config, task, axis)
        for axis in axes.flat[len(available) :]:
            axis.axis("off")
        figure.suptitle(
            "Layer-wise information in speech foundation models", fontsize=16, fontweight="bold"
        )
        path = output / "probing_summary.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        created.append(path)

    cka_dir = config.paths.runs / "cka"
    for artifact in sorted(cka_dir.glob("*.npz")) if cka_dir.exists() else []:
        data = np.load(artifact)
        matrix = data["matrix"]
        model_a, model_b = str(data["model_a"]), str(data["model_b"])
        figure, axis = plt.subplots(figsize=(7.5, 6))
        image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
        axis.set(
            title=f"Linear CKA: {config.model(model_a).display_name} vs {config.model(model_b).display_name}",
            xlabel=f"Layer — {config.model(model_b).display_name}",
            ylabel=f"Layer — {config.model(model_a).display_name}",
        )
        figure.colorbar(image, ax=axis, label="CKA similarity")
        figure.tight_layout()
        path = output / f"cka_{model_a}__{model_b}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        created.append(path)
    return created
