"""Metrics shared by probing and downstream evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "uar": float(recall_score(labels, predictions, average="macro", zero_division=0)),
    }


def regression_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(labels, predictions)),
        "rmse": float(mean_squared_error(labels, predictions) ** 0.5),
        "r2": float(r2_score(labels, predictions)),
    }


def summarize_runs(values: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    numeric_keys = sorted(
        key
        for key in values[0]
        if all(isinstance(value.get(key), (int, float)) for value in values)
    )
    return {
        key: {
            "mean": float(np.mean([value[key] for value in values])),
            "std": float(np.std([value[key] for value in values], ddof=1))
            if len(values) > 1
            else 0.0,
        }
        for key in numeric_keys
    }
