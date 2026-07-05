"""
Evaluation metrics for probing experiments.

Key metrics:
- Accuracy: overall classification accuracy
- Selectivity: accuracy - control_accuracy (measures genuine signal)
- Per-class F1: handles class imbalance
- UAR: Unweighted Average Recall (standard for emotion recognition)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)
from typing import Dict


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    """
    Compute all relevant probing metrics.
    
    Args:
        predictions: (num_samples,) predicted class indices
        labels: (num_samples,) ground truth class indices
        num_classes: total number of classes
    
    Returns:
        Dictionary of metric_name → value
    """
    metrics = {}
    
    # Overall accuracy
    metrics["accuracy"] = accuracy_score(labels, predictions)
    
    # Macro F1 (unweighted mean across classes)
    metrics["f1_macro"] = f1_score(
        labels, predictions, average="macro", zero_division=0
    )
    
    # Weighted F1 (weighted by class support)
    metrics["f1_weighted"] = f1_score(
        labels, predictions, average="weighted", zero_division=0
    )
    
    # UAR: Unweighted Average Recall (= balanced accuracy)
    metrics["uar"] = recall_score(
        labels, predictions, average="macro", zero_division=0
    )
    
    # Majority baseline (always predict most common class)
    unique, counts = np.unique(labels, return_counts=True)
    majority_acc = counts.max() / len(labels)
    metrics["majority_baseline"] = majority_acc
    
    # Selectivity (how much better than majority baseline)
    metrics["selectivity"] = metrics["accuracy"] - majority_acc
    
    # Random baseline (1 / num_classes)
    metrics["random_baseline"] = 1.0 / num_classes
    
    return metrics


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(labels, predictions)
