"""Deterministic CPU/GPU probe training with validation-only model selection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from speech_representations.probing.metrics import classification_metrics, regression_metrics


@dataclass(frozen=True)
class TrainedProbe:
    validation: dict[str, float]
    test: dict[str, float]
    best_epoch: int
    epochs: int


def standardize_splits(
    train: np.ndarray, validation: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return tuple(
        ((values - mean) / scale).astype(np.float32) for values in (train, validation, test)
    )  # type: ignore[return-value]


def _loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False))
    label_tensor = torch.from_numpy(labels)
    return DataLoader(
        TensorDataset(feature_tensor, label_tensor), batch_size=batch_size, shuffle=shuffle
    )


def _predict(
    model: nn.Module, loader: DataLoader, device: torch.device, kind: str
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    criterion: nn.Module = nn.CrossEntropyLoss() if kind == "classification" else nn.MSELoss()
    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    losses: list[float] = []
    with torch.inference_mode():
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            output = model(features)
            if kind == "classification":
                target = target.long()
                loss = criterion(output, target)
                prediction = output.argmax(dim=-1)
            else:
                target = target.float().view(-1, 1)
                loss = criterion(output, target)
                prediction = output.view(-1)
                target = target.view(-1)
            losses.append(float(loss))
            predictions.append(prediction.cpu())
            labels.append(target.cpu())
    return torch.cat(labels).numpy(), torch.cat(predictions).numpy(), float(np.mean(losses))


def train_probe(
    model: nn.Module,
    kind: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> TrainedProbe:
    train_features, validation_features, test_features = standardize_splits(
        train_features, validation_features, test_features
    )
    target_mean = 0.0
    target_scale = 1.0
    if kind == "regression":
        # Fit target normalization on the training split only. Besides preventing
        # leakage, this keeps regression optimization well-conditioned when the
        # target has a large offset (for example, pitch in semitones).
        target_mean = float(np.mean(train_labels))
        target_scale = float(np.std(train_labels))
        if target_scale < 1e-6:
            target_scale = 1.0
        train_labels = ((train_labels - target_mean) / target_scale).astype(np.float32)
        validation_labels = ((validation_labels - target_mean) / target_scale).astype(np.float32)
        test_labels = ((test_labels - target_mean) / target_scale).astype(np.float32)
    train_loader = _loader(train_features, train_labels, batch_size, True)
    validation_loader = _loader(validation_features, validation_labels, batch_size, False)
    test_loader = _loader(test_features, test_labels, batch_size, False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion: nn.Module = nn.CrossEntropyLoss() if kind == "classification" else nn.MSELoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0

    for epoch in range(max_epochs):
        model.train()
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            output = model(features)
            if kind == "classification":
                loss = criterion(output, target.long())
            else:
                loss = criterion(output, target.float().view(-1, 1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        _, _, validation_loss = _predict(model, validation_loader, device, kind)
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Probe training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    validation_true, validation_pred, _ = _predict(model, validation_loader, device, kind)
    test_true, test_pred, _ = _predict(model, test_loader, device, kind)
    if kind == "regression":
        validation_true = validation_true * target_scale + target_mean
        validation_pred = validation_pred * target_scale + target_mean
        test_true = test_true * target_scale + target_mean
        test_pred = test_pred * target_scale + target_mean
    metric = classification_metrics if kind == "classification" else regression_metrics
    return TrainedProbe(
        validation=metric(validation_true, validation_pred),
        test=metric(test_true, test_pred),
        best_epoch=best_epoch,
        epochs=epoch + 1,
    )
