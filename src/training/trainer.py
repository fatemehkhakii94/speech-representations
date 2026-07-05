"""
Probe training loop.

Handles:
- Training with early stopping
- Validation monitoring
- Clean result reporting

Design philosophy:
- One trainer handles ALL probing experiments
- Variation comes from configs (data, probe type, hyperparams)
- No experiment-specific code in the trainer
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingResult:
    """Container for probe training results."""
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_loss: float
    val_loss: float
    test_loss: float
    best_epoch: int
    total_epochs: int
    training_time: float
    
    def __repr__(self) -> str:
        return (
            f"TrainingResult(\n"
            f"  train_acc={self.train_accuracy:.4f}, "
            f"val_acc={self.val_accuracy:.4f}, "
            f"test_acc={self.test_accuracy:.4f}\n"
            f"  best_epoch={self.best_epoch}/{self.total_epochs}, "
            f"time={self.training_time:.1f}s\n"
            f")"
        )


class ProbeTrainer:
    """
    Trains a probe on cached features.
    
    Usage:
        trainer = ProbeTrainer(
            probe=probe,
            device="cuda",
            learning_rate=1e-3,
            max_epochs=50,
            patience=5,
        )
        result = trainer.train(train_features, train_labels,
                               val_features, val_labels,
                               test_features, test_labels)
    """
    
    def __init__(
        self,
        probe: nn.Module,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 50,
        patience: int = 5,
        num_workers: int = 0,
    ):
        self.probe = probe.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.num_workers = num_workers
        
        self.optimizer = torch.optim.Adam(
            self.probe.parameters(), lr=learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def train(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
    ) -> TrainingResult:
        """
        Full training loop with early stopping.
        
        Args:
            train/val/test_features: numpy arrays of shape (num_samples, input_dim)
            train/val/test_labels: numpy arrays of shape (num_samples,) with integer labels
        
        Returns:
            TrainingResult with all metrics
        """
        # Create data loaders
        train_loader = self._make_loader(train_features, train_labels, shuffle=True)
        val_loader = self._make_loader(val_features, val_labels, shuffle=False)
        test_loader = self._make_loader(test_features, test_labels, shuffle=False)
        
        # Training state
        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = 0
        epochs_without_improvement = 0
        best_state_dict = None
        
        t0 = time.time()
        
        for epoch in range(self.max_epochs):
            # --- Train ---
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # --- Validate ---
            val_loss, val_acc = self._evaluate(val_loader)
            
            # --- Early stopping check ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state_dict = {
                    k: v.clone() for k, v in self.probe.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= self.patience:
                break
        
        # Restore best model
        if best_state_dict is not None:
            self.probe.load_state_dict(best_state_dict)
        
        # Final evaluation on test set
        test_loss, test_acc = self._evaluate(test_loader)
        
        # Also get final train metrics with best model
        final_train_loss, final_train_acc = self._evaluate(train_loader)
        
        training_time = time.time() - t0
        
        return TrainingResult(
            train_accuracy=final_train_acc,
            val_accuracy=best_val_acc,
            test_accuracy=test_acc,
            train_loss=final_train_loss,
            val_loss=best_val_loss,
            test_loss=test_loss,
            best_epoch=best_epoch,
            total_epochs=epoch + 1,
            training_time=training_time,
        )
    
    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch. Returns (avg_loss, accuracy)."""
        self.probe.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            logits = self.probe(features)
            loss = self.criterion(logits, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * features.size(0)
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += features.size(0)
        
        return total_loss / total, correct / total
    
    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on a data split. Returns (avg_loss, accuracy)."""
        self.probe.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.probe(features)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item() * features.size(0)
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += features.size(0)
        
        return total_loss / total, correct / total
    
    def _make_loader(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        shuffle: bool = False,
    ) -> DataLoader:
        """Create a DataLoader from numpy arrays."""
        features_tensor = torch.from_numpy(features).float()
        labels_tensor = torch.from_numpy(labels).long()
        
        dataset = TensorDataset(features_tensor, labels_tensor)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=(self.device == "cuda"),
        )
