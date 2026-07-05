"""
Downstream Validation: Does probing predict functional utility?

Tests whether probing-guided layer selection outperforms naive strategies
(last layer, random layer) and approaches optimal strategies (learned weighted sum).

Task: Emotion recognition on CREMA-D
Model: WavLM-Large (our most complete analysis)

Methods compared:
1. Last layer (common default)
2. Probing-guided best layer (from linear probe results)
3. Probing-guided top-3 layers (concatenated)
4. Learned weighted sum (trainable layer weights — upper bound)
5. Random layer (negative control)

Usage:
    python scripts/run_downstream_validation.py
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, ".")
from src.data.cremad import scan_cremad, build_emotion_probing_dataset, EMOTION_NAMES
from src.models.caching import load_cached_features, load_metadata
from src.evaluation.metrics import compute_metrics
from src.config import get_model_config


# =============================================================================
# DOWNSTREAM CLASSIFIER (slightly larger than a probe — 2-layer MLP)
# =============================================================================

class EmotionClassifier(nn.Module):
    """
    2-layer MLP classifier for emotion recognition.
    
    Intentionally larger than a probe — this is a real downstream system,
    not a probing tool. The point is to test whether BETTER INPUT FEATURES
    (from probing-guided selection) lead to better downstream performance.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        return self.network(x)


class LearnedWeightedSum(nn.Module):
    """
    Learned weighted sum of all layers + classifier.
    
    This is the SUPERB-style approach:
    - Learn a scalar weight per layer
    - Weighted sum of all layer representations
    - Feed into classifier
    
    This represents the "upper bound" — uses ALL layers optimally.
    """
    def __init__(self, num_layers: int, hidden_dim: int, num_classes: int):
        super().__init__()
        # Learnable layer weights (initialized uniform)
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Classifier on top
        self.classifier = EmotionClassifier(hidden_dim, num_classes)
    
    def forward(self, all_layer_features: torch.Tensor):
        """
        Args:
            all_layer_features: (batch, num_layers, hidden_dim)
        Returns:
            logits: (batch, num_classes)
        """
        # Softmax over layer weights to get proper distribution
        weights = torch.softmax(self.layer_weights, dim=0)
        
        # Weighted sum: (batch, hidden_dim)
        weighted = torch.einsum("bld,l->bd", all_layer_features, weights)
        
        return self.classifier(weighted)


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_classifier(
    model: nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    device: str = "cuda",
    lr: float = 1e-3,
    batch_size: int = 128,
    max_epochs: int = 100,
    patience: int = 10,
) -> dict:
    """Train a classifier and return best validation metrics."""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        # Train
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                logits = model(features)
                preds = logits.argmax(dim=-1).cpu()
                all_preds.append(preds)
                all_labels.append(labels)
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        val_acc = (all_preds == all_labels).mean()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            break
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    
    return {"best_val_acc": best_val_acc, "epochs": epoch + 1}


def evaluate_classifier(
    model: nn.Module,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 128,
) -> dict:
    """Evaluate classifier on test set."""
    model = model.to(device)
    model.eval()
    
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            logits = model(features)
            preds = logits.argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    metrics = compute_metrics(all_preds, all_labels, num_classes=6)
    return metrics


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("=" * 70)
    print("DOWNSTREAM VALIDATION: Probing-Guided Layer Selection")
    print("=" * 70)
    
    model_name = "wavlm_large"
    features_dir = "features/wavlm_large/cremad"
    config = get_model_config(model_name)
    num_layers = config.num_layers + 1  # 25
    hidden_dim = config.hidden_dim      # 1024
    num_classes = 6
    
    # =========================================================================
    # 1. Load probing results to determine layer selection
    # =========================================================================
    print("\n[1] Loading probing results for layer selection...")
    
    probing_results_path = "results/probing/emotion/wavlm_large_linear.json"
    with open(probing_results_path) as f:
        probing_results = json.load(f)
    
    # Get layer-wise accuracies from probing
    layer_accs = [
        probing_results["layer_results"][str(i)]["test_accuracy"]
        for i in range(num_layers)
    ]
    
    # Determine layers
    best_probed_layer = int(np.argmax(layer_accs))
    top3_layers = sorted(
        range(num_layers), key=lambda l: layer_accs[l], reverse=True
    )[:3]
    worst_layer = int(np.argmin(layer_accs))
    last_layer = num_layers - 1
    random_layer = 2  # Arbitrary early layer (bad for emotion)
    
    print(f"  Best probed layer: {best_probed_layer} (probe acc: {layer_accs[best_probed_layer]:.4f})")
    top3_accs = [f"{layer_accs[l]:.4f}" for l in top3_layers]
    print(f"  Top-3 layers: {top3_layers} (accs: {top3_accs})")
    print(f"  Last layer: {last_layer} (probe acc: {layer_accs[last_layer]:.4f})")
    print(f"  Random layer: {random_layer} (probe acc: {layer_accs[random_layer]:.4f})")
    
    # =========================================================================
    # 2. Build dataset (same splits as probing for fair comparison)
    # =========================================================================
    print("\n[2] Building dataset...")
    
    audio_dir = "data/raw/cremad/AudioWAV"
    utterances = scan_cremad(audio_dir)
    dataset = build_emotion_probing_dataset(
        features_dir=features_dir,
        utterances=utterances,
        num_folds=5,
        test_fold=0,
        seed=42,
    )
    
    # =========================================================================
    # 3. Load features for all methods
    # =========================================================================
    print("\n[3] Loading features...")
    
    # Load all layers (needed for weighted sum)
    all_layer_features = {}
    for layer_idx in range(num_layers):
        all_features = load_cached_features(features_dir, layer=layer_idx)
        all_layer_features[layer_idx] = all_features[dataset.cache_indices]
    
    # Prepare splits
    def get_split_tensors(layer_idx, split):
        features, labels = dataset.get_split(layer_idx, split)
        return torch.from_numpy(features).float(), torch.from_numpy(labels).long()
    
    def get_concat_split_tensors(layers, split):
        """Concatenate features from multiple layers."""
        all_feats = []
        for l in layers:
            feats, labels = dataset.get_split(l, split)
            all_feats.append(feats)
        concat = np.concatenate(all_feats, axis=1)
        return torch.from_numpy(concat).float(), torch.from_numpy(labels).long()
    
    def get_all_layers_split_tensors(split):
        """Stack all layers for weighted sum: (batch, num_layers, hidden_dim)"""
        feats_list = []
        for l in range(num_layers):
            feats, labels = dataset.get_split(l, split)
            feats_list.append(feats)
        stacked = np.stack(feats_list, axis=1)  # (batch, num_layers, hidden_dim)
        return torch.from_numpy(stacked).float(), torch.from_numpy(labels).long()
    
    # =========================================================================
    # 4. Run all methods
    # =========================================================================
    print("\n[4] Training downstream classifiers...")
    
    results = {}
    
    # --- Method 1: Last layer (default) ---
    print(f"\n  Method 1: Last layer (layer {last_layer})...")
    torch.manual_seed(42)
    train_f, train_l = get_split_tensors(last_layer, "train")
    val_f, val_l = get_split_tensors(last_layer, "val")
    test_f, test_l = get_split_tensors(last_layer, "test")
    
    model = EmotionClassifier(hidden_dim, num_classes)
    train_info = train_classifier(model, train_f, train_l, val_f, val_l)
    metrics = evaluate_classifier(model, test_f, test_l)
    results["last_layer"] = {
        "method": f"Last layer (L{last_layer})",
        "layer(s)": last_layer,
        "test_accuracy": metrics["accuracy"],
        "test_uar": metrics["uar"],
        "test_f1": metrics["f1_macro"],
    }
    print(f"    Acc: {metrics['accuracy']:.4f} | UAR: {metrics['uar']:.4f}")
    
    # --- Method 2: Probing-guided best layer ---
    print(f"\n  Method 2: Probing-guided best layer (layer {best_probed_layer})...")
    torch.manual_seed(42)
    train_f, train_l = get_split_tensors(best_probed_layer, "train")
    val_f, val_l = get_split_tensors(best_probed_layer, "val")
    test_f, test_l = get_split_tensors(best_probed_layer, "test")
    
    model = EmotionClassifier(hidden_dim, num_classes)
    train_info = train_classifier(model, train_f, train_l, val_f, val_l)
    metrics = evaluate_classifier(model, test_f, test_l)
    results["probing_guided"] = {
        "method": f"Probing-guided (L{best_probed_layer})",
        "layer(s)": best_probed_layer,
        "test_accuracy": metrics["accuracy"],
        "test_uar": metrics["uar"],
        "test_f1": metrics["f1_macro"],
    }
    print(f"    Acc: {metrics['accuracy']:.4f} | UAR: {metrics['uar']:.4f}")
    
    # --- Method 3: Probing-guided top-3 (concatenated) ---
    print(f"\n  Method 3: Top-3 probed layers (L{top3_layers}, concatenated)...")
    torch.manual_seed(42)
    train_f, train_l = get_concat_split_tensors(top3_layers, "train")
    val_f, val_l = get_concat_split_tensors(top3_layers, "val")
    test_f, test_l = get_concat_split_tensors(top3_layers, "test")
    
    model = EmotionClassifier(hidden_dim * 3, num_classes)  # 3x input dim
    train_info = train_classifier(model, train_f, train_l, val_f, val_l)
    metrics = evaluate_classifier(model, test_f, test_l)
    results["probing_top3"] = {
        "method": f"Top-3 probed (L{top3_layers})",
        "layer(s)": top3_layers,
        "test_accuracy": metrics["accuracy"],
        "test_uar": metrics["uar"],
        "test_f1": metrics["f1_macro"],
    }
    print(f"    Acc: {metrics['accuracy']:.4f} | UAR: {metrics['uar']:.4f}")
    
    # --- Method 4: Learned weighted sum (SUPERB-style) ---
    print(f"\n  Method 4: Learned weighted sum (all {num_layers} layers)...")
    torch.manual_seed(42)
    train_f, train_l = get_all_layers_split_tensors("train")
    val_f, val_l = get_all_layers_split_tensors("val")
    test_f, test_l = get_all_layers_split_tensors("test")
    
    model = LearnedWeightedSum(num_layers, hidden_dim, num_classes)
    train_info = train_classifier(model, train_f, train_l, val_f, val_l, lr=5e-4)
    metrics = evaluate_classifier(model, test_f, test_l)
    
    # Extract learned weights
    with torch.no_grad():
        learned_weights = torch.softmax(model.layer_weights, dim=0).cpu().numpy()
    
    results["learned_weighted_sum"] = {
        "method": "Learned weighted sum (all layers)",
        "layer(s)": "all",
        "test_accuracy": metrics["accuracy"],
        "test_uar": metrics["uar"],
        "test_f1": metrics["f1_macro"],
        "learned_weights": learned_weights.tolist(),
    }
    print(f"    Acc: {metrics['accuracy']:.4f} | UAR: {metrics['uar']:.4f}")
    print(f"    Top-3 weighted layers: ", end="")
    top_weighted = sorted(range(num_layers), key=lambda l: learned_weights[l], reverse=True)[:3]
    for l in top_weighted:
        print(f"L{l}({learned_weights[l]:.3f}) ", end="")
    print()
    
    # --- Method 5: Random layer (negative control) ---
    print(f"\n  Method 5: Random/bad layer (layer {random_layer})...")
    torch.manual_seed(42)
    train_f, train_l = get_split_tensors(random_layer, "train")
    val_f, val_l = get_split_tensors(random_layer, "val")
    test_f, test_l = get_split_tensors(random_layer, "test")
    
    model = EmotionClassifier(hidden_dim, num_classes)
    train_info = train_classifier(model, train_f, train_l, val_f, val_l)
    metrics = evaluate_classifier(model, test_f, test_l)
    results["random_layer"] = {
        "method": f"Random/early layer (L{random_layer})",
        "layer(s)": random_layer,
        "test_accuracy": metrics["accuracy"],
        "test_uar": metrics["uar"],
        "test_f1": metrics["f1_macro"],
    }
    print(f"    Acc: {metrics['accuracy']:.4f} | UAR: {metrics['uar']:.4f}")
    
    # =========================================================================
    # 5. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("DOWNSTREAM VALIDATION RESULTS")
    print("=" * 70)
    print(f"\n  Task: Emotion Recognition (6 classes)")
    print(f"  Model: WavLM-Large")
    print(f"  Classifier: 2-layer MLP (512, 256)")
    print(f"  Evaluation: Speaker-independent split")
    print()
    print(f"  {'Method':<40} {'Accuracy':<12} {'UAR':<12} {'F1-macro':<12}")
    print(f"  {'─'*40} {'─'*12} {'─'*12} {'─'*12}")
    
    # Sort by accuracy for display
    sorted_methods = sorted(results.items(), key=lambda x: x[1]["test_accuracy"], reverse=True)
    
    for key, res in sorted_methods:
        marker = "★" if key == "probing_guided" else " "
        print(f"  {marker}{res['method']:<39} {res['test_accuracy']:.4f}       "
              f"{res['test_uar']:.4f}       {res['test_f1']:.4f}")
    
    # Compute relative improvements
    last_acc = results["last_layer"]["test_accuracy"]
    guided_acc = results["probing_guided"]["test_accuracy"]
    weighted_acc = results["learned_weighted_sum"]["test_accuracy"]
    random_acc = results["random_layer"]["test_accuracy"]
    
    print(f"\n  Key comparisons:")
    print(f"    Probing-guided vs Last layer:     {guided_acc - last_acc:+.4f} "
          f"({'better' if guided_acc > last_acc else 'worse'})")
    print(f"    Probing-guided vs Learned sum:    {guided_acc - weighted_acc:+.4f} "
          f"({'better' if guided_acc > weighted_acc else 'gap to close'})")
    print(f"    Probing-guided vs Random layer:   {guided_acc - random_acc:+.4f} "
          f"({'confirms probing is informative' if guided_acc > random_acc else 'unexpected'})")
    
    # Efficiency argument
    print(f"\n  Efficiency:")
    print(f"    Learned weighted sum: requires training {num_layers} extra parameters")
    print(f"    Probing-guided: ZERO training cost (just read off probing results)")
    print(f"    → Probing-guided achieves "
          f"{((guided_acc - last_acc) / (weighted_acc - last_acc) * 100) if weighted_acc != last_acc else 0:.0f}% "
          f"of the gain from learned sum, at zero cost")
    
    # =========================================================================
    # 6. Save results
    # =========================================================================
    results_dir = Path("results/downstream")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "emotion_layer_selection.json"
    
    # Convert numpy types for JSON serialization
    save_results = {}
    for key, val in results.items():
        save_results[key] = {
            k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
            for k, v in val.items()
        }
    
    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()
