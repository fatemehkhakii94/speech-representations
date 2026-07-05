"""
Visualization module for probing results.

Generates:
1. Layer-wise accuracy curves (per model, per property)
2. Information landscape heatmaps (per model, all properties)
3. Cross-model comparison plots
4. Information trajectory plots (build-up / suppression narrative)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Use non-interactive backend for saving figures
matplotlib.use("Agg")

# Consistent style
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "wavlm_large": "#2196F3",       # Blue
    "hubert_large": "#FF9800",       # Orange
    "whisper_large_v3": "#4CAF50",   # Green
}
MODEL_LABELS = {
    "wavlm_large": "WavLM-Large",
    "hubert_large": "HuBERT-Large",
    "whisper_large_v3": "Whisper-Large-v3",
}
PROPERTY_LABELS = {
    "speaker": "Speaker Identity",
    "phoneme": "Phoneme",
    "pitch": "Pitch (F0)",
    "noise": "Noise Type",
    "emotion": "Emotion",
}


def load_probing_results(results_dir: str = "results/probing") -> Dict:
    """
    Load all probing results from the results directory.
    
    Returns nested dict: results[property][model][probe_type] = {...}
    """
    results_dir = Path(results_dir)
    all_results = {}
    
    for property_dir in sorted(results_dir.iterdir()):
        if not property_dir.is_dir():
            continue
        property_name = property_dir.name
        all_results[property_name] = {}
        
        for result_file in sorted(property_dir.glob("*.json")):
            with open(result_file) as f:
                data = json.load(f)
            
            model = data["model"]
            probe_type = data["probe_type"]
            
            if model not in all_results[property_name]:
                all_results[property_name][model] = {}
            all_results[property_name][model][probe_type] = data
    
    return all_results


def get_layer_accuracies(result: dict) -> np.ndarray:
    """Extract layer-wise test accuracies from a result dict."""
    layer_results = result["layer_results"]
    num_layers = len(layer_results)
    accs = np.array([
        layer_results[str(i)]["test_accuracy"] for i in range(num_layers)
    ])
    return accs


# =============================================================================
# PLOT 1: Layer-wise accuracy curves (one property, all models)
# =============================================================================

def plot_layer_curves_by_model(
    all_results: Dict,
    property_name: str,
    probe_type: str = "linear",
    save_path: Optional[str] = None,
):
    """
    Plot layer-wise accuracy for one property across all models.
    
    This is the main comparison plot: same property, different models.
    Shows how pretraining objectives shape information encoding.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    property_results = all_results.get(property_name, {})
    
    for model_name, model_results in property_results.items():
        if probe_type not in model_results:
            continue
        
        result = model_results[probe_type]
        accs = get_layer_accuracies(result)
        layers = np.arange(len(accs))
        
        # Normalize layer indices to [0, 1] for cross-model comparison
        # (WavLM has 25 layers, Whisper has 33)
        color = COLORS.get(model_name, "#666666")
        label = MODEL_LABELS.get(model_name, model_name)
        
        ax.plot(layers, accs, color=color, linewidth=2.5, label=label,
                marker="o", markersize=4, alpha=0.9)
        
        # Mark the peak
        best_layer = np.argmax(accs)
        ax.scatter([best_layer], [accs[best_layer]], color=color,
                   s=120, zorder=5, edgecolors="white", linewidths=2)
    
    # Add majority baseline
    if property_results:
        first_result = next(iter(next(iter(property_results.values())).values()))
        baseline = first_result.get("majority_baseline", 0)
        ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5,
                   label=f"Majority baseline ({baseline:.3f})")
    
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(
        f"Layer-wise Probing: {PROPERTY_LABELS.get(property_name, property_name)}\n"
        f"(probe: {probe_type})",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11, loc="best")
    ax.set_ylim(bottom=max(0, baseline - 0.05) if property_results else 0)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    plt.close()


# =============================================================================
# PLOT 2: Information landscape heatmap (one model, all properties)
# =============================================================================

def plot_information_landscape(
    all_results: Dict,
    model_name: str,
    probe_type: str = "linear",
    save_path: Optional[str] = None,
):
    """
    Heatmap showing all properties across all layers for one model.
    
    This is the "model personality" visualization.
    Each row is a property, each column is a layer.
    Color intensity = probe accuracy.
    """
    # Collect data
    properties = []
    accuracies = []
    
    for prop_name in ["speaker", "phoneme", "pitch", "noise", "emotion"]:
        if prop_name not in all_results:
            continue
        if model_name not in all_results[prop_name]:
            continue
        if probe_type not in all_results[prop_name][model_name]:
            continue
        
        result = all_results[prop_name][model_name][probe_type]
        accs = get_layer_accuracies(result)
        
        # Normalize to [0, 1] relative to this property's range
        # (so properties with different baseline difficulty are comparable)
        baseline = result.get("majority_baseline", 0)
        normalized = (accs - baseline) / (accs.max() - baseline + 1e-8)
        
        properties.append(PROPERTY_LABELS.get(prop_name, prop_name))
        accuracies.append(normalized)
    
    if not properties:
        print(f"  No results found for {model_name}")
        return
    
    # Create heatmap
    data = np.array(accuracies)
    
    fig, ax = plt.subplots(figsize=(14, max(3, len(properties) * 1.2)))
    
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    
    # Labels
    ax.set_yticks(range(len(properties)))
    ax.set_yticklabels(properties, fontsize=11)
    ax.set_xlabel("Layer Index", fontsize=12)
    
    # Add layer numbers
    num_layers = data.shape[1]
    tick_positions = np.arange(0, num_layers, max(1, num_layers // 10))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions)
    
    ax.set_title(
        f"Information Landscape: {MODEL_LABELS.get(model_name, model_name)}\n"
        f"(normalized selectivity, probe: {probe_type})",
        fontsize=14, fontweight="bold"
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Normalized Selectivity", fontsize=10)
    
    # Mark peak per property
    for i, accs_row in enumerate(data):
        peak = np.argmax(accs_row)
        ax.scatter([peak], [i], color="black", marker=">", s=80, zorder=5)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    plt.close()


# =============================================================================
# PLOT 3: Cross-model comparison (normalized layer position)
# =============================================================================

def plot_cross_model_normalized(
    all_results: Dict,
    property_name: str,
    probe_type: str = "linear",
    save_path: Optional[str] = None,
):
    """
    Compare models with normalized layer position (0-100% depth).
    
    Since WavLM has 25 layers and Whisper has 33, we normalize
    to relative depth for fair comparison.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    property_results = all_results.get(property_name, {})
    
    for model_name, model_results in property_results.items():
        if probe_type not in model_results:
            continue
        
        result = model_results[probe_type]
        accs = get_layer_accuracies(result)
        
        # Normalize x-axis to [0, 1] (relative depth)
        num_layers = len(accs)
        relative_depth = np.linspace(0, 1, num_layers)
        
        color = COLORS.get(model_name, "#666666")
        label = MODEL_LABELS.get(model_name, model_name)
        
        ax.plot(relative_depth * 100, accs, color=color, linewidth=2.5,
                label=label, alpha=0.9)
    
    ax.set_xlabel("Relative Depth (%)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(
        f"Cross-Model Comparison: {PROPERTY_LABELS.get(property_name, property_name)}\n"
        f"(normalized depth, probe: {probe_type})",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    plt.close()


# =============================================================================
# PLOT 4: All properties for all models (summary figure)
# =============================================================================

def plot_full_summary(
    all_results: Dict,
    probe_type: str = "linear",
    save_path: Optional[str] = None,
):
    """
    Multi-panel figure: one subplot per property, all models overlaid.
    
    This is the main result figure for the project.
    """
    available_properties = [
        p for p in ["speaker", "phoneme", "pitch", "noise", "emotion"]
        if p in all_results
    ]
    
    if not available_properties:
        print("  No results to plot.")
        return
    
    n_props = len(available_properties)
    fig, axes = plt.subplots(1, n_props, figsize=(5 * n_props, 5), sharey=False)
    
    if n_props == 1:
        axes = [axes]
    
    for ax, prop_name in zip(axes, available_properties):
        property_results = all_results[prop_name]
        
        for model_name, model_results in property_results.items():
            if probe_type not in model_results:
                continue
            
            result = model_results[probe_type]
            accs = get_layer_accuracies(result)
            layers = np.arange(len(accs))
            
            color = COLORS.get(model_name, "#666666")
            label = MODEL_LABELS.get(model_name, model_name)
            ax.plot(layers, accs, color=color, linewidth=2, label=label, alpha=0.9)
        
        ax.set_title(PROPERTY_LABELS.get(prop_name, prop_name), fontsize=12,
                     fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8)
    
    plt.suptitle(
        "Information Landscape Across Speech Foundation Models",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    plt.close()


# =============================================================================
# PLOT 5: CKA Heatmaps
# =============================================================================

def plot_cka_heatmap(
    cka_matrix: "np.ndarray",
    title: str,
    xlabel: str = "Layer (Model B)",
    ylabel: str = "Layer (Model A)",
    save_path: Optional[str] = None,
):
    """
    Plot a CKA similarity heatmap.
    
    Args:
        cka_matrix: (num_layers_a, num_layers_b) array of CKA values
        title: Plot title
        xlabel/ylabel: Axis labels
        save_path: Where to save
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(
        cka_matrix,
        cmap="magma",
        vmin=0,
        vmax=1,
        aspect="auto",
        origin="lower",
    )
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Add tick labels at intervals
    num_a, num_b = cka_matrix.shape
    step_a = max(1, num_a // 8)
    step_b = max(1, num_b // 8)
    ax.set_xticks(range(0, num_b, step_b))
    ax.set_xticklabels(range(0, num_b, step_b))
    ax.set_yticks(range(0, num_a, step_a))
    ax.set_yticklabels(range(0, num_a, step_a))
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CKA Similarity", fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    plt.close()


def plot_all_cka(results_dir: str = "results/cka", figures_dir: str = "results/figures"):
    """Generate all CKA heatmap figures."""
    import numpy as np
    
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    
    model_labels = {
        "wavlm_large": "WavLM-Large",
        "hubert_large": "HuBERT-Large",
        "whisper_large_v3": "Whisper-Large-v3",
    }
    
    # Within-model CKA
    for model_name in ["wavlm_large", "hubert_large", "whisper_large_v3"]:
        path = results_dir / f"within_{model_name}.npy"
        if path.exists():
            matrix = np.load(path)
            label = model_labels[model_name]
            plot_cka_heatmap(
                matrix,
                title=f"Within-Model CKA: {label}",
                xlabel=f"Layer ({label})",
                ylabel=f"Layer ({label})",
                save_path=str(figures_dir / f"cka_within_{model_name}.png"),
            )
    
    # Cross-model CKA
    model_pairs = [
        ("wavlm_large", "hubert_large"),
        ("wavlm_large", "whisper_large_v3"),
        ("hubert_large", "whisper_large_v3"),
    ]
    
    for model_a, model_b in model_pairs:
        path = results_dir / f"cross_{model_a}_vs_{model_b}.npy"
        if path.exists():
            matrix = np.load(path)
            label_a = model_labels[model_a]
            label_b = model_labels[model_b]
            plot_cka_heatmap(
                matrix,
                title=f"Cross-Model CKA: {label_a} vs {label_b}",
                xlabel=f"Layer ({label_b})",
                ylabel=f"Layer ({label_a})",
                save_path=str(figures_dir / f"cka_cross_{model_a}_vs_{model_b}.png"),
            )
