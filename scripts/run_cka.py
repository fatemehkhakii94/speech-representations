"""
Compute CKA similarity matrices between models.

Uses LibriSpeech features (shared across all models) to compute
layer-wise structural similarity.

Produces:
- Within-model CKA (which layers are similar to each other?)
- Cross-model CKA (how do models relate to each other?)

Usage:
    python scripts/run_cka.py
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, ".")
from src.models.caching import load_cached_features, load_metadata
from src.analysis.cka import compute_cka, compute_cka_matrix
from src.config import get_model_config


def load_all_layers(features_dir: str, num_layers: int) -> dict:
    """Load features from all layers into a dict."""
    features = {}
    for layer_idx in range(num_layers):
        features[layer_idx] = load_cached_features(features_dir, layer=layer_idx)
    return features


def run_cka_analysis():
    print("=" * 70)
    print("CKA ANALYSIS: Cross-Model Structural Similarity")
    print("=" * 70)
    
    # We use LibriSpeech features (shared audio across all models)
    models = {
        "wavlm_large": "features/wavlm_large/librispeech",
        "hubert_large": "features/hubert_large/librispeech",
        "whisper_large_v3": "features/whisper_large_v3/librispeech",
    }
    
    # Verify all features exist
    print("\n[1] Checking feature caches...")
    for model_name, features_dir in models.items():
        if not Path(features_dir).exists():
            print(f"  [ERROR] Missing: {features_dir}")
            print(f"  Run: python scripts/run_probing.py --model {model_name} --property speaker")
            sys.exit(1)
        metadata = load_metadata(features_dir)
        print(f"  [OK] {model_name}: {metadata['num_utterances']} utterances, "
              f"{metadata['num_layers']} layers")
    
    # Load all features
    print("\n[2] Loading features into memory...")
    all_features = {}
    for model_name, features_dir in models.items():
        config = get_model_config(model_name)
        num_layers = config.num_layers + 1
        print(f"  Loading {model_name} ({num_layers} layers)...")
        all_features[model_name] = load_all_layers(features_dir, num_layers)
    
    # Number of samples for CKA (1000 is standard — balances speed vs accuracy)
    num_samples = 1000
    print(f"\n  Using {num_samples} samples for CKA computation")
    
    results_dir = Path("results/cka")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # WITHIN-MODEL CKA
    # =========================================================================
    print("\n[3] Computing within-model CKA...")
    
    for model_name in models:
        config = get_model_config(model_name)
        num_layers = config.num_layers + 1
        features = all_features[model_name]
        
        print(f"\n  {model_name} ({num_layers}×{num_layers} matrix)...")
        t0 = time.time()
        
        cka_matrix = compute_cka_matrix(
            features_a=features,
            features_b=features,
            num_layers_a=num_layers,
            num_layers_b=num_layers,
            num_samples=num_samples,
        )
        
        elapsed = time.time() - t0
        print(f"  Computed in {elapsed:.1f}s")
        
        # Save
        output_path = results_dir / f"within_{model_name}.npy"
        np.save(output_path, cka_matrix)
        print(f"  Saved: {output_path}")
    
    # =========================================================================
    # CROSS-MODEL CKA
    # =========================================================================
    print("\n[4] Computing cross-model CKA...")
    
    model_pairs = [
        ("wavlm_large", "hubert_large"),
        ("wavlm_large", "whisper_large_v3"),
        ("hubert_large", "whisper_large_v3"),
    ]
    
    for model_a, model_b in model_pairs:
        config_a = get_model_config(model_a)
        config_b = get_model_config(model_b)
        num_layers_a = config_a.num_layers + 1
        num_layers_b = config_b.num_layers + 1
        
        print(f"\n  {model_a} vs {model_b} ({num_layers_a}×{num_layers_b})...")
        t0 = time.time()
        
        # Need to ensure same samples across models
        # Use min available samples
        n_a = all_features[model_a][0].shape[0]
        n_b = all_features[model_b][0].shape[0]
        n_common = min(n_a, n_b, num_samples)
        
        # Subsample consistently
        rng = np.random.RandomState(42)
        indices = rng.choice(min(n_a, n_b), n_common, replace=False)
        
        # Build subsampled feature dicts
        features_a = {l: all_features[model_a][l][indices] for l in range(num_layers_a)}
        features_b = {l: all_features[model_b][l][indices] for l in range(num_layers_b)}
        
        cka_matrix = compute_cka_matrix(
            features_a=features_a,
            features_b=features_b,
            num_layers_a=num_layers_a,
            num_layers_b=num_layers_b,
            num_samples=n_common,  # Already subsampled
        )
        
        elapsed = time.time() - t0
        print(f"  Computed in {elapsed:.1f}s")
        
        # Save
        output_path = results_dir / f"cross_{model_a}_vs_{model_b}.npy"
        np.save(output_path, cka_matrix)
        print(f"  Saved: {output_path}")
    
    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    print("\n[5] Summary statistics...")
    
    for model_a, model_b in model_pairs:
        output_path = results_dir / f"cross_{model_a}_vs_{model_b}.npy"
        cka_matrix = np.load(output_path)
        
        # Find maximum alignment per row (which layer in B is most similar to each layer in A)
        max_per_row = cka_matrix.max(axis=1)
        best_match = cka_matrix.argmax(axis=1)
        
        print(f"\n  {model_a} vs {model_b}:")
        print(f"    Mean CKA: {cka_matrix.mean():.4f}")
        print(f"    Max CKA:  {cka_matrix.max():.4f}")
        print(f"    Layer alignment (A→B best match): ", end="")
        # Show every 4th layer
        config_a = get_model_config(model_a)
        step = max(1, (config_a.num_layers + 1) // 6)
        for i in range(0, config_a.num_layers + 1, step):
            print(f"L{i}→L{best_match[i]}({max_per_row[i]:.2f}) ", end="")
        print()
    
    print("\n[DONE]")


if __name__ == "__main__":
    run_cka_analysis()
