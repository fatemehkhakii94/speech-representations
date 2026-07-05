"""
First real experiment: Speaker probing across all layers of WavLM-Large.

This script:
1. Prepares speaker probing data from LibriSpeech
2. Loads audio and extracts features (caches to disk)
3. Trains a linear probe at every layer
4. Reports layer-wise accuracy
5. Produces the first layer-wise probing curve

Usage:
    python scripts/run_speaker_experiment.py
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from src.data.librispeech import (
    scan_librispeech,
    load_audio,
    prepare_speaker_probing,
    create_splits,
)
from src.models.feature_extractor import FeatureExtractor
from src.models.caching import extract_and_cache_utterance_level, load_cached_features, load_metadata
from src.models.probes import create_probe
from src.training.trainer import ProbeTrainer
from src.evaluation.metrics import compute_metrics


def main():
    print("=" * 70)
    print("EXPERIMENT: Speaker Probing on WavLM-Large")
    print("=" * 70)
    
    # =========================================================================
    # 1. PREPARE DATA
    # =========================================================================
    print("\n[PHASE 1] Preparing speaker probing data...")
    
    dataset_path = "data/raw/librispeech/LibriSpeech/train-clean-100"
    
    # Scan all utterances
    utterances = scan_librispeech(dataset_path, max_duration=10.0, min_duration=1.0)
    
    # Prepare for speaker probing (filter speakers, balance)
    selected_utts, speaker_to_class = prepare_speaker_probing(
        utterances,
        max_utterances_per_speaker=50,
        min_utterances_per_speaker=10,
        seed=42,
    )
    
    num_classes = len(speaker_to_class)
    print(f"  Number of speaker classes: {num_classes}")
    
    # Create train/val/test splits
    train_list, val_list, test_list = create_splits(
        selected_utts, speaker_to_class,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        seed=42,
    )
    
    # =========================================================================
    # 2. EXTRACT FEATURES (or load from cache)
    # =========================================================================
    print("\n[PHASE 2] Feature extraction...")
    
    features_dir = Path("features/wavlm_large/speaker")
    h5_path = features_dir / "utterance_level.h5"
    
    if h5_path.exists():
        print(f"  Cached features found at {h5_path}. Skipping extraction.")
        metadata = load_metadata(str(features_dir))
    else:
        print("  Loading WavLM-Large...")
        extractor = FeatureExtractor.from_name("wavlm_large", device="cuda")
        
        # Load all audio files
        print(f"  Loading {len(selected_utts)} audio files...")
        audio_list = []
        for utt in selected_utts:
            waveform = load_audio(utt["path"])
            audio_list.append((utt["utterance_id"], waveform))
        
        print(f"  Extracting features (this may take a few minutes)...")
        extract_and_cache_utterance_level(
            extractor=extractor,
            audio_list=audio_list,
            output_dir=str(features_dir),
            batch_size=4,
        )
        
        # Free GPU memory
        del extractor
        torch.cuda.empty_cache()
        
        metadata = load_metadata(str(features_dir))
    
    # =========================================================================
    # 3. BUILD FEATURE MATRICES FOR PROBING
    # =========================================================================
    print("\n[PHASE 3] Building probing matrices...")
    
    # Map utterance IDs to indices in the cached features
    utt_id_to_idx = {
        uid: idx for idx, uid in enumerate(metadata["utterance_ids"])
    }
    
    def get_indices_and_labels(data_list):
        """Convert list of (utt_dict, label) to (indices, labels) arrays."""
        indices = []
        labels = []
        for utt_dict, label in data_list:
            uid = utt_dict["utterance_id"]
            if uid in utt_id_to_idx:
                indices.append(utt_id_to_idx[uid])
                labels.append(label)
        return np.array(indices), np.array(labels)
    
    train_indices, train_labels = get_indices_and_labels(train_list)
    val_indices, val_labels = get_indices_and_labels(val_list)
    test_indices, test_labels = get_indices_and_labels(test_list)
    
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val:   {len(val_indices)} samples")
    print(f"  Test:  {len(test_indices)} samples")
    print(f"  Classes: {num_classes}")
    
    # =========================================================================
    # 4. PROBE EVERY LAYER
    # =========================================================================
    print("\n[PHASE 4] Probing all layers...")
    print(f"  Probe type: linear")
    print(f"  Layers: 0-24 (25 total)")
    print()
    
    num_layers = metadata["num_layers"]
    hidden_dim = metadata["hidden_dim"]
    
    results = {}
    
    for layer_idx in range(num_layers):
        # Load features for this layer
        all_features = load_cached_features(str(features_dir), layer=layer_idx)
        
        # Select train/val/test subsets
        train_features = all_features[train_indices]
        val_features = all_features[val_indices]
        test_features = all_features[test_indices]
        
        # Create and train probe
        probe = create_probe("linear", hidden_dim, num_classes)
        trainer = ProbeTrainer(
            probe=probe,
            device="cuda",
            learning_rate=1e-3,
            batch_size=256,
            max_epochs=50,
            patience=5,
        )
        
        result = trainer.train(
            train_features, train_labels,
            val_features, val_labels,
            test_features, test_labels,
        )
        
        results[layer_idx] = {
            "test_accuracy": result.test_accuracy,
            "val_accuracy": result.val_accuracy,
            "train_accuracy": result.train_accuracy,
            "best_epoch": result.best_epoch,
            "training_time": result.training_time,
        }
        
        # Progress output
        bar_len = int(result.test_accuracy * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  Layer {layer_idx:2d}: [{bar}] {result.test_accuracy:.4f} "
              f"(train={result.train_accuracy:.4f}, epoch={result.best_epoch})")
    
    # =========================================================================
    # 5. SUMMARY AND SAVE
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Find best layer
    best_layer = max(results, key=lambda l: results[l]["test_accuracy"])
    best_acc = results[best_layer]["test_accuracy"]
    
    # Find worst layer
    worst_layer = min(results, key=lambda l: results[l]["test_accuracy"])
    worst_acc = results[worst_layer]["test_accuracy"]
    
    print(f"\n  Best layer:  {best_layer} (accuracy: {best_acc:.4f})")
    print(f"  Worst layer: {worst_layer} (accuracy: {worst_acc:.4f})")
    print(f"  Spread:      {best_acc - worst_acc:.4f}")
    
    # Majority baseline
    _, counts = np.unique(test_labels, return_counts=True)
    majority_baseline = counts.max() / len(test_labels)
    print(f"  Majority baseline: {majority_baseline:.4f}")
    print(f"  Best selectivity:  {best_acc - majority_baseline:.4f}")
    
    # Save results
    results_dir = Path("results/probing/speaker")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "model": "wavlm_large",
        "property": "speaker",
        "probe_type": "linear",
        "num_classes": num_classes,
        "num_train": len(train_indices),
        "num_val": len(val_indices),
        "num_test": len(test_indices),
        "majority_baseline": float(majority_baseline),
        "layer_results": {str(k): v for k, v in results.items()},
    }
    
    output_path = results_dir / "wavlm_large_linear.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()
