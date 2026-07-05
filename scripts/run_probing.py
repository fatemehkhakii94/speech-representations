"""
General-purpose probing experiment runner.

Runs probing for a specific (model, property, probe_type) combination
across all layers. Saves results to JSON.

Usage:
    python scripts/run_probing.py --model wavlm_large --property speaker --probe linear
    python scripts/run_probing.py --model hubert_large --property speaker --probe linear
    python scripts/run_probing.py --model whisper_large_v3 --property speaker --probe linear
    python scripts/run_probing.py --model wavlm_large --property speaker --probe mlp
"""

import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from src.data.librispeech import scan_librispeech, load_audio
from src.data.probing_dataset import (
    ProbingDataset,
    build_speaker_probing_dataset,
    build_pitch_probing_dataset,
)
from src.models.feature_extractor import FeatureExtractor
from src.models.caching import extract_and_cache_utterance_level, load_metadata
from src.models.probes import create_probe
from src.training.trainer import ProbeTrainer
from src.config import get_model_config


# =============================================================================
# FEATURE EXTRACTION (shared across properties)
# =============================================================================

def ensure_features_extracted(
    model_name: str,
    dataset_name: str,
    utterances: list,
) -> str:
    """
    Ensure features are extracted and cached. Returns features_dir.
    
    If features already exist, skip extraction.
    """
    features_dir = f"features/{model_name}/{dataset_name}"
    h5_path = Path(features_dir) / "utterance_level.h5"
    
    if h5_path.exists():
        metadata = load_metadata(features_dir)
        print(f"  [CACHED] Features exist: {features_dir} "
              f"({metadata['num_utterances']} utterances)")
        return features_dir
    
    print(f"  Extracting features for {model_name} on {dataset_name}...")
    print(f"  Loading model...")
    extractor = FeatureExtractor.from_name(model_name, device="cuda")
    
    # Load audio
    print(f"  Loading {len(utterances)} audio files...")
    audio_list = []
    for utt in utterances:
        waveform = load_audio(utt["path"])
        audio_list.append((utt["utterance_id"], waveform))
    
    # Extract and cache
    extract_and_cache_utterance_level(
        extractor=extractor,
        audio_list=audio_list,
        output_dir=features_dir,
        batch_size=4,
    )
    
    # Free GPU
    del extractor
    torch.cuda.empty_cache()
    
    return features_dir


# =============================================================================
# PROBING LOOP
# =============================================================================

def run_probing_experiment(
    dataset: ProbingDataset,
    model_name: str,
    probe_type: str = "linear",
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 50,
    patience: int = 5,
    seed: int = 42,
) -> dict:
    """
    Run probing across all layers for a given dataset.
    
    Returns results dictionary with layer-wise metrics.
    """
    model_config = get_model_config(model_name)
    num_layers = model_config.num_layers + 1  # +1 for input embedding
    hidden_dim = model_config.hidden_dim
    
    print(f"\n  Probing: model={model_name}, property={dataset.property_name}, "
          f"probe={probe_type}")
    print(f"  Layers: {num_layers}, Classes: {dataset.num_classes}")
    print(f"  Train/Val/Test: {dataset.num_train}/{dataset.num_val}/{dataset.num_test}")
    print()
    
    layer_results = {}
    
    for layer_idx in range(num_layers):
        # Get data for this layer
        train_features, train_labels = dataset.get_split(layer_idx, "train")
        val_features, val_labels = dataset.get_split(layer_idx, "val")
        test_features, test_labels = dataset.get_split(layer_idx, "test")
        
        # Create probe
        torch.manual_seed(seed)
        probe = create_probe(
            probe_type=probe_type,
            input_dim=hidden_dim,
            num_classes=dataset.num_classes,
        )
        
        # Train
        trainer = ProbeTrainer(
            probe=probe,
            device="cuda",
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
        )
        
        result = trainer.train(
            train_features, train_labels,
            val_features, val_labels,
            test_features, test_labels,
        )
        
        layer_results[layer_idx] = {
            "test_accuracy": result.test_accuracy,
            "val_accuracy": result.val_accuracy,
            "train_accuracy": result.train_accuracy,
            "best_epoch": result.best_epoch,
            "training_time": result.training_time,
        }
        
        # Progress
        bar_len = int(result.test_accuracy * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  Layer {layer_idx:2d}: [{bar}] {result.test_accuracy:.4f}")
    
    # Summary
    accs = [layer_results[l]["test_accuracy"] for l in range(num_layers)]
    best_layer = int(np.argmax(accs))
    
    # Majority baseline
    _, test_labels = dataset.get_split(0, "test")
    _, counts = np.unique(test_labels, return_counts=True)
    majority_baseline = float(counts.max() / len(test_labels))
    
    results = {
        "model": model_name,
        "property": dataset.property_name,
        "probe_type": probe_type,
        "num_classes": dataset.num_classes,
        "num_train": int(dataset.num_train),
        "num_val": int(dataset.num_val),
        "num_test": int(dataset.num_test),
        "majority_baseline": majority_baseline,
        "best_layer": best_layer,
        "best_accuracy": float(accs[best_layer]),
        "layer_results": {str(k): v for k, v in layer_results.items()},
    }
    
    print(f"\n  Best: layer {best_layer} ({accs[best_layer]:.4f})")
    print(f"  Worst: layer {int(np.argmin(accs))} ({min(accs):.4f})")
    print(f"  Majority baseline: {majority_baseline:.4f}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run probing experiment")
    parser.add_argument("--model", type=str, required=True,
                        choices=["wavlm_large", "hubert_large", "whisper_large_v3"])
    parser.add_argument("--property", type=str, required=True,
                        choices=["speaker", "pitch"])
    parser.add_argument("--probe", type=str, default="linear",
                        choices=["linear", "mlp"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"PROBING EXPERIMENT: {args.model} / {args.property} / {args.probe}")
    print("=" * 70)
    
    # --- Step 1: Load LibriSpeech utterances ---
    print("\n[1] Scanning LibriSpeech...")
    dataset_path = "data/raw/librispeech/LibriSpeech/train-clean-100"
    utterances = scan_librispeech(dataset_path, max_duration=10.0, min_duration=1.0)
    
    # --- Step 2: Ensure features are extracted ---
    print("\n[2] Ensuring features are extracted...")
    features_dir = ensure_features_extracted(
        model_name=args.model,
        dataset_name="librispeech",
        utterances=utterances,
    )
    
    # --- Step 3: Build probing dataset ---
    print("\n[3] Building probing dataset...")
    if args.property == "speaker":
        dataset = build_speaker_probing_dataset(
            features_dir=features_dir,
            utterances=utterances,
            seed=args.seed,
        )
    elif args.property == "pitch":
        dataset = build_pitch_probing_dataset(
            features_dir=features_dir,
            utterances=utterances,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Property '{args.property}' not yet implemented")
    
    print(f"  {dataset.summary()}")
    
    # --- Step 4: Run probing ---
    print("\n[4] Running probing experiment...")
    results = run_probing_experiment(
        dataset=dataset,
        model_name=args.model,
        probe_type=args.probe,
        seed=args.seed,
    )
    
    # --- Step 5: Save results ---
    results_dir = Path("results/probing") / args.property
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{args.model}_{args.probe}.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()
