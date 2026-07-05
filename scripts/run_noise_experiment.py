"""
Noise type probing experiment.

Creates synthetic noisy speech (LibriSpeech + MUSAN), extracts features,
and probes whether models encode noise type information.

Usage:
    python scripts/run_noise_experiment.py --model wavlm_large
    python scripts/run_noise_experiment.py --model all
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from src.data.librispeech import scan_librispeech
from src.data.noise_mixing import (
    load_noise_files,
    create_noisy_dataset,
    build_noise_probing_dataset,
    NOISE_NAMES,
)
from src.models.feature_extractor import FeatureExtractor
from src.models.caching import extract_and_cache_utterance_level, load_metadata
from src.models.probes import create_probe
from src.training.trainer import ProbeTrainer
from src.config import get_model_config


def run_noise_probing(model_name: str, probe_type: str = "linear"):
    """Run noise type probing for one model."""
    
    print(f"\n{'='*70}")
    print(f"NOISE PROBING: {model_name} / {probe_type}")
    print(f"{'='*70}")
    
    # --- Step 1: Prepare noisy dataset ---
    print("\n[1] Preparing noisy dataset...")
    
    # Scan LibriSpeech for clean source audio
    dataset_path = "data/raw/librispeech/LibriSpeech/train-clean-100"
    utterances = scan_librispeech(dataset_path, max_duration=10.0, min_duration=1.0)
    
    # Load MUSAN noise files
    print("\n  Loading MUSAN noise sources...")
    musan_dir = "data/raw/musan/musan"
    noise_files = load_noise_files(musan_dir)
    
    # Create noisy mixtures
    dataset_items = create_noisy_dataset(
        utterances=utterances,
        noise_files=noise_files,
        snr_db=10.0,
        max_utterances=1200,
        seed=42,
    )
    
    # --- Step 2: Extract features ---
    print("\n[2] Feature extraction...")
    features_dir = f"features/{model_name}/noisy"
    h5_path = Path(features_dir) / "utterance_level.h5"
    
    if h5_path.exists():
        metadata = load_metadata(features_dir)
        print(f"  [CACHED] Features exist: {features_dir} "
              f"({metadata['num_utterances']} utterances)")
    else:
        print(f"  Loading {model_name}...")
        extractor = FeatureExtractor.from_name(model_name, device="cuda")
        
        # Prepare audio list (already have waveforms from create_noisy_dataset)
        audio_list = [(utt_id, waveform) for utt_id, waveform, _ in dataset_items]
        
        print(f"  Extracting features for {len(audio_list)} noisy utterances...")
        extract_and_cache_utterance_level(
            extractor=extractor,
            audio_list=audio_list,
            output_dir=features_dir,
            batch_size=4,
        )
        
        del extractor
        torch.cuda.empty_cache()
    
    # --- Step 3: Build probing dataset ---
    print("\n[3] Building noise probing dataset...")
    dataset = build_noise_probing_dataset(
        features_dir=features_dir,
        dataset_items=dataset_items,
        seed=42,
    )
    print(f"  {dataset.summary()}")
    
    # --- Step 4: Probe all layers ---
    print("\n[4] Probing all layers...")
    model_config = get_model_config(model_name)
    num_layers = model_config.num_layers + 1
    hidden_dim = model_config.hidden_dim
    
    layer_results = {}
    
    for layer_idx in range(num_layers):
        train_features, train_labels = dataset.get_split(layer_idx, "train")
        val_features, val_labels = dataset.get_split(layer_idx, "val")
        test_features, test_labels = dataset.get_split(layer_idx, "test")
        
        torch.manual_seed(42)
        probe = create_probe(probe_type, hidden_dim, dataset.num_classes)
        
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
        
        layer_results[layer_idx] = {
            "test_accuracy": result.test_accuracy,
            "val_accuracy": result.val_accuracy,
            "train_accuracy": result.train_accuracy,
            "best_epoch": result.best_epoch,
            "training_time": result.training_time,
        }
        
        bar_len = int(result.test_accuracy * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  Layer {layer_idx:2d}: [{bar}] {result.test_accuracy:.4f}")
    
    # --- Summary ---
    accs = [layer_results[l]["test_accuracy"] for l in range(num_layers)]
    best_layer = int(np.argmax(accs))
    
    _, test_labels = dataset.get_split(0, "test")
    _, counts = np.unique(test_labels, return_counts=True)
    majority_baseline = float(counts.max() / len(test_labels))
    
    print(f"\n  Best: layer {best_layer} ({accs[best_layer]:.4f})")
    print(f"  Worst: layer {int(np.argmin(accs))} ({min(accs):.4f})")
    print(f"  Majority baseline: {majority_baseline:.4f}")
    print(f"  Best selectivity: {accs[best_layer] - majority_baseline:.4f}")
    
    # --- Save ---
    results_dir = Path("results/probing/noise")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "model": model_name,
        "property": "noise",
        "probe_type": probe_type,
        "num_classes": dataset.num_classes,
        "num_train": int(dataset.num_train),
        "num_val": int(dataset.num_val),
        "num_test": int(dataset.num_test),
        "majority_baseline": majority_baseline,
        "best_layer": best_layer,
        "best_accuracy": float(accs[best_layer]),
        "snr_db": 10.0,
        "noise_categories": NOISE_NAMES,
        "layer_results": {str(k): v for k, v in layer_results.items()},
    }
    
    output_path = results_dir / f"{model_name}_{probe_type}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="wavlm_large",
                        choices=["wavlm_large", "hubert_large", "whisper_large_v3", "all"])
    parser.add_argument("--probe", type=str, default="linear",
                        choices=["linear", "mlp"])
    args = parser.parse_args()
    
    if args.model == "all":
        for model in ["wavlm_large", "hubert_large", "whisper_large_v3"]:
            run_noise_probing(model, args.probe)
    else:
        run_noise_probing(args.model, args.probe)
    
    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
