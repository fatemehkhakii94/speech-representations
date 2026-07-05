"""
Feature caching system.

Extracts hidden states from speech models and saves them to disk
in an efficient format for probing experiments.

Storage layout:
    features/{model_name}/
        utterance_level.h5     # Mean-pooled features for all utterances
        frame_level.h5         # Full frame features (for phoneme probing)
        metadata.json          # Sample IDs, lengths, labels

HDF5 internal structure:
    utterance_level.h5:
        /layer_00  -> (num_utterances, hidden_dim)
        /layer_01  -> (num_utterances, hidden_dim)
        ...
        /layer_24  -> (num_utterances, hidden_dim)

    frame_level.h5:
        /layer_00/{utterance_id}  -> (num_frames, hidden_dim)
        ...
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.models.feature_extractor import FeatureExtractor


def extract_and_cache_utterance_level(
    extractor: FeatureExtractor,
    audio_list: List[Tuple[str, torch.Tensor]],
    output_dir: str,
    batch_size: int = 4,
) -> str:
    """
    Extract mean-pooled features for all utterances and save to HDF5.
    
    Args:
        extractor: Configured FeatureExtractor instance
        audio_list: List of (utterance_id, waveform_tensor) tuples
                    waveform_tensor shape: (num_samples,)
        output_dir: Directory to save the HDF5 file
        batch_size: Number of utterances to process at once
    
    Returns:
        Path to the saved HDF5 file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "utterance_level.h5"
    
    num_layers = extractor.num_layers + 1  # +1 for input embedding
    hidden_dim = extractor.hidden_dim
    num_utterances = len(audio_list)
    
    print(f"\nExtracting utterance-level features:")
    print(f"  Model: {extractor.name}")
    print(f"  Utterances: {num_utterances}")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Output: {output_path}")
    
    # Pre-allocate arrays for all layers
    # Shape: (num_layers, num_utterances, hidden_dim)
    all_features = np.zeros(
        (num_layers, num_utterances, hidden_dim), dtype=np.float32
    )
    
    # Track actual frame counts (important for Whisper masking)
    frame_counts = []
    utterance_ids = []
    
    # Process in batches
    t0 = time.time()
    for batch_start in tqdm(
        range(0, num_utterances, batch_size),
        desc="Extracting",
        unit="batch",
    ):
        batch_end = min(batch_start + batch_size, num_utterances)
        batch_items = audio_list[batch_start:batch_end]
        
        # Collect IDs
        batch_ids = [item[0] for item in batch_items]
        utterance_ids.extend(batch_ids)
        
        # Pad waveforms to same length within batch
        waveforms = [item[1] for item in batch_items]
        max_len = max(w.shape[0] for w in waveforms)
        actual_lengths = [w.shape[0] for w in waveforms]
        
        # Pad and stack
        padded = torch.zeros(len(waveforms), max_len)
        for i, w in enumerate(waveforms):
            padded[i, :w.shape[0]] = w
        
        # Extract hidden states
        hidden_states = extractor(padded, sample_rate=16000)
        
        # Mean pool each layer (only over actual frames, not padding)
        for layer_idx, layer_hidden in enumerate(hidden_states):
            # layer_hidden shape: (batch, num_frames, hidden_dim)
            for i, actual_len in enumerate(actual_lengths):
                # Calculate actual number of frames for this utterance
                if extractor.name == "whisper_large_v3":
                    # Whisper: 30s → 1500 frames, so scale proportionally
                    actual_frames = int(
                        (actual_len / (30 * 16000)) * 1500
                    )
                    actual_frames = max(1, min(actual_frames, layer_hidden.shape[1]))
                else:
                    # WavLM/HuBERT: ~320 samples per frame
                    actual_frames = actual_len // 320
                    actual_frames = max(1, min(actual_frames, layer_hidden.shape[1]))
                
                # Mean pool over actual frames only
                pooled = layer_hidden[i, :actual_frames, :].mean(dim=0)
                all_features[layer_idx, batch_start + i, :] = pooled.cpu().numpy()
                
                if layer_idx == 0:
                    frame_counts.append(actual_frames)
    
    elapsed = time.time() - t0
    print(f"  Extraction completed in {elapsed:.1f}s "
          f"({num_utterances / elapsed:.1f} utt/s)")
    
    # Save to HDF5
    print(f"  Saving to {output_path}...")
    with h5py.File(output_path, "w") as f:
        for layer_idx in range(num_layers):
            dataset_name = f"layer_{layer_idx:02d}"
            f.create_dataset(
                dataset_name,
                data=all_features[layer_idx],
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )
        
        # Store metadata as attributes
        f.attrs["model_name"] = extractor.name
        f.attrs["num_layers"] = num_layers
        f.attrs["hidden_dim"] = hidden_dim
        f.attrs["num_utterances"] = num_utterances
    
    # Save metadata separately (utterance IDs, frame counts)
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "model_name": extractor.name,
        "num_utterances": num_utterances,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "utterance_ids": utterance_ids,
        "frame_counts": frame_counts,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Report file size
    file_size_mb = output_path.stat().st_size / 1e6
    print(f"  Saved: {output_path} ({file_size_mb:.1f} MB)")
    print(f"  Metadata: {metadata_path}")
    
    return str(output_path)


def load_cached_features(
    features_dir: str,
    layer: int,
    level: str = "utterance",
) -> np.ndarray:
    """
    Load cached features for a specific layer.
    
    Args:
        features_dir: Path to features directory (e.g., "features/wavlm_large")
        layer: Layer index (0 = input embedding, 1 = first transformer, etc.)
        level: "utterance" (mean-pooled) or "frame" (full sequences)
    
    Returns:
        numpy array of shape (num_utterances, hidden_dim) for utterance level
    """
    features_dir = Path(features_dir)
    
    if level == "utterance":
        h5_path = features_dir / "utterance_level.h5"
    else:
        h5_path = features_dir / "frame_level.h5"
    
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Cached features not found: {h5_path}\n"
            f"Run feature extraction first."
        )
    
    dataset_name = f"layer_{layer:02d}"
    
    with h5py.File(h5_path, "r") as f:
        if dataset_name not in f:
            available = list(f.keys())
            raise KeyError(
                f"Layer {layer} not found. Available: {available}"
            )
        features = f[dataset_name][:]  # Load into memory
    
    return features


def load_metadata(features_dir: str) -> dict:
    """Load metadata for cached features."""
    metadata_path = Path(features_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, "r") as f:
        return json.load(f)
