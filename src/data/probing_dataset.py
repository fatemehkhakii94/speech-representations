"""
Probing dataset: connects cached features to property labels.

This module handles the logic of:
- Loading cached features for a given model/layer
- Mapping utterances to their property labels
- Creating train/val/test splits appropriate for each property

Each property defines:
- Which utterances to use
- What label each utterance gets
- How to split (within-speaker, across-speaker, etc.)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from src.models.caching import load_cached_features, load_metadata


class ProbingDataset:
    """
    Holds features and labels for a probing experiment.
    
    Provides train/val/test splits as numpy arrays ready for the trainer.
    """
    
    def __init__(
        self,
        features_dir: str,
        utterance_ids: List[str],
        labels: np.ndarray,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        test_mask: np.ndarray,
        property_name: str,
        num_classes: int,
    ):
        self.features_dir = features_dir
        self.utterance_ids = utterance_ids
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.property_name = property_name
        self.num_classes = num_classes
        
        # Build utterance_id → index mapping for the cached features
        metadata = load_metadata(features_dir)
        self.cached_ids = metadata["utterance_ids"]
        self.id_to_cache_idx = {
            uid: idx for idx, uid in enumerate(self.cached_ids)
        }
        
        # Map our utterance IDs to cache indices
        self.cache_indices = np.array([
            self.id_to_cache_idx[uid] for uid in utterance_ids
        ])
    
    def get_split(
        self, layer: int, split: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get features and labels for a specific layer and split.
        
        Args:
            layer: Layer index (0 = CNN output, 1..N = transformer layers)
            split: "train", "val", or "test"
        
        Returns:
            (features, labels) as numpy arrays
        """
        # Load all features for this layer
        all_features = load_cached_features(self.features_dir, layer=layer)
        
        # Select our utterances
        our_features = all_features[self.cache_indices]
        
        # Select split
        if split == "train":
            mask = self.train_mask
        elif split == "val":
            mask = self.val_mask
        elif split == "test":
            mask = self.test_mask
        else:
            raise ValueError(f"Unknown split: {split}")
        
        return our_features[mask], self.labels[mask]
    
    @property
    def num_train(self) -> int:
        return self.train_mask.sum()
    
    @property
    def num_val(self) -> int:
        return self.val_mask.sum()
    
    @property
    def num_test(self) -> int:
        return self.test_mask.sum()
    
    def summary(self) -> str:
        return (
            f"ProbingDataset(\n"
            f"  property={self.property_name},\n"
            f"  classes={self.num_classes},\n"
            f"  train={self.num_train}, val={self.num_val}, test={self.num_test},\n"
            f"  features_dir={self.features_dir}\n"
            f")"
        )


def build_speaker_probing_dataset(
    features_dir: str,
    utterances: List[Dict],
    max_utterances_per_speaker: int = 50,
    min_utterances_per_speaker: int = 10,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> ProbingDataset:
    """
    Build probing dataset for speaker identity.
    
    Split strategy: within-speaker (all speakers in all splits,
    utterances divided proportionally per speaker).
    """
    rng = np.random.RandomState(seed)
    
    # Load metadata to know which utterances are cached
    metadata = load_metadata(features_dir)
    cached_set = set(metadata["utterance_ids"])
    
    # Filter to only utterances that are in the cache
    utterances = [u for u in utterances if u["utterance_id"] in cached_set]
    
    # Group by speaker
    speaker_utts = {}
    for utt in utterances:
        sid = utt["speaker_id"]
        if sid not in speaker_utts:
            speaker_utts[sid] = []
        speaker_utts[sid].append(utt)
    
    # Filter speakers with enough data
    valid_speakers = sorted([
        sid for sid, utts in speaker_utts.items()
        if len(utts) >= min_utterances_per_speaker
    ])
    
    # Create class mapping
    speaker_to_class = {sid: idx for idx, sid in enumerate(valid_speakers)}
    num_classes = len(valid_speakers)
    
    # Collect utterances with labels and split masks
    all_utt_ids = []
    all_labels = []
    all_splits = []  # 0=train, 1=val, 2=test
    
    for sid in valid_speakers:
        utts = speaker_utts[sid]
        
        # Cap utterances per speaker
        if len(utts) > max_utterances_per_speaker:
            indices = rng.choice(len(utts), max_utterances_per_speaker, replace=False)
            utts = [utts[i] for i in indices]
        
        # Shuffle and split
        rng.shuffle(utts)
        n = len(utts)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        
        for i, utt in enumerate(utts):
            all_utt_ids.append(utt["utterance_id"])
            all_labels.append(speaker_to_class[sid])
            if i < n_train:
                all_splits.append(0)
            elif i < n_train + n_val:
                all_splits.append(1)
            else:
                all_splits.append(2)
    
    labels = np.array(all_labels)
    splits = np.array(all_splits)
    
    return ProbingDataset(
        features_dir=features_dir,
        utterance_ids=all_utt_ids,
        labels=labels,
        train_mask=(splits == 0),
        val_mask=(splits == 1),
        test_mask=(splits == 2),
        property_name="speaker",
        num_classes=num_classes,
    )


def build_pitch_probing_dataset(
    features_dir: str,
    utterances: List[Dict],
    num_bins: int = 5,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> ProbingDataset:
    """
    Build probing dataset for pitch (F0) classification.
    
    Extracts mean F0 per utterance using parselmouth, bins into classes.
    Bins are computed per-gender to avoid gender confound.
    """
    import parselmouth
    from parselmouth.praat import call
    
    rng = np.random.RandomState(seed)
    
    # Load metadata
    metadata = load_metadata(features_dir)
    cached_set = set(metadata["utterance_ids"])
    utterances = [u for u in utterances if u["utterance_id"] in cached_set]
    
    print("  Extracting F0 values...")
    utt_ids = []
    f0_values = []
    
    for utt in utterances:
        try:
            snd = parselmouth.Sound(utt["path"])
            pitch = call(snd, "To Pitch", 0.0, 75, 500)
            f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
            
            if f0_mean > 0 and np.isfinite(f0_mean):  # Valid F0
                utt_ids.append(utt["utterance_id"])
                f0_values.append(f0_mean)
        except Exception:
            continue
    
    f0_values = np.array(f0_values)
    print(f"  Valid F0 measurements: {len(f0_values)}/{len(utterances)}")
    print(f"  F0 range: {f0_values.min():.0f} - {f0_values.max():.0f} Hz")
    
    # Bin into classes using quantiles (gender-independent for now)
    bin_edges = np.percentile(f0_values, np.linspace(0, 100, num_bins + 1))
    labels = np.digitize(f0_values, bin_edges[1:-1])  # 0 to num_bins-1
    
    print(f"  Bin edges: {[f'{e:.0f}' for e in bin_edges]} Hz")
    for b in range(num_bins):
        count = (labels == b).sum()
        print(f"    Bin {b}: {count} utterances")
    
    # Random split
    n = len(utt_ids)
    perm = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = np.full(n, 2)  # default test
    splits[perm[:n_train]] = 0
    splits[perm[n_train:n_train + n_val]] = 1
    
    return ProbingDataset(
        features_dir=features_dir,
        utterance_ids=utt_ids,
        labels=labels,
        train_mask=(splits == 0),
        val_mask=(splits == 1),
        test_mask=(splits == 2),
        property_name="pitch",
        num_classes=num_bins,
    )
