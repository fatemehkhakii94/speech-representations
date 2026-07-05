"""
Noise mixing pipeline for acoustic environment probing.

Creates synthetic noisy speech by mixing clean LibriSpeech utterances
with noise from MUSAN at controlled SNR levels.

Noise categories:
- noise: ambient/environmental noise (air conditioning, street, etc.)
- music: background music
- speech: babble/multi-speaker noise
- clean: no added noise (control condition)

This gives us perfect experimental control:
- Same speech content across noise conditions
- Known noise type and SNR
- Probe results cleanly measure noise encoding
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from src.data.librispeech import load_audio


# Noise categories we probe
NOISE_CATEGORIES = {
    "clean": 0,
    "noise": 1,     # Ambient/environmental
    "music": 2,     # Background music
    "speech": 3,    # Babble/multi-speaker
}

NOISE_NAMES = ["clean", "noise", "music", "speech"]


def load_noise_files(musan_dir: str) -> Dict[str, List[str]]:
    """
    Scan MUSAN directory and collect paths by category.
    
    Returns:
        Dict mapping category name to list of file paths
    """
    musan_dir = Path(musan_dir)
    noise_files = {}
    
    for category in ["noise", "music", "speech"]:
        category_dir = musan_dir / category
        if not category_dir.exists():
            print(f"  [WARNING] {category_dir} not found")
            noise_files[category] = []
            continue
        
        files = sorted(category_dir.rglob("*.wav"))
        noise_files[category] = [str(f) for f in files]
        print(f"  {category}: {len(files)} files")
    
    return noise_files


def mix_audio(
    clean_waveform: torch.Tensor,
    noise_waveform: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """
    Mix clean speech with noise at a target SNR.
    
    SNR (dB) = 10 * log10(power_signal / power_noise)
    
    Args:
        clean_waveform: (num_samples,) clean speech
        noise_waveform: (num_samples,) noise (same length as clean)
        snr_db: target SNR in dB
    
    Returns:
        Mixed waveform (num_samples,)
    """
    # Compute RMS
    clean_rms = torch.sqrt(torch.mean(clean_waveform ** 2) + 1e-8)
    noise_rms = torch.sqrt(torch.mean(noise_waveform ** 2) + 1e-8)
    
    # Target noise RMS for desired SNR
    # SNR = 20 * log10(clean_rms / target_noise_rms)
    # target_noise_rms = clean_rms / 10^(SNR/20)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    
    # Scale noise
    scaling_factor = target_noise_rms / noise_rms
    scaled_noise = noise_waveform * scaling_factor
    
    # Mix
    mixed = clean_waveform + scaled_noise
    
    return mixed


def prepare_noise_segment(
    noise_path: str,
    target_length: int,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Load a noise file and adjust it to match the target length.
    
    If noise is shorter than target: loop it.
    If noise is longer: take a random segment.
    """
    waveform, sr = torchaudio.load(noise_path)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Adjust length
    if waveform.shape[0] >= target_length:
        # Random crop
        start = np.random.randint(0, waveform.shape[0] - target_length + 1)
        waveform = waveform[start:start + target_length]
    else:
        # Loop
        repeats = (target_length // waveform.shape[0]) + 1
        waveform = waveform.repeat(repeats)[:target_length]
    
    return waveform


def create_noisy_dataset(
    utterances: List[Dict],
    noise_files: Dict[str, List[str]],
    snr_db: float = 10.0,
    max_utterances: int = 1500,
    seed: int = 42,
) -> List[Tuple[str, torch.Tensor, int]]:
    """
    Create a balanced noisy dataset.
    
    For each noise category (including clean), creates max_utterances samples.
    Total dataset size: max_utterances * 4 categories.
    
    Args:
        utterances: LibriSpeech utterances to use as clean source
        noise_files: Dict from load_noise_files
        snr_db: SNR for mixing (dB)
        max_utterances: Number of samples per category
        seed: Random seed
    
    Returns:
        List of (utterance_id, waveform, noise_label) tuples
    """
    rng = np.random.RandomState(seed)
    
    # Limit clean utterances
    if len(utterances) > max_utterances * 4:
        indices = rng.choice(len(utterances), max_utterances * 4, replace=False)
        utterances = [utterances[i] for i in indices]
    
    # Shuffle and divide into 4 groups (one per noise condition)
    rng.shuffle(utterances)
    per_category = min(max_utterances, len(utterances) // 4)
    
    dataset = []
    
    print(f"\n  Creating noisy dataset (SNR={snr_db}dB, {per_category} per category)...")
    
    # Category 0: Clean (no noise added)
    print(f"    Clean: {per_category} samples")
    for i in range(per_category):
        utt = utterances[i]
        waveform = load_audio(utt["path"])
        utt_id = f"{utt['utterance_id']}_clean"
        dataset.append((utt_id, waveform, NOISE_CATEGORIES["clean"]))
    
    # Categories 1-3: Different noise types
    offset = per_category
    for noise_cat in ["noise", "music", "speech"]:
        if not noise_files.get(noise_cat):
            print(f"    {noise_cat}: SKIPPED (no files)")
            continue
        
        print(f"    {noise_cat}: {per_category} samples")
        available_noise = noise_files[noise_cat]
        
        for i in range(per_category):
            utt = utterances[offset + i]
            clean_waveform = load_audio(utt["path"])
            
            # Select random noise file
            noise_path = available_noise[rng.randint(len(available_noise))]
            noise_waveform = prepare_noise_segment(
                noise_path, len(clean_waveform)
            )
            
            # Mix
            mixed = mix_audio(clean_waveform, noise_waveform, snr_db)
            
            utt_id = f"{utt['utterance_id']}_{noise_cat}"
            dataset.append((utt_id, mixed, NOISE_CATEGORIES[noise_cat]))
        
        offset += per_category
    
    print(f"  Total noisy dataset: {len(dataset)} samples")
    print(f"  Categories: {NOISE_NAMES}")
    
    return dataset


def build_noise_probing_dataset(
    features_dir: str,
    dataset_items: List[Tuple[str, torch.Tensor, int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Build probing dataset for noise type classification.
    
    Simple random split (no speaker-based splitting needed since
    we're probing noise, not speaker characteristics).
    """
    from src.data.probing_dataset import ProbingDataset
    from src.models.caching import load_metadata
    
    rng = np.random.RandomState(seed)
    
    # Load metadata
    metadata = load_metadata(features_dir)
    cached_set = set(metadata["utterance_ids"])
    
    # Filter to cached
    utt_ids = []
    labels = []
    for utt_id, _, label in dataset_items:
        if utt_id in cached_set:
            utt_ids.append(utt_id)
            labels.append(label)
    
    labels = np.array(labels)
    n = len(utt_ids)
    
    # Random split
    perm = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = np.full(n, 2)  # default test
    splits[perm[:n_train]] = 0
    splits[perm[n_train:n_train + n_val]] = 1
    
    num_classes = len(NOISE_CATEGORIES)
    
    print(f"\nNoise probing dataset:")
    print(f"  Classes: {num_classes} ({NOISE_NAMES})")
    print(f"  Train: {(splits==0).sum()}, Val: {(splits==1).sum()}, Test: {(splits==2).sum()}")
    
    return ProbingDataset(
        features_dir=features_dir,
        utterance_ids=utt_ids,
        labels=labels,
        train_mask=(splits == 0),
        val_mask=(splits == 1),
        test_mask=(splits == 2),
        property_name="noise",
        num_classes=num_classes,
    )
