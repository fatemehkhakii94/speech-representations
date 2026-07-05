"""
LibriSpeech dataset handling.

Handles:
- Downloading train-clean-100 subset
- Loading audio files with metadata (speaker_id, chapter, utterance)
- Preparing audio for feature extraction
- Creating splits for probing experiments

Directory structure after download:
    data/raw/librispeech/
        LibriSpeech/
            train-clean-100/
                {speaker_id}/
                    {chapter_id}/
                        {speaker_id}-{chapter_id}-{utterance_id}.flac
                        {speaker_id}-{chapter_id}.trans.txt
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def download_librispeech(
    data_dir: str = "data/raw/librispeech",
    subset: str = "train-clean-100",
) -> Path:
    """
    Download LibriSpeech subset using torchaudio.
    
    Args:
        data_dir: Where to save the dataset
        subset: Which subset ("train-clean-100", "test-clean", etc.)
    
    Returns:
        Path to the downloaded dataset root
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    expected_path = data_dir / "LibriSpeech" / subset
    if expected_path.exists():
        num_speakers = len(list(expected_path.iterdir()))
        print(f"[OK] LibriSpeech {subset} already exists at {expected_path}")
        print(f"     Found {num_speakers} speakers")
        return expected_path
    
    print(f"Downloading LibriSpeech {subset} to {data_dir}...")
    print("(This may take 10-20 minutes depending on connection speed)")
    
    # torchaudio handles the download
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=str(data_dir),
        url=subset,
        download=True,
    )
    
    print(f"[OK] Download complete: {expected_path}")
    return expected_path


def scan_librispeech(
    dataset_path: str,
    max_duration: float = 10.0,
    min_duration: float = 1.0,
    sample_rate: int = 16000,
) -> List[Dict]:
    """
    Scan LibriSpeech directory and collect metadata for all utterances.
    
    Args:
        dataset_path: Path to LibriSpeech/{subset}/ directory
        max_duration: Maximum utterance duration in seconds
        min_duration: Minimum utterance duration in seconds
        sample_rate: Expected sample rate
    
    Returns:
        List of dicts with keys: utterance_id, speaker_id, path, num_samples, duration
    """
    dataset_path = Path(dataset_path)
    utterances = []
    skipped_short = 0
    skipped_long = 0
    
    # Walk the directory structure
    speaker_dirs = sorted(dataset_path.iterdir())
    
    for speaker_dir in tqdm(speaker_dirs, desc="Scanning speakers"):
        if not speaker_dir.is_dir():
            continue
        
        speaker_id = int(speaker_dir.name)
        
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue
            
            for audio_file in sorted(chapter_dir.glob("*.flac")):
                # Get audio info without loading the full file
                info = torchaudio.info(str(audio_file))
                num_samples = info.num_frames
                duration = num_samples / info.sample_rate
                
                if duration < min_duration:
                    skipped_short += 1
                    continue
                if duration > max_duration:
                    skipped_long += 1
                    continue
                
                utterance_id = audio_file.stem  # e.g., "103-1240-0000"
                
                utterances.append({
                    "utterance_id": utterance_id,
                    "speaker_id": speaker_id,
                    "path": str(audio_file),
                    "num_samples": num_samples,
                    "duration": duration,
                })
    
    print(f"\nScan complete:")
    print(f"  Valid utterances: {len(utterances)}")
    print(f"  Skipped (too short < {min_duration}s): {skipped_short}")
    print(f"  Skipped (too long > {max_duration}s): {skipped_long}")
    print(f"  Unique speakers: {len(set(u['speaker_id'] for u in utterances))}")
    
    return utterances


def load_audio(path: str, target_sample_rate: int = 16000) -> torch.Tensor:
    """
    Load a single audio file and resample if needed.
    
    Args:
        path: Path to audio file
        target_sample_rate: Desired sample rate
    
    Returns:
        Waveform tensor of shape (num_samples,)
    """
    waveform, sr = torchaudio.load(path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if necessary
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        waveform = resampler(waveform)
    
    # Remove channel dimension: (1, samples) → (samples,)
    waveform = waveform.squeeze(0)
    
    # Normalize to zero mean, unit variance (per-utterance)
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
    
    return waveform


def prepare_speaker_probing(
    utterances: List[Dict],
    max_utterances_per_speaker: int = 50,
    min_utterances_per_speaker: int = 10,
    seed: int = 42,
) -> Tuple[List[Dict], Dict[int, int]]:
    """
    Prepare data for speaker probing.
    
    Selects speakers with enough utterances and subsamples to balance.
    Creates a mapping from original speaker_id to probe class index (0..N-1).
    
    Args:
        utterances: Full list from scan_librispeech
        max_utterances_per_speaker: Cap per speaker (balance classes)
        min_utterances_per_speaker: Minimum to include a speaker
        seed: Random seed for subsampling
    
    Returns:
        (selected_utterances, speaker_to_class_mapping)
    """
    rng = np.random.RandomState(seed)
    
    # Group by speaker
    speaker_utts = {}
    for utt in utterances:
        sid = utt["speaker_id"]
        if sid not in speaker_utts:
            speaker_utts[sid] = []
        speaker_utts[sid].append(utt)
    
    # Filter speakers with enough utterances
    valid_speakers = {
        sid: utts for sid, utts in speaker_utts.items()
        if len(utts) >= min_utterances_per_speaker
    }
    
    # Create class mapping (sorted speaker IDs → 0..N-1)
    sorted_speakers = sorted(valid_speakers.keys())
    speaker_to_class = {sid: idx for idx, sid in enumerate(sorted_speakers)}
    
    # Subsample utterances per speaker
    selected = []
    for sid in sorted_speakers:
        utts = valid_speakers[sid]
        if len(utts) > max_utterances_per_speaker:
            indices = rng.choice(len(utts), max_utterances_per_speaker, replace=False)
            utts = [utts[i] for i in indices]
        selected.extend(utts)
    
    print(f"\nSpeaker probing data:")
    print(f"  Speakers (classes): {len(sorted_speakers)}")
    print(f"  Total utterances: {len(selected)}")
    print(f"  Utterances per speaker: {min_utterances_per_speaker}-{max_utterances_per_speaker}")
    
    return selected, speaker_to_class


def create_splits(
    utterances: List[Dict],
    speaker_to_class: Dict[int, int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Split utterances into train/val/test.
    
    For speaker probing: split within speakers (all speakers in all splits).
    Each speaker's utterances are divided proportionally.
    
    Returns:
        (train_list, val_list, test_list) where each item is (utterance_dict, class_label)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    rng = np.random.RandomState(seed)
    
    # Group by speaker
    speaker_utts = {}
    for utt in utterances:
        sid = utt["speaker_id"]
        if sid not in speaker_utts:
            speaker_utts[sid] = []
        speaker_utts[sid].append(utt)
    
    train_list, val_list, test_list = [], [], []
    
    for sid, utts in speaker_utts.items():
        class_label = speaker_to_class[sid]
        
        # Shuffle within speaker
        perm = rng.permutation(len(utts))
        utts = [utts[i] for i in perm]
        
        # Split
        n = len(utts)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # Rest goes to test
        
        train_utts = utts[:n_train]
        val_utts = utts[n_train:n_train + n_val]
        test_utts = utts[n_train + n_val:]
        
        for u in train_utts:
            train_list.append((u, class_label))
        for u in val_utts:
            val_list.append((u, class_label))
        for u in test_utts:
            test_list.append((u, class_label))
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_list)}")
    print(f"  Val:   {len(val_list)}")
    print(f"  Test:  {len(test_list)}")
    
    return train_list, val_list, test_list
