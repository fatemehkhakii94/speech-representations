"""
CREMA-D dataset handling.

CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset
- 91 actors (48 male, 43 female)
- 7,442 utterances
- 6 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad
- Same sentences spoken with different emotions (controlled lexical content)

File naming: {ActorID}_{SentenceID}_{Emotion}_{EmotionLevel}.wav
Example: 1001_DFA_ANG_XX.wav

Download from: https://github.com/CheyneyComputerScience/CREMA-D
Audio files: https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/

For practical purposes, we download from a direct source.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


# Emotion mapping from filename codes to labels
EMOTION_MAP = {
    "ANG": 0,  # Angry
    "DIS": 1,  # Disgust
    "FEA": 2,  # Fear
    "HAP": 3,  # Happy
    "NEU": 4,  # Neutral
    "SAD": 5,  # Sad
}

EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


def download_cremad(data_dir: str = "data/raw/cremad") -> Path:
    """
    Download CREMA-D dataset.
    
    CREMA-D audio is hosted on GitHub via git-lfs.
    We clone only the AudioWAV directory.
    """
    data_dir = Path(data_dir)
    audio_dir = data_dir / "AudioWAV"
    
    if audio_dir.exists() and len(list(audio_dir.glob("*.wav"))) > 7000:
        num_files = len(list(audio_dir.glob("*.wav")))
        print(f"[OK] CREMA-D already exists at {audio_dir} ({num_files} files)")
        return audio_dir
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading CREMA-D dataset...")
    print("(This requires git-lfs. If it fails, install with: sudo apt install git-lfs)")
    
    # Clone the repository (sparse checkout for just audio)
    repo_dir = data_dir / "CREMA-D"
    
    if not repo_dir.exists():
        subprocess.run([
            "git", "clone",
            "--depth", "1",
            "https://github.com/CheyneyComputerScience/CREMA-D.git",
            str(repo_dir),
        ], check=True)
    
    # The audio files should be in CREMA-D/AudioWAV/
    cloned_audio = repo_dir / "AudioWAV"
    
    if cloned_audio.exists():
        # Symlink or move
        if not audio_dir.exists():
            audio_dir.symlink_to(cloned_audio.resolve())
        return audio_dir
    else:
        print("[WARNING] AudioWAV not found after clone. Files may need git-lfs pull.")
        print(f"  Try: cd {repo_dir} && git lfs pull")
        return audio_dir


def scan_cremad(
    audio_dir: str,
    max_duration: float = 10.0,
    min_duration: float = 1.0,
) -> List[Dict]:
    """
    Scan CREMA-D directory and collect metadata.
    
    Parses filenames to extract actor ID, sentence, emotion, level.
    
    Returns:
        List of dicts with: utterance_id, speaker_id, emotion, emotion_label, path, duration
    """
    audio_dir = Path(audio_dir)
    utterances = []
    skipped = 0
    parse_errors = 0
    
    wav_files = sorted(audio_dir.glob("*.wav"))
    
    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found in {audio_dir}. "
            f"Check download or run: cd data/raw/cremad/CREMA-D && git lfs pull"
        )
    
    for wav_file in tqdm(wav_files, desc="Scanning CREMA-D"):
        # Parse filename: {ActorID}_{SentenceID}_{Emotion}_{Level}.wav
        # Example: 1001_DFA_ANG_XX.wav
        parts = wav_file.stem.split("_")
        
        if len(parts) < 4:
            parse_errors += 1
            continue
        
        actor_id = parts[0]
        sentence_id = parts[1]
        emotion_code = parts[2]
        
        # Skip if emotion not in our set
        if emotion_code not in EMOTION_MAP:
            parse_errors += 1
            continue
        
        # Get duration
        try:
            info = torchaudio.info(str(wav_file))
            duration = info.num_frames / info.sample_rate
        except Exception:
            skipped += 1
            continue
        
        if duration < min_duration or duration > max_duration:
            skipped += 1
            continue
        
        utterances.append({
            "utterance_id": wav_file.stem,
            "speaker_id": int(actor_id),
            "emotion_code": emotion_code,
            "emotion_label": EMOTION_MAP[emotion_code],
            "sentence_id": sentence_id,
            "path": str(wav_file),
            "duration": duration,
        })
    
    print(f"\nCREMA-D scan complete:")
    print(f"  Valid utterances: {len(utterances)}")
    print(f"  Skipped (duration): {skipped}")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Unique speakers: {len(set(u['speaker_id'] for u in utterances))}")
    print(f"  Emotions:")
    for emo_code, emo_idx in EMOTION_MAP.items():
        count = sum(1 for u in utterances if u["emotion_code"] == emo_code)
        print(f"    {EMOTION_NAMES[emo_idx]:10s} ({emo_code}): {count}")
    
    return utterances


def build_emotion_probing_dataset(
    features_dir: str,
    utterances: List[Dict],
    num_folds: int = 5,
    test_fold: int = 0,
    seed: int = 42,
):
    """
    Build probing dataset for emotion classification.
    
    Uses speaker-independent splitting: speakers in test never appear in train.
    This prevents the probe from learning speaker→emotion shortcuts.
    
    Args:
        features_dir: Path to cached features
        utterances: List from scan_cremad
        num_folds: Number of CV folds for speaker-independent split
        test_fold: Which fold to use as test (0-indexed)
        seed: Random seed
    """
    from src.data.probing_dataset import ProbingDataset
    from src.models.caching import load_metadata
    
    rng = np.random.RandomState(seed)
    
    # Load metadata to know which utterances are cached
    metadata = load_metadata(features_dir)
    cached_set = set(metadata["utterance_ids"])
    
    # Filter to cached utterances
    utterances = [u for u in utterances if u["utterance_id"] in cached_set]
    
    # Get unique speakers and assign to folds
    speakers = sorted(set(u["speaker_id"] for u in utterances))
    rng.shuffle(speakers)
    
    # Assign speakers to folds
    speaker_to_fold = {}
    for i, spk in enumerate(speakers):
        speaker_to_fold[spk] = i % num_folds
    
    # Build arrays
    utt_ids = [u["utterance_id"] for u in utterances]
    labels = np.array([u["emotion_label"] for u in utterances])
    
    # Split based on speaker folds
    # test_fold → test, (test_fold + 1) % num_folds → val, rest → train
    val_fold = (test_fold + 1) % num_folds
    
    splits = np.zeros(len(utterances), dtype=int)  # 0=train
    for i, utt in enumerate(utterances):
        fold = speaker_to_fold[utt["speaker_id"]]
        if fold == test_fold:
            splits[i] = 2  # test
        elif fold == val_fold:
            splits[i] = 1  # val
    
    num_classes = len(EMOTION_MAP)
    
    print(f"\nEmotion probing dataset:")
    print(f"  Classes: {num_classes} ({EMOTION_NAMES})")
    print(f"  Split: speaker-independent (fold {test_fold}/{num_folds})")
    print(f"  Train speakers: {sum(1 for s in speakers if speaker_to_fold[s] not in [test_fold, val_fold])}")
    print(f"  Val speakers: {sum(1 for s in speakers if speaker_to_fold[s] == val_fold)}")
    print(f"  Test speakers: {sum(1 for s in speakers if speaker_to_fold[s] == test_fold)}")
    
    return ProbingDataset(
        features_dir=features_dir,
        utterance_ids=utt_ids,
        labels=labels,
        train_mask=(splits == 0),
        val_mask=(splits == 1),
        test_mask=(splits == 2),
        property_name="emotion",
        num_classes=num_classes,
    )
