"""
Download all required datasets.

Usage:
    python scripts/download_data.py --dataset librispeech
    python scripts/download_data.py --dataset all
"""

import sys
import argparse

sys.path.insert(0, ".")
from src.data.librispeech import download_librispeech, scan_librispeech


def download_librispeech_data():
    """Download and verify LibriSpeech train-clean-100."""
    print("=" * 60)
    print("DOWNLOADING LIBRISPEECH train-clean-100")
    print("=" * 60)
    
    dataset_path = download_librispeech(
        data_dir="data/raw/librispeech",
        subset="train-clean-100",
    )
    
    # Scan to verify
    print("\nScanning dataset...")
    utterances = scan_librispeech(str(dataset_path))
    
    # Summary statistics
    speakers = set(u["speaker_id"] for u in utterances)
    durations = [u["duration"] for u in utterances]
    
    print(f"\nDataset summary:")
    print(f"  Total utterances: {len(utterances)}")
    print(f"  Total speakers:   {len(speakers)}")
    print(f"  Duration range:   {min(durations):.1f}s - {max(durations):.1f}s")
    print(f"  Mean duration:    {sum(durations)/len(durations):.1f}s")
    print(f"  Total audio:      {sum(durations)/3600:.1f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech",
        choices=["librispeech", "all"],
    )
    args = parser.parse_args()
    
    if args.dataset in ["librispeech", "all"]:
        download_librispeech_data()
    
    print("\n[DONE]")
