"""
Download MUSAN noise corpus.

MUSAN contains:
- Music: ~42 hours
- Speech (babble): ~60 hours  
- Noise (ambient): ~6 hours

We use the 'noise' and 'music' subsets for creating noisy mixtures.

Source: https://www.openslr.org/17/
"""

import sys
import subprocess
import tarfile
from pathlib import Path

def download_musan(data_dir: str = "data/raw/musan") -> Path:
    """Download and extract MUSAN corpus."""
    data_dir = Path(data_dir)
    
    # Check if already exists
    noise_dir = data_dir / "musan" / "noise"
    music_dir = data_dir / "musan" / "music"
    
    if noise_dir.exists() and music_dir.exists():
        noise_count = len(list(noise_dir.rglob("*.wav")))
        music_count = len(list(music_dir.rglob("*.wav")))
        print(f"[OK] MUSAN already exists at {data_dir}/musan")
        print(f"     Noise files: {noise_count}")
        print(f"     Music files: {music_count}")
        return data_dir / "musan"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / "musan.tar.gz"
    
    # Download
    if not tar_path.exists():
        print("Downloading MUSAN (~1.4 GB)...")
        url = "https://www.openslr.org/resources/17/musan.tar.gz"
        subprocess.run(
            ["wget", "-O", str(tar_path), url],
            check=True,
        )
    
    # Extract
    print("Extracting MUSAN...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=str(data_dir))
    
    # Verify
    noise_count = len(list(noise_dir.rglob("*.wav")))
    music_count = len(list(music_dir.rglob("*.wav")))
    print(f"[OK] MUSAN extracted:")
    print(f"     Noise files: {noise_count}")
    print(f"     Music files: {music_count}")
    
    # Optionally remove tar
    # tar_path.unlink()
    
    return data_dir / "musan"


if __name__ == "__main__":
    download_musan()
