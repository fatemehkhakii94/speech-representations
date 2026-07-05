"""
Test feature caching pipeline.

Creates dummy audio, extracts features, saves to HDF5, and verifies
we can reload them correctly. Uses a small number of samples to be fast.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from src.models.feature_extractor import FeatureExtractor
from src.models.caching import (
    extract_and_cache_utterance_level,
    load_cached_features,
    load_metadata,
)


def test_caching_pipeline():
    """End-to-end test: extract → save → reload → verify."""
    
    print("=" * 60)
    print("CACHING PIPELINE TEST")
    print("=" * 60)
    
    # --- Setup ---
    print("\n1. Loading model...")
    extractor = FeatureExtractor.from_name("wavlm_large", device="cuda")
    
    # --- Create dummy audio samples (varying lengths) ---
    print("\n2. Creating dummy audio (10 samples, 2-5 seconds)...")
    audio_list = []
    for i in range(10):
        duration = 2.0 + i * 0.3  # 2.0s to 4.7s
        num_samples = int(16000 * duration)
        waveform = torch.randn(num_samples)
        audio_list.append((f"utt_{i:04d}", waveform))
        print(f"   utt_{i:04d}: {duration:.1f}s ({num_samples} samples)")
    
    # --- Extract and cache ---
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "wavlm_large"
        
        print(f"\n3. Extracting and caching to: {output_dir}")
        h5_path = extract_and_cache_utterance_level(
            extractor=extractor,
            audio_list=audio_list,
            output_dir=str(output_dir),
            batch_size=4,
        )
        
        # --- Verify saved files exist ---
        print("\n4. Verifying saved files...")
        assert Path(h5_path).exists(), "HDF5 file not created!"
        print(f"   [OK] HDF5 file exists: {h5_path}")
        
        metadata_path = output_dir / "metadata.json"
        assert metadata_path.exists(), "Metadata file not created!"
        print(f"   [OK] Metadata file exists: {metadata_path}")
        
        # --- Load and verify metadata ---
        print("\n5. Verifying metadata...")
        metadata = load_metadata(str(output_dir))
        assert metadata["model_name"] == "wavlm_large"
        assert metadata["num_utterances"] == 10
        assert metadata["num_layers"] == 25
        assert metadata["hidden_dim"] == 1024
        assert len(metadata["utterance_ids"]) == 10
        assert len(metadata["frame_counts"]) == 10
        print(f"   [OK] Model: {metadata['model_name']}")
        print(f"   [OK] Utterances: {metadata['num_utterances']}")
        print(f"   [OK] Layers: {metadata['num_layers']}")
        print(f"   [OK] Frame counts: {metadata['frame_counts']}")
        
        # --- Load features and verify shapes ---
        print("\n6. Verifying loaded features...")
        for layer_idx in [0, 12, 24]:
            features = load_cached_features(str(output_dir), layer=layer_idx)
            assert features.shape == (10, 1024), (
                f"Layer {layer_idx}: expected (10, 1024), got {features.shape}"
            )
            assert np.isfinite(features).all(), f"Layer {layer_idx}: non-finite values!"
            # Check features are not all zeros (actual extraction happened)
            assert np.abs(features).max() > 0.01, (
                f"Layer {layer_idx}: features are near-zero — extraction may have failed"
            )
            print(f"   [OK] Layer {layer_idx:2d}: shape={features.shape}, "
                  f"mean={features.mean():.4f}, std={features.std():.4f}")
        
        # --- Verify different layers give different features ---
        print("\n7. Verifying layer diversity...")
        feat_0 = load_cached_features(str(output_dir), layer=0)
        feat_12 = load_cached_features(str(output_dir), layer=12)
        feat_24 = load_cached_features(str(output_dir), layer=24)
        
        # Cosine similarity between layers (should not be 1.0)
        def cos_sim(a, b):
            a_flat = a.flatten()
            b_flat = b.flatten()
            return np.dot(a_flat, b_flat) / (
                np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8
            )
        
        sim_0_12 = cos_sim(feat_0, feat_12)
        sim_0_24 = cos_sim(feat_0, feat_24)
        sim_12_24 = cos_sim(feat_12, feat_24)
        
        print(f"   Cosine sim (layer 0 vs 12): {sim_0_12:.4f}")
        print(f"   Cosine sim (layer 0 vs 24): {sim_0_24:.4f}")
        print(f"   Cosine sim (layer 12 vs 24): {sim_12_24:.4f}")
        
        assert abs(sim_0_24) < 0.95, "Layer 0 and 24 too similar!"
        print(f"   [OK] Layers produce distinct cached features")
    
    # --- Cleanup ---
    del extractor
    torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("[PASSED] Caching pipeline works correctly")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_caching_pipeline()
