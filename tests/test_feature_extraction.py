"""
Test feature extraction from speech foundation models.

This test:
1. Loads a model (downloads weights on first run)
2. Creates dummy audio input
3. Runs forward pass
4. Verifies output shapes and properties

Run with: python tests/test_feature_extraction.py --model wavlm_large
"""

import sys
import argparse
import torch
import time

sys.path.insert(0, ".")
from src.models.feature_extractor import FeatureExtractor
from src.config import get_model_config


def test_extraction(model_name: str):
    """Test feature extraction for a single model."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    config = get_model_config(model_name)
    print(f"\nModel config:")
    print(f"  Hub ID: {config.hub_id}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  FP16: {config.use_fp16}")
    
    # --- Load model ---
    print(f"\nLoading model...")
    t0 = time.time()
    extractor = FeatureExtractor.from_name(model_name, device="cuda")
    t_load = time.time() - t0
    print(f"[OK] Model loaded in {t_load:.1f}s")
    print(f"     {extractor}")
    
    # --- Report GPU memory after loading ---
    mem_allocated = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    print(f"\n  GPU memory allocated: {mem_allocated:.2f} GB")
    print(f"  GPU memory reserved:  {mem_reserved:.2f} GB")
    
    # --- Create dummy audio (3 seconds of white noise) ---
    duration_sec = 3.0
    num_samples = int(16000 * duration_sec)
    batch_size = 2
    
    dummy_audio = torch.randn(batch_size, num_samples)
    print(f"\nInput: batch_size={batch_size}, duration={duration_sec}s, "
          f"samples={num_samples}")
    
    # --- Run extraction ---
    print(f"\nRunning forward pass...")
    t0 = time.time()
    hidden_states = extractor(dummy_audio, sample_rate=16000)
    t_forward = time.time() - t0
    print(f"[OK] Forward pass completed in {t_forward:.2f}s")
    
    # --- Verify outputs ---
    print(f"\nVerifying outputs:")
    
    # Check number of layers
    expected_num_outputs = config.num_layers + 1  # +1 for input embedding
    assert len(hidden_states) == expected_num_outputs, (
        f"Expected {expected_num_outputs} hidden states, got {len(hidden_states)}"
    )
    print(f"  [OK] Number of hidden states: {len(hidden_states)} "
          f"(= {config.num_layers} layers + 1 input embedding)")
    
    # Check shape of each layer's output
    for i, h in enumerate(hidden_states):
        assert h.dim() == 3, f"Layer {i}: expected 3D tensor, got {h.dim()}D"
        assert h.shape[0] == batch_size, (
            f"Layer {i}: batch dim is {h.shape[0]}, expected {batch_size}"
        )
        assert h.shape[2] == config.hidden_dim, (
            f"Layer {i}: hidden dim is {h.shape[2]}, expected {config.hidden_dim}"
        )
        # Check all values are finite
        assert torch.isfinite(h).all(), f"Layer {i}: contains non-finite values!"
    
    num_frames = hidden_states[0].shape[1]
    print(f"  [OK] Shape per layer: ({batch_size}, {num_frames}, {config.hidden_dim})")
    print(f"  [OK] Frame rate: ~{num_frames/duration_sec:.0f} frames/sec "
          f"(~{1000*duration_sec/num_frames:.1f} ms/frame)")
    print(f"  [OK] All values finite")
    
    # Check that layers produce DIFFERENT representations
    # (if all layers are identical, something is wrong)
    layer_0 = hidden_states[0]
    layer_mid = hidden_states[config.num_layers // 2]
    layer_last = hidden_states[-1]
    
    cos_sim_0_mid = torch.nn.functional.cosine_similarity(
        layer_0.reshape(-1, config.hidden_dim),
        layer_mid.reshape(-1, config.hidden_dim),
        dim=-1,
    ).mean().item()
    
    cos_sim_0_last = torch.nn.functional.cosine_similarity(
        layer_0.reshape(-1, config.hidden_dim),
        layer_last.reshape(-1, config.hidden_dim),
        dim=-1,
    ).mean().item()
    
    print(f"  [OK] Cosine similarity (layer 0 vs mid): {cos_sim_0_mid:.3f}")
    print(f"  [OK] Cosine similarity (layer 0 vs last): {cos_sim_0_last:.3f}")
    assert cos_sim_0_last < 0.99, "First and last layer are nearly identical — something is wrong"
    print(f"  [OK] Layers produce distinct representations")
    
    # --- Memory cleanup ---
    del hidden_states, dummy_audio, extractor
    torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"[PASSED] {model_name} feature extraction works correctly")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="wavlm_large",
        choices=["wavlm_large", "hubert_large", "whisper_large_v3", "all"],
        help="Which model to test",
    )
    args = parser.parse_args()
    
    if args.model == "all":
        for name in ["wavlm_large", "hubert_large", "whisper_large_v3"]:
            test_extraction(name)
            # Clear GPU between models
            torch.cuda.empty_cache()
    else:
        test_extraction(args.model)
