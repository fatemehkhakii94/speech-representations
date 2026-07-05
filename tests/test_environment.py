"""
Environment sanity checks.
Run this FIRST before any other code.
Verifies: Python, PyTorch, CUDA, GPU, and key dependencies.
"""

import sys
import os


def test_python_version():
    """Require Python 3.9+"""
    assert sys.version_info >= (3, 9), (
        f"Python 3.9+ required, got {sys.version}"
    )
    print(f"[OK] Python version: {sys.version}")


def test_pytorch():
    """Verify PyTorch is installed and reports version."""
    import torch
    print(f"[OK] PyTorch version: {torch.__version__}")


def test_cuda_available():
    """Verify CUDA is accessible from PyTorch."""
    import torch
    assert torch.cuda.is_available(), (
        "CUDA not available! Check PyTorch installation."
    )
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
    print(f"[OK] CUDA version (PyTorch built with): {torch.version.cuda}")


def test_gpu_detection():
    """Verify GPU is detected and report its properties."""
    import torch
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPU detected!"

    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"[OK] GPU {i}: {name} ({mem:.1f} GB)")


def test_gpu_computation():
    """Run a small computation on GPU to verify it works end-to-end."""
    import torch

    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")
    z = torch.mm(x, y)

    assert z.device.type == "cuda"
    assert z.shape == (1000, 1000)
    assert torch.isfinite(z).all()

    print(f"[OK] GPU computation successful (matmul 1000x1000)")


def test_key_dependencies():
    """Verify all critical packages are importable."""
    packages = {
        "transformers": "HuggingFace Transformers (model loading)",
        "torchaudio": "Audio processing",
        "numpy": "Numerical computing",
        "scipy": "Scientific computing",
        "pandas": "Data manipulation",
        "matplotlib": "Plotting",
        "seaborn": "Statistical visualization",
        "sklearn": "Metrics and utilities",
        "yaml": "Configuration parsing",
        "h5py": "Feature storage",
        "librosa": "Audio analysis",
        "soundfile": "Audio I/O",
        "parselmouth": "F0 extraction (Praat)",
        "tqdm": "Progress bars",
    }

    for package, purpose in packages.items():
        try:
            __import__(package)
            print(f"[OK] {package:15s} — {purpose}")
        except ImportError as e:
            print(f"[FAIL] {package:15s} — {purpose}")
            print(f"       Error: {e}")
            raise


def test_transformers_with_torch():
    """Verify transformers can use PyTorch for model loading."""
    import transformers
    import torch

    assert transformers.is_torch_available(), (
        "Transformers cannot find PyTorch! "
        f"torch version: {torch.__version__}, "
        f"transformers version: {transformers.__version__}"
    )
    print(f"[OK] Transformers {transformers.__version__} with PyTorch {torch.__version__}")

    from transformers import WavLMConfig, HubertConfig, WhisperConfig
    _ = WavLMConfig()
    _ = HubertConfig()
    _ = WhisperConfig()
    print("[OK] Model configs accessible (WavLM, HuBERT, Whisper)")


def test_disk_space():
    """Check available disk space for features and data."""
    import shutil

    # Use current working directory (project root)
    path = os.path.abspath(".")
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1 << 30)

    print(f"[OK] Disk space available: {free_gb:.1f} GB at {path}")

    if free_gb < 50:
        print("[WARNING] Less than 50GB free. Feature caching may be tight.")
    elif free_gb < 100:
        print("[NOTE] 50-100GB free. Sufficient with careful management.")
    else:
        print("[OK] Plenty of space for features and data.")


def test_gpu_memory():
    """Report GPU memory and verify enough for model inference."""
    import torch

    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[OK] GPU memory: {mem_total:.1f} GB")

    if mem_total >= 12:
        print("[OK] Sufficient for WavLM-Large / HuBERT-Large inference in fp16")
    elif mem_total >= 8:
        print("[WARNING] Tight. May need to reduce batch size during extraction.")
    else:
        print("[WARNING] Limited GPU memory. Consider CPU extraction.")


if __name__ == "__main__":
    print("=" * 60)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 60)
    print()

    tests = [
        test_python_version,
        test_pytorch,
        test_cuda_available,
        test_gpu_detection,
        test_gpu_computation,
        test_key_dependencies,
        test_transformers_with_torch,
        test_disk_space,
        test_gpu_memory,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        print(f"\n--- {test_fn.__name__} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAILED] {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n All checks passed. Environment is ready.")
    else:
        print("\n Fix the failures above before continuing.")
        sys.exit(1)
