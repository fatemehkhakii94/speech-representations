"""Test that configuration system works correctly."""

import sys
sys.path.insert(0, ".")

from src.config import load_yaml, get_model_config, MODELS


def test_model_configs():
    """Verify predefined model configs."""
    for name, config in MODELS.items():
        assert config.name == name
        assert config.num_layers > 0
        assert config.hidden_dim > 0
        assert config.hub_id != ""
        print(f"[OK] {name}: {config.num_layers} layers, dim={config.hidden_dim}")


def test_load_extraction_configs():
    """Verify extraction YAML files parse correctly."""
    for model_name in ["wavlm_large", "hubert_large", "whisper_large_v3"]:
        config = load_yaml(f"configs/extraction/{model_name}.yaml")
        assert "model" in config
        assert "data" in config
        assert config["model"]["name"] == model_name
        print(f"[OK] configs/extraction/{model_name}.yaml loaded")


def test_load_probing_configs():
    """Verify probing YAML files parse correctly."""
    for task in ["speaker", "phoneme", "emotion"]:
        config = load_yaml(f"configs/probing/{task}.yaml")
        assert "property_name" in config
        assert "num_classes" in config
        assert config["property_name"] == task
        print(f"[OK] configs/probing/{task}.yaml loaded")


def test_get_model_config():
    """Verify model config lookup."""
    config = get_model_config("wavlm_large")
    assert config.hub_id == "microsoft/wavlm-large"
    assert config.num_layers == 24
    assert config.hidden_dim == 1024
    print("[OK] get_model_config('wavlm_large') works")

    # Test invalid model name raises error
    try:
        get_model_config("nonexistent_model")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("[OK] Invalid model name raises ValueError")


if __name__ == "__main__":
    print("=" * 60)
    print("CONFIGURATION SYSTEM TESTS")
    print("=" * 60)

    tests = [
        test_model_configs,
        test_load_extraction_configs,
        test_load_probing_configs,
        test_get_model_config,
    ]

    for test_fn in tests:
        print(f"\n--- {test_fn.__name__} ---")
        test_fn()

    print("\n" + "=" * 60)
    print("All config tests passed.")
    print("=" * 60)
