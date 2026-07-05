"""
Test the complete probing pipeline: features → probe → train → evaluate.

Uses synthetic data to verify:
1. Probes have correct shapes and can forward pass
2. Training loop converges on easy data
3. Training loop doesn't converge on random data (sanity check)
4. Metrics are computed correctly
"""

import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from src.models.probes import create_probe, LinearProbe, MLPProbe
from src.training.trainer import ProbeTrainer
from src.evaluation.metrics import compute_metrics


def test_probe_architectures():
    """Verify probe shapes and forward pass."""
    print("\n--- test_probe_architectures ---")
    
    input_dim = 1024
    num_classes = 39
    batch_size = 32
    
    # Linear probe
    linear = LinearProbe(input_dim, num_classes)
    x = torch.randn(batch_size, input_dim)
    out = linear(x)
    assert out.shape == (batch_size, num_classes)
    print(f"[OK] LinearProbe: {linear}")
    
    # MLP probe
    mlp = MLPProbe(input_dim, num_classes, hidden_dim=256, dropout=0.1)
    out = mlp(x)
    assert out.shape == (batch_size, num_classes)
    print(f"[OK] MLPProbe: {mlp}")
    
    # Factory function
    probe = create_probe("linear", input_dim, num_classes)
    assert isinstance(probe, LinearProbe)
    probe = create_probe("mlp", input_dim, num_classes)
    assert isinstance(probe, MLPProbe)
    print("[OK] create_probe factory works")


def test_training_converges_on_easy_data():
    """
    Create linearly separable data and verify the probe can learn it.
    
    If a linear probe can't learn linearly separable data, our trainer is broken.
    """
    print("\n--- test_training_converges_on_easy_data ---")
    
    np.random.seed(42)
    
    # Create easy data: 4 classes, well-separated clusters
    input_dim = 64
    num_classes = 4
    samples_per_class = 200
    
    features = []
    labels = []
    for c in range(num_classes):
        # Each class is a Gaussian cluster centered at a different point
        center = np.zeros(input_dim)
        center[c * (input_dim // num_classes)] = 5.0  # Large separation
        class_features = np.random.randn(samples_per_class, input_dim) * 0.5 + center
        features.append(class_features)
        labels.append(np.full(samples_per_class, c))
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Shuffle
    perm = np.random.permutation(len(features))
    features = features[perm]
    labels = labels[perm]
    
    # Split: 60% train, 20% val, 20% test
    n = len(features)
    train_features = features[:int(0.6*n)]
    train_labels = labels[:int(0.6*n)]
    val_features = features[int(0.6*n):int(0.8*n)]
    val_labels = labels[int(0.6*n):int(0.8*n)]
    test_features = features[int(0.8*n):]
    test_labels = labels[int(0.8*n):]
    
    # Train linear probe
    probe = create_probe("linear", input_dim, num_classes)
    trainer = ProbeTrainer(
        probe=probe,
        device="cuda",
        learning_rate=1e-2,
        batch_size=64,
        max_epochs=50,
        patience=10,
    )
    
    result = trainer.train(
        train_features, train_labels,
        val_features, val_labels,
        test_features, test_labels,
    )
    
    print(f"  {result}")
    
    # On easy data, we expect near-perfect accuracy
    assert result.test_accuracy > 0.90, (
        f"Linear probe should easily solve this. Got {result.test_accuracy:.3f}"
    )
    print(f"[OK] Linear probe converges on easy data: test_acc={result.test_accuracy:.4f}")


def test_training_fails_on_random_labels():
    """
    Train a probe on random labels (no real signal).
    
    The probe should NOT achieve high accuracy. If it does, the trainer
    is somehow leaking information or the data splits are wrong.
    This is our control task sanity check.
    """
    print("\n--- test_training_fails_on_random_labels ---")
    
    np.random.seed(42)
    
    input_dim = 64
    num_classes = 10
    num_samples = 1000
    
    # Random features, random labels — no learnable relationship
    features = np.random.randn(num_samples, input_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=num_samples)
    
    # Split
    train_features = features[:600]
    train_labels = labels[:600]
    val_features = features[600:800]
    val_labels = labels[600:800]
    test_features = features[800:]
    test_labels = labels[800:]
    
    # Train
    probe = create_probe("linear", input_dim, num_classes)
    trainer = ProbeTrainer(
        probe=probe,
        device="cuda",
        learning_rate=1e-3,
        batch_size=64,
        max_epochs=30,
        patience=5,
    )
    
    result = trainer.train(
        train_features, train_labels,
        val_features, val_labels,
        test_features, test_labels,
    )
    
    print(f"  {result}")
    
    # With 10 random classes, chance is 10%. Probe shouldn't get much above that.
    expected_chance = 1.0 / num_classes
    assert result.test_accuracy < expected_chance + 0.10, (
        f"Probe is too good on random labels! Got {result.test_accuracy:.3f}, "
        f"chance is {expected_chance:.3f}. Possible data leak."
    )
    print(f"[OK] Probe fails on random labels: test_acc={result.test_accuracy:.4f} "
          f"(chance={expected_chance:.2f})")


def test_metrics_computation():
    """Verify metric computation."""
    print("\n--- test_metrics_computation ---")
    
    # Perfect predictions
    labels = np.array([0, 0, 1, 1, 2, 2])
    predictions = np.array([0, 0, 1, 1, 2, 2])
    
    metrics = compute_metrics(predictions, labels, num_classes=3)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["selectivity"] > 0.0
    print(f"[OK] Perfect predictions: accuracy={metrics['accuracy']:.2f}")
    
    # Random predictions
    np.random.seed(0)
    labels = np.random.randint(0, 5, size=1000)
    predictions = np.random.randint(0, 5, size=1000)
    
    metrics = compute_metrics(predictions, labels, num_classes=5)
    assert 0.15 < metrics["accuracy"] < 0.30  # Around 20% for 5 classes
    print(f"[OK] Random predictions: accuracy={metrics['accuracy']:.3f} "
          f"(expected ~0.20)")
    print(f"     selectivity={metrics['selectivity']:.3f} "
          f"(majority_baseline={metrics['majority_baseline']:.3f})")


def test_mlp_vs_linear():
    """
    Verify MLP probe can learn nonlinear patterns that linear probe cannot.
    
    Create XOR-like data that is NOT linearly separable.
    MLP should outperform linear probe.
    """
    print("\n--- test_mlp_vs_linear ---")
    
    np.random.seed(42)
    
    # Create XOR-like 2D data embedded in higher dimensions
    input_dim = 32
    num_classes = 2
    n = 400
    
    # Generate 2D XOR pattern
    x1 = np.random.randn(n, 2)
    # XOR: label = 1 if (x>0 and y>0) or (x<0 and y<0), else 0
    xor_labels = ((x1[:, 0] > 0) == (x1[:, 1] > 0)).astype(int)
    
    # Embed in higher dimensions (pad with noise)
    features = np.zeros((n, input_dim), dtype=np.float32)
    features[:, :2] = x1
    features[:, 2:] = np.random.randn(n, input_dim - 2) * 0.1
    
    # Split
    train_f, train_l = features[:240], xor_labels[:240]
    val_f, val_l = features[240:320], xor_labels[240:320]
    test_f, test_l = features[320:], xor_labels[320:]
    
    # Linear probe
    linear_probe = create_probe("linear", input_dim, num_classes)
    linear_trainer = ProbeTrainer(
        probe=linear_probe, device="cuda",
        learning_rate=1e-2, batch_size=64, max_epochs=50, patience=10,
    )
    linear_result = linear_trainer.train(train_f, train_l, val_f, val_l, test_f, test_l)
    
    # MLP probe
    mlp_probe = create_probe("mlp", input_dim, num_classes, hidden_dim=64)
    mlp_trainer = ProbeTrainer(
        probe=mlp_probe, device="cuda",
        learning_rate=1e-2, batch_size=64, max_epochs=100, patience=15,
    )
    mlp_result = mlp_trainer.train(train_f, train_l, val_f, val_l, test_f, test_l)
    
    print(f"  Linear: test_acc={linear_result.test_accuracy:.3f}")
    print(f"  MLP:    test_acc={mlp_result.test_accuracy:.3f}")
    
    # Linear should be near chance (50%), MLP should do much better
    assert mlp_result.test_accuracy > linear_result.test_accuracy + 0.10, (
        "MLP should outperform linear on XOR data"
    )
    print(f"[OK] MLP outperforms linear on nonlinear data "
          f"(gap={mlp_result.test_accuracy - linear_result.test_accuracy:.3f})")


if __name__ == "__main__":
    print("=" * 60)
    print("PROBING PIPELINE TESTS")
    print("=" * 60)
    
    tests = [
        test_probe_architectures,
        test_training_converges_on_easy_data,
        test_training_fails_on_random_labels,
        test_metrics_computation,
        test_mlp_vs_linear,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAILED] {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    if failed == 0:
        print("\nAll probing pipeline tests passed.")
    else:
        print("\nFix failures before continuing.")
        sys.exit(1)
