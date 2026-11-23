import numpy as np
import torch

from speech_representations.probing.models import create_probe
from speech_representations.probing.trainer import standardize_splits, train_probe
from speech_representations.utils import seed_everything


def test_standardization_uses_training_statistics_only() -> None:
    train = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    validation = np.asarray([[100.0, 100.0]], dtype=np.float32)
    test = np.asarray([[-100.0, -100.0]], dtype=np.float32)
    normalized_train, normalized_validation, normalized_test = standardize_splits(
        train, validation, test
    )
    np.testing.assert_allclose(normalized_train.mean(axis=0), 0.0, atol=1e-6)
    assert normalized_validation.mean() > 10
    assert normalized_test.mean() < -10


def test_linear_classification_probe_converges_on_cpu() -> None:
    rng = np.random.default_rng(9)
    features = rng.normal(size=(360, 12)).astype(np.float32)
    labels = (features[:, 0] + features[:, 1] > 0).astype(np.int64)
    seed_everything(9)
    result = train_probe(
        create_probe("linear", 12, 2),
        "classification",
        features[:240],
        labels[:240],
        features[240:300],
        labels[240:300],
        features[300:],
        labels[300:],
        torch.device("cpu"),
        learning_rate=0.03,
        batch_size=64,
        max_epochs=40,
        patience=8,
    )
    assert result.test["accuracy"] > 0.9


def test_linear_regression_probe_converges_on_cpu() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(300, 8)).astype(np.float32)
    # A large offset catches regressors that normalize inputs but not targets.
    labels = (50 + 2 * features[:, 0] - features[:, 1]).astype(np.float32)
    seed_everything(4)
    result = train_probe(
        create_probe("linear", 8, 1),
        "regression",
        features[:200],
        labels[:200],
        features[200:250],
        labels[200:250],
        features[250:],
        labels[250:],
        torch.device("cpu"),
        learning_rate=0.03,
        batch_size=64,
        max_epochs=60,
        patience=10,
    )
    assert result.test["r2"] > 0.95
