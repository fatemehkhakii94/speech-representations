import numpy as np
import pytest

from speech_representations.analysis.cka import aligned_sample_ids, linear_cka


def test_linear_cka_invariances() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(120, 16))
    orthogonal, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    transformed = 4.5 * features @ orthogonal + 10.0
    assert linear_cka(features, transformed) == pytest.approx(1.0, abs=1e-10)


def test_linear_cka_rejects_misaligned_shapes() -> None:
    with pytest.raises(ValueError, match="same number"):
        linear_cka(np.ones((3, 2)), np.ones((4, 2)))


class _Store:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids


def test_alignment_uses_id_intersection() -> None:
    stores = [_Store(["c", "b", "a"]), _Store(["d", "b", "c"])]
    assert aligned_sample_ids(stores, maximum_samples=10, seed=1) == ["b", "c"]  # type: ignore[arg-type]
