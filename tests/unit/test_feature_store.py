from pathlib import Path

import h5py
import numpy as np
import pytest

from speech_representations.features.store import FeatureStore


def _cache(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.arange(3 * 2 * 4).reshape(3, 2, 4))
        handle.create_dataset(
            "ids", data=np.asarray(["a", "b", "c"], dtype=object), dtype=h5py.string_dtype()
        )
        handle.attrs["fingerprint"] = "example"


def test_store_selects_by_explicit_id(tmp_path: Path) -> None:
    path = tmp_path / "features.h5"
    _cache(path)
    store = FeatureStore(path)
    selected = store.load_layer(1, ["c", "a"])
    np.testing.assert_array_equal(selected, np.asarray([[20, 21, 22, 23], [4, 5, 6, 7]]))
    assert store.load_layers([0, 1], ["b"]).shape == (1, 2, 4)


def test_store_rejects_missing_ids(tmp_path: Path) -> None:
    path = tmp_path / "features.h5"
    _cache(path)
    store = FeatureStore(path)
    with pytest.raises(KeyError, match="absent"):
        store.load_layer(0, ["missing"])


def test_store_rejects_invalid_files_layers_and_duplicate_ids(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        FeatureStore(tmp_path / "missing.h5")

    path = tmp_path / "features.h5"
    _cache(path)
    store = FeatureStore(path)
    with pytest.raises(IndexError, match="outside"):
        store.load_layer(store.num_layers)

    with h5py.File(path, "r+") as handle:
        handle["ids"][2] = "b"
    with pytest.raises(ValueError, match="Duplicate IDs"):
        FeatureStore(path)
