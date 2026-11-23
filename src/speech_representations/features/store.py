"""Resumable HDF5 feature caches with strict provenance validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from speech_representations.config import ProjectConfig
from speech_representations.data.audio import load_audio
from speech_representations.data.records import AudioRecord
from speech_representations.features.extractor import SpeechFeatureExtractor
from speech_representations.utils import (
    atomic_write_json,
    canonical_hash,
    environment_snapshot,
    git_revision,
    resolve_device,
    source_fingerprint,
)

CACHE_SCHEMA_VERSION = 1


def _cache_identity(
    config: ProjectConfig, model_name: str, dataset: str, records: Sequence[AudioRecord]
) -> dict[str, Any]:
    model = config.model(model_name)
    record_identity = [
        {
            "id": record.id,
            "path": record.path,
            "duration": record.duration,
            "metadata": record.metadata,
        }
        for record in records
    ]
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model": asdict(model),
        "dataset": dataset,
        "sample_count": len(records),
        "sample_hash": canonical_hash(record_identity),
        "pooling": "valid-frame-mean",
        "sample_rate": 16_000,
        "source_fingerprint": source_fingerprint(config.root),
    }


class FeatureStore:
    """Read-only access to a validated feature cache."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Feature cache not found: {self.path}")
        with h5py.File(self.path, "r") as handle:
            self.ids = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in handle["ids"][:]
            ]
            self.num_layers = int(handle["features"].shape[1])
            self.hidden_size = int(handle["features"].shape[2])
            self.fingerprint = str(handle.attrs["fingerprint"])
        if len(self.ids) != len(set(self.ids)):
            raise ValueError(f"Duplicate IDs in feature cache: {self.path}")
        self._indices = {identifier: index for index, identifier in enumerate(self.ids)}

    def indices(self, ids: Sequence[str]) -> np.ndarray:
        missing = [identifier for identifier in ids if identifier not in self._indices]
        if missing:
            raise KeyError(
                f"{len(missing)} IDs are absent from {self.path}; examples: {missing[:5]}"
            )
        return np.asarray([self._indices[identifier] for identifier in ids], dtype=np.int64)

    def load_layer(self, layer: int, ids: Sequence[str] | None = None) -> np.ndarray:
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"Layer {layer} is outside [0, {self.num_layers})")
        with h5py.File(self.path, "r") as handle:
            features = handle["features"][:, layer, :]
        return features if ids is None else features[self.indices(ids)]

    def load_layers(self, layers: Sequence[int], ids: Sequence[str]) -> np.ndarray:
        return np.stack([self.load_layer(layer, ids) for layer in layers], axis=1)


def extract_features(
    config: ProjectConfig,
    model_name: str,
    dataset: str,
    records: Sequence[AudioRecord],
    overwrite: bool = False,
) -> Path:
    """Extract or resume one model/dataset cache and return its final path."""
    if not records:
        raise ValueError("No audio records were supplied")
    record_ids = [record.id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"Duplicate record IDs supplied for {dataset}")
    feature_dir = config.paths.features / model_name
    final_path = feature_dir / f"{dataset}.h5"
    partial_path = feature_dir / f"{dataset}.partial.h5"
    manifest_path = feature_dir / f"{dataset}.manifest.json"
    identity = _cache_identity(config, model_name, dataset, records)
    fingerprint = canonical_hash(identity)
    feature_dir.mkdir(parents=True, exist_ok=True)

    if final_path.exists() and not overwrite:
        existing_store = FeatureStore(final_path)
        if (
            existing_store.fingerprint != fingerprint
            or existing_store.ids != record_ids
            or existing_store.num_layers != config.model(model_name).representation_layers
            or existing_store.hidden_size != config.model(model_name).hidden_size
        ):
            raise RuntimeError(
                f"Existing cache fingerprint does not match {dataset}/{model_name}. "
                "Use --overwrite only after verifying the intended inputs."
            )
        return final_path
    if overwrite:
        for path in (final_path, partial_path, manifest_path):
            if path.exists():
                path.unlink()

    model = config.model(model_name)
    string_type = h5py.string_dtype(encoding="utf-8")
    if not partial_path.exists():
        with h5py.File(partial_path, "w") as handle:
            handle.create_dataset(
                "features",
                shape=(len(records), model.representation_layers, model.hidden_size),
                dtype="float32",
                chunks=(1, model.representation_layers, model.hidden_size),
                compression="lzf",
            )
            handle.create_dataset(
                "ids",
                data=np.asarray(record_ids, dtype=object),
                dtype=string_type,
            )
            handle.create_dataset("completed", shape=(len(records),), dtype="bool", fillvalue=False)
            handle.attrs["fingerprint"] = fingerprint
            handle.attrs["schema_version"] = CACHE_SCHEMA_VERSION
    else:
        with h5py.File(partial_path, "r") as handle:
            stored_ids = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in handle["ids"][:]
            ]
            expected_shape = (
                len(records),
                model.representation_layers,
                model.hidden_size,
            )
            if (
                str(handle.attrs.get("fingerprint", "")) != fingerprint
                or stored_ids != record_ids
                or handle["features"].shape != expected_shape
                or handle["completed"].shape != (len(records),)
            ):
                raise RuntimeError(f"Partial cache has incompatible inputs: {partial_path}")

    device = resolve_device(config.runtime.device)
    extractor = SpeechFeatureExtractor(model, device, config.runtime.precision)
    batch_size = config.runtime.extraction_batch_size
    with h5py.File(partial_path, "r+") as handle:
        completed = handle["completed"][:]
        pending = np.flatnonzero(~completed).tolist()
        progress = tqdm(
            total=len(records), initial=int(completed.sum()), desc=f"{model_name}/{dataset}"
        )
        for offset in range(0, len(pending), batch_size):
            indices = pending[offset : offset + batch_size]
            waveforms = [load_audio(records[index].path) for index in indices]
            features = extractor.extract(waveforms)
            handle["features"][indices, :, :] = features
            handle["completed"][indices] = True
            handle.flush()
            progress.update(len(indices))
        progress.close()
        if not bool(handle["completed"][:].all()):
            raise RuntimeError(f"Extraction stopped before cache completion: {partial_path}")

    partial_path.replace(final_path)
    manifest = {
        **identity,
        "fingerprint": fingerprint,
        "cache": str(final_path.relative_to(config.root)),
        "git_revision": git_revision(config.root),
        "environment": environment_snapshot(
            ("torch", "torchaudio", "transformers", "h5py", "numpy")
        ),
    }
    atomic_write_json(manifest_path, manifest)
    return final_path
