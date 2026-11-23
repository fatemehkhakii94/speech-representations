"""Structural validation for prepared benchmark manifests."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from speech_representations.config import ProjectConfig
from speech_representations.data.records import (
    AudioRecord,
    TaskRecord,
    read_audio_records,
    read_task_records,
)


def _validate_task(
    name: str, records: list[TaskRecord], dataset_ids: set[str], kind: str
) -> dict[str, Any]:
    if not records:
        raise ValueError(f"Task {name} is empty")
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Task {name} contains duplicate IDs")
    missing_ids = set(ids) - dataset_ids
    if missing_ids:
        raise ValueError(f"Task {name} references {len(missing_ids)} absent dataset IDs")
    split_counts = Counter(record.split for record in records)
    expected_splits = {"train", "val", "test"}
    if set(split_counts) != expected_splits:
        raise ValueError(f"Task {name} has invalid splits: {sorted(split_counts)}")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_splits[record.group].add(record.split)
    leaked = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"Task {name} leaks {len(leaked)} groups across splits")

    class_counts: dict[str, dict[str, int]] | None = None
    if kind == "classification":
        classes = {str(record.target) for record in records}
        class_counts = {}
        for split in sorted(expected_splits):
            counts = Counter(str(record.target) for record in records if record.split == split)
            if set(counts) != classes:
                raise ValueError(f"Task {name} does not represent every class in {split}")
            class_counts[split] = dict(sorted(counts.items()))
    elif not all(np.isfinite(float(record.target)) for record in records):
        raise ValueError(f"Task {name} contains non-finite regression targets")

    return {
        "records": len(records),
        "groups": len(group_splits),
        "splits": dict(sorted(split_counts.items())),
        "class_counts": class_counts,
    }


def validate_prepared_data(config: ProjectConfig) -> dict[str, Any]:
    """Validate IDs, files, splits, groups, and class coverage for every task."""
    manifest_root = config.paths.data / "manifests"
    datasets: dict[str, list[AudioRecord]] = {}
    dataset_report: dict[str, Any] = {}
    for dataset_name in sorted({task.dataset for task in config.tasks.values()}):
        path = manifest_root / "datasets" / f"{dataset_name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Dataset manifest missing: {path}")
        records = read_audio_records(path)
        ids = [record.id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Dataset {dataset_name} contains duplicate IDs")
        missing_files = [record.path for record in records if not Path(record.path).is_file()]
        if missing_files:
            raise ValueError(f"Dataset {dataset_name} references {len(missing_files)} missing files")
        datasets[dataset_name] = records
        dataset_report[dataset_name] = {"records": len(records), "missing_files": 0}

    task_report: dict[str, Any] = {}
    for name, task in config.tasks.items():
        path = manifest_root / "tasks" / f"{name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Task manifest missing: {path}")
        task_report[name] = _validate_task(
            name,
            read_task_records(path),
            {record.id for record in datasets[task.dataset]},
            task.kind,
        )
    return {"datasets": dataset_report, "tasks": task_report, "status": "valid"}
