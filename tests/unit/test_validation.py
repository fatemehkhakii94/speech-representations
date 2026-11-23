import pytest

from speech_representations.data.records import TaskRecord
from speech_representations.data.validation import _validate_task


def test_task_validation_accepts_group_disjoint_class_coverage() -> None:
    records = [
        TaskRecord(
            id=f"{split}-{label}",
            target=label,
            split=split,
            group=f"{split}-{label}",
        )
        for split in ("train", "val", "test")
        for label in (0, 1)
    ]
    report = _validate_task("example", records, {record.id for record in records}, "classification")
    assert report["records"] == 6
    assert report["splits"] == {"test": 2, "train": 2, "val": 2}


def test_task_validation_rejects_group_leakage() -> None:
    records = [
        TaskRecord("a", 0, "train", "shared"),
        TaskRecord("b", 0, "val", "shared"),
        TaskRecord("c", 0, "test", "test-only"),
    ]
    with pytest.raises(ValueError, match="leaks"):
        _validate_task("example", records, {"a", "b", "c"}, "classification")
