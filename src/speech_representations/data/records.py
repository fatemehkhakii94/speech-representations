"""Serializable records used by dataset and feature pipelines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class AudioRecord:
    id: str
    path: str
    dataset: str
    speaker_id: str | None = None
    chapter_id: str | None = None
    duration: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskRecord:
    id: str
    target: int | float
    split: str
    group: str
    label_name: str | None = None


def write_jsonl(path: str | Path, records: Iterable[AudioRecord | TaskRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    temporary.replace(destination)


def read_audio_records(path: str | Path) -> list[AudioRecord]:
    return [AudioRecord(**json.loads(line)) for line in Path(path).read_text().splitlines() if line]


def read_task_records(path: str | Path) -> list[TaskRecord]:
    return [TaskRecord(**json.loads(line)) for line in Path(path).read_text().splitlines() if line]

