from pathlib import Path

from speech_representations.data.records import (
    AudioRecord,
    TaskRecord,
    read_audio_records,
    read_task_records,
    write_jsonl,
)
from speech_representations.utils import (
    atomic_write_json,
    canonical_hash,
    environment_snapshot,
    git_revision,
    source_fingerprint,
)


def test_record_manifests_round_trip(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.jsonl"
    task_path = tmp_path / "task.jsonl"
    audio = AudioRecord(
        id="sample-1",
        path="audio.wav",
        dataset="example",
        speaker_id="speaker-1",
        metadata={"source": "fixture"},
    )
    task = TaskRecord(id="sample-1", target=2, split="train", group="speaker-1")

    write_jsonl(audio_path, [audio])
    write_jsonl(task_path, [task])

    assert read_audio_records(audio_path) == [audio]
    assert read_task_records(task_path) == [task]


def test_provenance_utilities_are_stable(tmp_path: Path) -> None:
    first = {"model": "example", "layers": [1, 2], "seed": 42}
    second = {"seed": 42, "layers": [1, 2], "model": "example"}
    assert canonical_hash(first) == canonical_hash(second)

    destination = tmp_path / "nested" / "metadata.json"
    atomic_write_json(destination, first)
    assert destination.read_text(encoding="utf-8").endswith("\n")
    assert not list(destination.parent.glob("*.tmp"))

    snapshot = environment_snapshot(("torch", "definitely-not-installed"))
    assert snapshot["packages"]["definitely-not-installed"] == "not-installed"
    assert snapshot["python"]
    assert git_revision(Path.cwd())
    assert len(source_fingerprint(Path.cwd())) == 64
