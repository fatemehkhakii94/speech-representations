"""Dataset acquisition and creation of leakage-aware experiment manifests."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import parselmouth
import torchaudio
from parselmouth.praat import call
from tqdm import tqdm

from speech_representations.config import ProjectConfig
from speech_representations.data.audio import SAMPLE_RATE, load_audio, match_length, mix_at_snr
from speech_representations.data.records import (
    AudioRecord,
    TaskRecord,
    read_audio_records,
    write_jsonl,
)
from speech_representations.data.splits import (
    assert_group_disjoint,
    assign_groups,
    assign_within_class_groups,
)

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
LIBRISPEECH_MD5 = "2a93770f6d5c6c964bc36631d331a522"
MUSAN_URL = (
    "https://huggingface.co/datasets/Emulated-Inc/enhancement/resolve/main/"
    "musan.tar.gz?download=true"
)
MUSAN_MD5 = "0c472d4fc0c5141eca47ad1ffeb2a7df"
CREMAD_REPOSITORY = "https://github.com/CheyneyComputerScience/CREMA-D.git"

EMOTIONS = {
    "ANG": (0, "angry"),
    "DIS": (1, "disgust"),
    "FEA": (2, "fear"),
    "HAP": (3, "happy"),
    "NEU": (4, "neutral"),
    "SAD": (5, "sad"),
}
NOISE_LABELS = {"clean": 0, "noise": 1, "music": 2, "speech": 3}


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_md5: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected_md5 and _file_md5(destination) == expected_md5:
        return
    curl = shutil.which("curl")
    if curl is not None:
        subprocess.run(
            [
                curl,
                "--location",
                "--fail",
                "--retry",
                "3",
                "--continue-at",
                "-",
                "--output",
                str(destination),
                url,
            ],
            check=True,
        )
    else:
        destination.unlink(missing_ok=True)

        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=destination.name) as progress:

            def report(blocks: int, block_size: int, total: int) -> None:
                if total > 0:
                    progress.total = total
                progress.update(blocks * block_size - progress.n)

            urllib.request.urlretrieve(url, destination, reporthook=report)
    if expected_md5 and _file_md5(destination) != expected_md5:
        raise RuntimeError(f"Checksum mismatch for {destination}; delete it and retry")


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        bundle.extractall(destination, filter="data")


def acquire_librispeech(data_root: Path) -> Path:
    target = data_root / "raw" / "librispeech" / "LibriSpeech" / "train-clean-100"
    marker = target / ".prepared"
    if marker.is_file():
        return target
    archive = data_root / "downloads" / "train-clean-100.tar.gz"
    _download(LIBRISPEECH_URL, archive, LIBRISPEECH_MD5)
    _safe_extract(archive, data_root / "raw" / "librispeech")
    file_count = sum(1 for _ in target.rglob("*.flac")) if target.is_dir() else 0
    if file_count < 28_000:
        raise RuntimeError(f"LibriSpeech extraction is incomplete: found {file_count} FLAC files")
    marker.write_text(f"md5={LIBRISPEECH_MD5}\nfiles={file_count}\n", encoding="utf-8")
    archive.unlink()
    return target


def acquire_musan(data_root: Path) -> Path:
    target = data_root / "raw" / "musan" / "musan"
    marker = target / ".prepared"
    if marker.is_file():
        return target
    archive = data_root / "downloads" / "musan.tar.gz"
    _download(MUSAN_URL, archive, MUSAN_MD5)
    _safe_extract(archive, data_root / "raw" / "musan")
    file_count = sum(1 for _ in target.rglob("*.wav")) if target.is_dir() else 0
    if file_count < 1_000:
        raise RuntimeError(f"MUSAN extraction is incomplete: found {file_count} WAV files")
    marker.write_text(f"md5={MUSAN_MD5}\nfiles={file_count}\n", encoding="utf-8")
    archive.unlink()
    return target


def acquire_cremad(data_root: Path) -> Path:
    repository = data_root / "raw" / "cremad" / "CREMA-D"
    audio = repository / "AudioWAV"
    existing = list(audio.glob("*.wav"))
    if len(existing) >= 7_000 and all(path.stat().st_size > 1_000 for path in existing[:20]):
        return audio
    if shutil.which("git") is None:
        raise RuntimeError("Git is required to acquire CREMA-D")
    if not repository.exists():
        repository.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", CREMAD_REPOSITORY, str(repository)],
            check=True,
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
        )
    if shutil.which("git-lfs") is not None:
        subprocess.run(
            ["git", "lfs", "pull", "--include", "AudioWAV/**", "--exclude", ""],
            cwd=repository,
            check=True,
        )
    files = list(audio.glob("*.wav"))
    if len(files) < 7_000 or any(path.stat().st_size < 1_000 for path in files[:20]):
        raise RuntimeError("CREMA-D audio is incomplete. Install Git LFS and rerun preparation.")
    return audio


def scan_librispeech(
    root: Path, minimum_seconds: float = 1.0, maximum_seconds: float = 10.0
) -> list[AudioRecord]:
    records: list[AudioRecord] = []
    for path in tqdm(sorted(root.rglob("*.flac")), desc="Scanning LibriSpeech"):
        info = torchaudio.info(str(path))
        duration = info.num_frames / info.sample_rate
        if minimum_seconds <= duration <= maximum_seconds:
            speaker, chapter, _ = path.stem.split("-", maxsplit=2)
            records.append(
                AudioRecord(
                    id=path.stem,
                    path=str(path.resolve()),
                    dataset="librispeech",
                    speaker_id=speaker,
                    chapter_id=chapter,
                    duration=duration,
                )
            )
    if not records:
        raise RuntimeError(f"No eligible LibriSpeech audio found under {root}")
    return records


def scan_cremad(
    root: Path, minimum_seconds: float = 1.0, maximum_seconds: float = 10.0
) -> list[AudioRecord]:
    records: list[AudioRecord] = []
    for path in tqdm(sorted(root.glob("*.wav")), desc="Scanning CREMA-D"):
        parts = path.stem.split("_")
        if len(parts) != 4 or parts[2] not in EMOTIONS:
            continue
        info = torchaudio.info(str(path))
        duration = info.num_frames / info.sample_rate
        if minimum_seconds <= duration <= maximum_seconds:
            label, name = EMOTIONS[parts[2]]
            records.append(
                AudioRecord(
                    id=path.stem,
                    path=str(path.resolve()),
                    dataset="cremad",
                    speaker_id=parts[0],
                    duration=duration,
                    metadata={"emotion": name, "emotion_label": label, "sentence": parts[1]},
                )
            )
    if not records:
        raise RuntimeError(f"No eligible CREMA-D audio found under {root}")
    return records


def _round_robin_cap(records: list[AudioRecord], limit: int, seed: int) -> list[AudioRecord]:
    rng = np.random.default_rng(seed)
    by_chapter: dict[str, list[AudioRecord]] = defaultdict(list)
    for record in records:
        by_chapter[str(record.chapter_id)].append(record)
    for values in by_chapter.values():
        rng.shuffle(values)
    selected: list[AudioRecord] = []
    chapters = sorted(by_chapter)
    while chapters and len(selected) < limit:
        for chapter in list(chapters):
            if by_chapter[chapter] and len(selected) < limit:
                selected.append(by_chapter[chapter].pop())
            if not by_chapter[chapter]:
                chapters.remove(chapter)
    return selected


def select_librispeech_cohort(
    records: list[AudioRecord],
    seed: int,
    speakers: int,
    samples_per_speaker: int,
    minimum_samples: int,
) -> list[AudioRecord]:
    """Select a deterministic, chapter-diverse cohort for the shared benchmark."""
    by_speaker: dict[str, list[AudioRecord]] = defaultdict(list)
    for record in records:
        by_speaker[str(record.speaker_id)].append(record)
    eligible = sorted(
        speaker
        for speaker, values in by_speaker.items()
        if len(values) >= max(minimum_samples, samples_per_speaker)
        and len({str(value.chapter_id) for value in values}) >= 3
    )
    if len(eligible) < speakers:
        raise ValueError(f"Requested {speakers} speakers but only {len(eligible)} are eligible")
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(eligible, size=speakers, replace=False).tolist())
    cohort = [
        record
        for offset, speaker in enumerate(chosen)
        for record in _round_robin_cap(
            by_speaker[speaker], samples_per_speaker, seed + offset
        )
    ]
    return sorted(cohort, key=lambda record: record.id)


def select_cremad_cohort(
    records: list[AudioRecord], seed: int, samples_per_speaker: int
) -> list[AudioRecord]:
    """Cap each actor with round-robin emotion sampling to retain class balance."""
    by_speaker: dict[str, list[AudioRecord]] = defaultdict(list)
    for record in records:
        by_speaker[str(record.speaker_id)].append(record)
    selected: list[AudioRecord] = []
    for offset, speaker in enumerate(sorted(by_speaker)):
        rng = np.random.default_rng(seed + offset)
        by_emotion: dict[str, list[AudioRecord]] = defaultdict(list)
        for record in by_speaker[speaker]:
            by_emotion[str(record.metadata["emotion"])].append(record)
        for values in by_emotion.values():
            rng.shuffle(values)
        emotions = sorted(by_emotion)
        speaker_selected: list[AudioRecord] = []
        while emotions and len(speaker_selected) < samples_per_speaker:
            for emotion in list(emotions):
                if by_emotion[emotion] and len(speaker_selected) < samples_per_speaker:
                    speaker_selected.append(by_emotion[emotion].pop())
                if not by_emotion[emotion]:
                    emotions.remove(emotion)
        selected.extend(speaker_selected)
    return sorted(selected, key=lambda record: record.id)


def build_speaker_task(
    records: list[AudioRecord], seed: int, minimum: int, maximum: int
) -> list[TaskRecord]:
    by_speaker: dict[str, list[AudioRecord]] = defaultdict(list)
    for record in records:
        by_speaker[str(record.speaker_id)].append(record)
    eligible = {
        speaker: values
        for speaker, values in by_speaker.items()
        if len(values) >= minimum and len({value.chapter_id for value in values}) >= 3
    }
    class_ids = {speaker: index for index, speaker in enumerate(sorted(eligible))}
    selected = {
        speaker: _round_robin_cap(values, maximum, seed + class_ids[speaker])
        for speaker, values in eligible.items()
    }
    chapter_map = assign_within_class_groups(
        {
            speaker: sorted({str(record.chapter_id) for record in values})
            for speaker, values in selected.items()
        },
        seed,
    )
    task = [
        TaskRecord(
            id=record.id,
            target=class_ids[speaker],
            split=chapter_map[(speaker, str(record.chapter_id))],
            group=f"{speaker}:{record.chapter_id}",
            label_name=speaker,
        )
        for speaker, values in selected.items()
        for record in values
    ]
    return sorted(task, key=lambda item: item.id)


def _median_f0_semitones(path: str) -> float | None:
    sound = parselmouth.Sound(path)
    pitch = call(sound, "To Pitch", 0.0, 75.0, 500.0)
    values = pitch.selected_array["frequency"]
    voiced = values[values > 0]
    if voiced.size == 0:
        return None
    return float(12.0 * np.log2(np.median(voiced) / 55.0))


def build_pitch_task(records: list[AudioRecord], seed: int) -> list[TaskRecord]:
    speaker_map = assign_groups([str(record.speaker_id) for record in records], seed)
    task: list[TaskRecord] = []
    for record in tqdm(records, desc="Measuring pitch"):
        target = _median_f0_semitones(record.path)
        if target is not None and np.isfinite(target):
            speaker = str(record.speaker_id)
            task.append(TaskRecord(record.id, target, speaker_map[speaker], speaker))
    assert_group_disjoint([item.group for item in task], [item.split for item in task])
    return task


def build_emotion_task(records: list[AudioRecord], seed: int) -> list[TaskRecord]:
    speakers = [str(record.speaker_id) for record in records]
    assignments = assign_groups(speakers, seed, train_ratio=0.6, val_ratio=0.2)
    task = [
        TaskRecord(
            id=record.id,
            target=int(record.metadata["emotion_label"]),
            split=assignments[str(record.speaker_id)],
            group=str(record.speaker_id),
            label_name=str(record.metadata["emotion"]),
        )
        for record in records
    ]
    assert_group_disjoint([item.group for item in task], [item.split for item in task])
    return task


def _noise_files(root: Path) -> dict[str, list[Path]]:
    result = {
        category: sorted((root / category).rglob("*.wav"))
        for category in ("noise", "music", "speech")
    }
    missing = [category for category, paths in result.items() if not paths]
    if missing:
        raise RuntimeError(f"MUSAN contains no files for categories: {missing}")
    return result


def build_noise_dataset(
    source_records: list[AudioRecord],
    musan_root: Path,
    output_root: Path,
    seed: int,
    snr_db: float,
    maximum_sources: int,
) -> tuple[list[AudioRecord], list[TaskRecord]]:
    rng = np.random.default_rng(seed)
    chosen = list(source_records)
    rng.shuffle(chosen)
    chosen = chosen[:maximum_sources]
    split_map = assign_groups([record.id for record in chosen], seed)
    noises = _noise_files(musan_root)
    audio_records: list[AudioRecord] = []
    task_records: list[TaskRecord] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for source in tqdm(chosen, desc="Creating matched noise conditions"):
        clean = load_audio(source.path, normalize=False)
        clean = clean - clean.mean()
        for category, label in NOISE_LABELS.items():
            identifier = f"{source.id}__{category}"
            if category == "clean":
                path = source.path
            else:
                candidates = noises[category]
                noise_path = candidates[int(rng.integers(0, len(candidates)))]
                raw_noise = load_audio(noise_path, normalize=False)
                max_start = max(0, raw_noise.numel() - clean.numel())
                start = int(rng.integers(0, max_start + 1)) if max_start else 0
                noise = match_length(raw_noise, clean.numel(), start)
                noise = noise - noise.mean()
                mixed = mix_at_snr(clean, noise, snr_db)
                peak = mixed.abs().max().clamp_min(1e-8)
                if peak > 0.99:
                    mixed = mixed * (0.99 / peak)
                destination = output_root / category / f"{source.id}.flac"
                destination.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(destination), mixed.unsqueeze(0), SAMPLE_RATE, format="flac")
                path = str(destination.resolve())
            audio_records.append(
                AudioRecord(
                    id=identifier,
                    path=str(Path(path).resolve()),
                    dataset="noise",
                    speaker_id=source.speaker_id,
                    duration=source.duration,
                    metadata={"condition": category, "source_id": source.id, "snr_db": snr_db},
                )
            )
            task_records.append(
                TaskRecord(
                    id=identifier,
                    target=label,
                    split=split_map[source.id],
                    group=source.id,
                    label_name=category,
                )
            )
    assert_group_disjoint(
        [item.group for item in task_records], [item.split for item in task_records]
    )
    return audio_records, task_records


def prepare_data(config: ProjectConfig, task_names: Iterable[str]) -> dict[str, int]:
    """Acquire inputs and write canonical dataset/task manifests."""
    requested = set(task_names)
    unknown = requested - config.tasks.keys()
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")
    manifests = config.paths.data / "manifests"
    counts: dict[str, int] = {}

    librispeech_records: list[AudioRecord] | None = None
    if requested & {"speaker", "pitch", "noise"}:
        inventory = manifests / "inventory" / "librispeech.jsonl"
        if inventory.is_file():
            librispeech_records = read_audio_records(inventory)
        else:
            librispeech_records = scan_librispeech(acquire_librispeech(config.paths.data))
            write_jsonl(inventory, librispeech_records)
        speaker_options = config.task("speaker").options
        librispeech_records = select_librispeech_cohort(
            librispeech_records,
            config.runtime.seed,
            int(speaker_options["cohort_speakers"]),
            int(speaker_options["cohort_samples_per_speaker"]),
            int(speaker_options["min_samples_per_speaker"]),
        )
        if requested & {"speaker", "pitch"}:
            write_jsonl(manifests / "datasets" / "librispeech.jsonl", librispeech_records)

    if "speaker" in requested and librispeech_records is not None:
        task = config.task("speaker")
        records = build_speaker_task(
            librispeech_records,
            config.runtime.seed,
            int(task.options["min_samples_per_speaker"]),
            int(task.options["max_samples_per_speaker"]),
        )
        write_jsonl(manifests / "tasks" / "speaker.jsonl", records)
        counts["speaker"] = len(records)

    if "pitch" in requested and librispeech_records is not None:
        records = build_pitch_task(librispeech_records, config.runtime.seed)
        write_jsonl(manifests / "tasks" / "pitch.jsonl", records)
        counts["pitch"] = len(records)

    if "emotion" in requested:
        emotion_task = config.task("emotion")
        inventory = manifests / "inventory" / "cremad.jsonl"
        if inventory.is_file():
            cremad_inventory = read_audio_records(inventory)
        else:
            cremad_inventory = scan_cremad(acquire_cremad(config.paths.data))
            write_jsonl(inventory, cremad_inventory)
        cremad = select_cremad_cohort(
            cremad_inventory,
            config.runtime.seed,
            int(emotion_task.options["max_samples_per_speaker"]),
        )
        write_jsonl(manifests / "datasets" / "cremad.jsonl", cremad)
        records = build_emotion_task(cremad, config.runtime.seed)
        write_jsonl(manifests / "tasks" / "emotion.jsonl", records)
        counts["emotion"] = len(records)

    if "noise" in requested and librispeech_records is not None:
        task = config.task("noise")
        audio, records = build_noise_dataset(
            librispeech_records,
            acquire_musan(config.paths.data),
            config.paths.data / "processed" / "noise",
            config.runtime.seed,
            float(task.options["snr_db"]),
            int(task.options["max_source_utterances"]),
        )
        write_jsonl(manifests / "datasets" / "noise.jsonl", audio)
        write_jsonl(manifests / "tasks" / "noise.jsonl", records)
        counts["noise"] = len(records)

    return counts
