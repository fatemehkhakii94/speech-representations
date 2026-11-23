from collections import Counter

from speech_representations.data.datasets import (
    select_cremad_cohort,
    select_librispeech_cohort,
)
from speech_representations.data.records import AudioRecord


def test_librispeech_cohort_is_bounded_and_chapter_diverse() -> None:
    records = [
        AudioRecord(
            id=f"{speaker}-{chapter}-{utterance}",
            path="unused.flac",
            dataset="librispeech",
            speaker_id=str(speaker),
            chapter_id=str(chapter),
        )
        for speaker in range(6)
        for chapter in range(3)
        for utterance in range(3)
    ]
    cohort = select_librispeech_cohort(records, 42, 3, 6, 6)

    speakers = Counter(record.speaker_id for record in cohort)
    assert len(cohort) == 18
    assert set(speakers.values()) == {6}
    for speaker in speakers:
        assert len({record.chapter_id for record in cohort if record.speaker_id == speaker}) == 3
    assert cohort == select_librispeech_cohort(records, 42, 3, 6, 6)


def test_cremad_cohort_balances_emotions_per_speaker() -> None:
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
    records = [
        AudioRecord(
            id=f"{speaker}-{emotion}-{index}",
            path="unused.wav",
            dataset="cremad",
            speaker_id=speaker,
            metadata={"emotion": emotion, "emotion_label": label},
        )
        for speaker in ("a", "b")
        for label, emotion in enumerate(emotions)
        for index in range(3)
    ]
    cohort = select_cremad_cohort(records, seed=7, samples_per_speaker=12)

    assert len(cohort) == 24
    for speaker in ("a", "b"):
        counts = Counter(
            record.metadata["emotion"] for record in cohort if record.speaker_id == speaker
        )
        assert set(counts.values()) == {2}
