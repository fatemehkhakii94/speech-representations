import pytest

from speech_representations.data.splits import assert_group_disjoint, assign_groups


def test_group_assignment_is_deterministic_and_disjoint() -> None:
    groups = [f"speaker-{index}" for index in range(20)]
    first = assign_groups(groups, seed=42)
    second = assign_groups(groups, seed=42)
    assert first == second
    assert set(first.values()) == {"train", "val", "test"}
    assert_group_disjoint(groups, [first[group] for group in groups])


def test_group_leakage_is_detected() -> None:
    with pytest.raises(ValueError, match="spanning multiple splits"):
        assert_group_disjoint(["a", "a"], ["train", "test"])
