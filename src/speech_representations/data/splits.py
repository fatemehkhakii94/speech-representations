"""Deterministic split strategies with explicit leakage checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T", bound=Hashable)


def assign_groups(
    groups: Sequence[T],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[T, str]:
    """Assign complete groups to train/validation/test deterministically."""
    unique = sorted(set(groups), key=str)
    if len(unique) < 3:
        raise ValueError("At least three groups are required for train/validation/test splitting")
    rng = np.random.default_rng(seed)
    order = np.asarray(unique, dtype=object)
    rng.shuffle(order)
    n_train = max(1, int(len(order) * train_ratio))
    n_val = max(1, int(len(order) * val_ratio))
    if n_train + n_val >= len(order):
        n_train = len(order) - 2
        n_val = 1
    assignments: dict[T, str] = {}
    for index, group in enumerate(order.tolist()):
        assignments[group] = "train" if index < n_train else "val" if index < n_train + n_val else "test"
    return assignments


def assign_within_class_groups(
    class_to_groups: dict[T, Sequence[Hashable]], seed: int
) -> dict[tuple[T, Hashable], str]:
    """Split groups within every class, preserving all classes in every split."""
    result: dict[tuple[T, Hashable], str] = {}
    for offset, (class_id, groups) in enumerate(sorted(class_to_groups.items(), key=lambda item: str(item[0]))):
        assignments = assign_groups(groups, seed + offset)
        result.update({(class_id, group): split for group, split in assignments.items()})
    return result


def assert_group_disjoint(groups: Sequence[Hashable], splits: Sequence[str]) -> None:
    observed: dict[Hashable, set[str]] = defaultdict(set)
    for group, split in zip(groups, splits, strict=True):
        observed[group].add(split)
    leaked = [group for group, values in observed.items() if len(values) > 1]
    if leaked:
        raise ValueError(f"Found {len(leaked)} groups spanning multiple splits; examples: {leaked[:5]}")

