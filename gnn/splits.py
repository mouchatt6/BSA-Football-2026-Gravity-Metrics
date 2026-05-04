"""Grouped train/val/test splits.

Splits are made at the *gameId* level so that frames from the same play and
plays from the same game never leak across splits.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class GroupSplit:
    train_idx: List[int]
    val_idx: List[int]
    test_idx: List[int]
    train_games: List[int]
    val_games: List[int]
    test_games: List[int]


def grouped_split(
    game_ids: Sequence[int],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> GroupSplit:
    """Random split of unique gameIds; returns positional indices for each side."""
    rng = random.Random(seed)
    unique = sorted(set(int(g) for g in game_ids))
    rng.shuffle(unique)
    n = len(unique)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    test = set(unique[:n_test])
    val = set(unique[n_test : n_test + n_val])
    train = set(unique[n_test + n_val :])

    train_idx, val_idx, test_idx = [], [], []
    for i, g in enumerate(game_ids):
        gi = int(g)
        if gi in train:
            train_idx.append(i)
        elif gi in val:
            val_idx.append(i)
        elif gi in test:
            test_idx.append(i)

    return GroupSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_games=sorted(train),
        val_games=sorted(val),
        test_games=sorted(test),
    )


def leave_one_week_out(
    weeks: Sequence[int],
    test_week: int,
) -> Tuple[List[int], List[int]]:
    train_idx = [i for i, w in enumerate(weeks) if int(w) != int(test_week)]
    test_idx = [i for i, w in enumerate(weeks) if int(w) == int(test_week)]
    return train_idx, test_idx
