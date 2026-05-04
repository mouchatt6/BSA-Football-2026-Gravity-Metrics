"""Play eligibility and pass-rush-window logic.

Rules (mirrors the existing pipeline):
    - Drop screens, RPOs, QB spikes (`is_gravity_candidate_base`)
    - Drop plays with TTT <= 1.5s AND throw behind LOS
    - Drop dropBackType not in {Traditional, Scramble}
    - Pass-rush window: [ball_snap + 0.5s, min(end_event, snap+3.0s))
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    MIN_WINDOW_LENGTH,
    SNAP_EVENTS,
    SNAP_LAG_FRAMES,
    WINDOW_END_EVENTS,
    WINDOW_HARD_CAP_FRAMES,
)


def filter_eligible_plays(play_context: pd.DataFrame) -> pd.DataFrame:
    """Return play_context filtered to gravity-eligible pass-rush plays."""
    df = play_context.copy()
    df["is_gravity_candidate_base"] = df["is_gravity_candidate_base"].astype("boolean").fillna(False)

    eligible_dropback = df["dropBackType"].astype(str).isin({"Traditional", "Scramble"})
    bad_ttt = df["ttt_le_1_5_and_behind_los"].astype(str).str.lower().isin({"true", "1", "1.0"})

    mask = df["is_gravity_candidate_base"].astype(bool) & eligible_dropback & ~bad_ttt
    return df.loc[mask].reset_index(drop=True)


def _first_frame_with_event(events: pd.Series, frame_ids: pd.Series, target_events: Iterable[str]) -> Optional[int]:
    targets = set(target_events)
    mask = events.isin(targets)
    if not mask.any():
        return None
    return int(frame_ids[mask].min())


def find_pass_rush_window(play_frames: pd.DataFrame) -> Optional[Tuple[int, int]]:
    """Return [start_frame, end_frame) for the pass-rush window of a single play.

    play_frames is the per-frame slice for one (gameId, playId), restricted to
    a single nflId (any player works because events are play-level on the
    canonical row, but in this dataset the event column is repeated for every
    player per frame, so we collapse on frameID).
    """
    by_frame = (
        play_frames[["frameID", "event"]]
        .drop_duplicates(subset="frameID")
        .sort_values("frameID")
        .reset_index(drop=True)
    )
    if by_frame.empty:
        return None

    snap = _first_frame_with_event(by_frame["event"], by_frame["frameID"], SNAP_EVENTS)
    if snap is None:
        return None

    end_event = _first_frame_with_event(
        by_frame.loc[by_frame["frameID"] > snap, "event"],
        by_frame.loc[by_frame["frameID"] > snap, "frameID"],
        WINDOW_END_EVENTS,
    )

    start = snap + SNAP_LAG_FRAMES
    cap_end = snap + WINDOW_HARD_CAP_FRAMES
    end = cap_end if end_event is None else min(end_event, cap_end)

    if end - start < MIN_WINDOW_LENGTH:
        return None
    return start, end


def sample_window_frames(start: int, end: int, n: int) -> np.ndarray:
    """Evenly-spaced frame ids inside [start, end)."""
    end = max(end, start + 1)
    if n >= end - start:
        return np.arange(start, end)
    return np.linspace(start, end - 1, n).round().astype(int)
