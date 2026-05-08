"""Inference & output writing.

Produces:
    outputs_csv/gnn_attention_scores.csv   (play-rusher level)
    outputs_csv/gnn_player_gravity.csv     (player-level rollup with z-scores)
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    GNN_ATTENTION_CSV,
    GNN_PLAYER_GRAVITY_CSV,
    PLAYER_ATTENTION_CSV,
)
from .evaluate import predict_play_rusher


def _player_name_lookup(play_attention: pd.DataFrame) -> Dict[int, str]:
    return dict(zip(play_attention["rusher_nflId"].astype(int), play_attention["rusher_name"].astype(str)))


def write_play_rusher_predictions(
    pred_df: pd.DataFrame,
    play_attention: pd.DataFrame,
    output_path = GNN_ATTENTION_CSV,
) -> pd.DataFrame:
    df = pred_df.copy()
    df["gravity_gnn"] = df["actual_attention_gnn"] - df["expected_attention_gnn"]
    name_lookup = _player_name_lookup(play_attention)
    df["rusher_name"] = df["rusher_nflId"].map(name_lookup)
    cols = [
        "gameId", "playId", "rusher_nflId", "rusher_name",
        "actual_attention_gnn", "expected_attention_gnn", "gravity_gnn", "n_frames",
    ]
    df = df[cols].sort_values(["gameId", "playId", "rusher_nflId"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df


def _auto_min_plays(plays_per_rusher: pd.Series) -> int:
    """Pick a sensible min_plays threshold based on the empirical distribution.

    Strategy: aim to retain roughly the top ~75% of rushers by play count,
    bounded between [3, 25]. With the full ~7.4K play dataset this lands near
    25; with a small subsample (e.g. MAX_PLAYS=400) it adapts down to ~3-5.
    """
    if plays_per_rusher.empty:
        return 1
    p25 = float(plays_per_rusher.quantile(0.25))
    return int(max(3, min(25, round(p25))))


def write_player_gravity(
    play_rusher_df: pd.DataFrame,
    rusher_position_lookup: Optional[Dict[int, str]] = None,
    min_plays = "auto",
    output_path = GNN_PLAYER_GRAVITY_CSV,
    verbose: bool = True,
) -> pd.DataFrame:
    df = play_rusher_df.copy()
    grouped = df.groupby(["rusher_nflId", "rusher_name"]).agg(
        plays=("playId", "count"),
        mean_actual=("actual_attention_gnn", "mean"),
        mean_expected=("expected_attention_gnn", "mean"),
        mean_gravity=("gravity_gnn", "mean"),
        std_gravity=("gravity_gnn", "std"),
    ).reset_index()

    if min_plays == "auto":
        threshold = _auto_min_plays(grouped["plays"])
    else:
        threshold = int(min_plays)

    qualified = grouped[grouped["plays"] >= threshold].copy()

    # Adaptive fallback: if the requested threshold filters everyone, lower it
    # (with a one-line note) so callers always get a non-empty rollup when at
    # least one rusher exists.
    if qualified.empty and not grouped.empty:
        new_threshold = max(1, int(grouped["plays"].max()))
        if verbose:
            print(f"[write_player_gravity] min_plays={threshold} produced 0 qualified rushers; "
                  f"falling back to min_plays={new_threshold} (the max plays/rusher in the sample). "
                  f"Increase MAX_PLAYS or pass an explicit min_plays for a stricter cut.")
        threshold = new_threshold
        qualified = grouped[grouped["plays"] >= threshold].copy()

    if verbose:
        print(f"[write_player_gravity] {len(grouped)} unique rushers, "
              f"plays/rusher: median={grouped['plays'].median():.0f} "
              f"max={grouped['plays'].max()} min={grouped['plays'].min()} | "
              f"min_plays={threshold} -> {len(qualified)} qualified")

    if not qualified.empty:
        mu = qualified["mean_gravity"].mean()
        sd = qualified["mean_gravity"].std(ddof=0)
        qualified["gravity_z"] = (qualified["mean_gravity"] - mu) / (sd if sd > 0 else 1.0)
        qualified["gravity_pct"] = qualified["mean_gravity"].rank(pct=True) * 100.0
    else:
        qualified["gravity_z"] = np.nan
        qualified["gravity_pct"] = np.nan

    if rusher_position_lookup is not None:
        qualified["position_group"] = qualified["rusher_nflId"].map(rusher_position_lookup).fillna("Other")

    qualified = qualified.sort_values("mean_gravity", ascending=False).reset_index(drop=True)
    qualified.to_csv(output_path, index=False)
    return qualified


def run_inference_and_write(
    model,
    data_list: Sequence,
    play_attention: pd.DataFrame,
    rusher_position_lookup: Optional[Dict[int, str]] = None,
    device: str = "auto",
    batch_size: int = 256,
    min_plays = "auto",
):
    pred_df = predict_play_rusher(model, data_list, device=device, batch_size=batch_size)
    play_rusher = write_play_rusher_predictions(pred_df, play_attention)
    player_df = write_player_gravity(play_rusher, rusher_position_lookup=rusher_position_lookup, min_plays=min_plays)
    return play_rusher, player_df
