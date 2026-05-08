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


def write_player_gravity(
    play_rusher_df: pd.DataFrame,
    rusher_position_lookup: Optional[Dict[int, str]] = None,
    min_plays: int = 25,
    output_path = GNN_PLAYER_GRAVITY_CSV,
) -> pd.DataFrame:
    df = play_rusher_df.copy()
    grouped = df.groupby(["rusher_nflId", "rusher_name"]).agg(
        plays=("playId", "count"),
        mean_actual=("actual_attention_gnn", "mean"),
        mean_expected=("expected_attention_gnn", "mean"),
        mean_gravity=("gravity_gnn", "mean"),
        std_gravity=("gravity_gnn", "std"),
    ).reset_index()

    qualified = grouped[grouped["plays"] >= min_plays].copy()
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
    min_plays: int = 25,
):
    pred_df = predict_play_rusher(model, data_list, device=device, batch_size=batch_size)
    play_rusher = write_play_rusher_predictions(pred_df, play_attention)
    player_df = write_player_gravity(play_rusher, rusher_position_lookup=rusher_position_lookup, min_plays=min_plays)
    return play_rusher, player_df
