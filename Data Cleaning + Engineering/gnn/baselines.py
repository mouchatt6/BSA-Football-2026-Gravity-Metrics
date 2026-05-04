"""Tabular baselines for expected attention.

Two are provided:
    fit_linear_baseline   ridge regression on play-level globals + rusher info
    fit_tree_baseline     gradient-boosted tree (sklearn HistGradientBoosting)

Both operate at the play-rusher level so they are directly comparable to the
GNN play-rusher predictions.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import position_group


PLAY_FEATURES = [
    "down", "yardsToGo", "quarter", "defendersInBox",
    "defenseDL", "defenseLB", "defenseDB",
    "offenseRB", "offenseTE", "offenseWR",
    "absoluteYardlineNumber",
]
CATEGORICAL_FEATURES = [
    "offenseFormation", "passCoverage", "passCoverageType", "dropBackType",
]


def build_baseline_table(
    play_attention: pd.DataFrame,
    play_context: pd.DataFrame,
    rusher_position_lookup: Dict[int, str],
) -> pd.DataFrame:
    df = play_attention.merge(play_context, on=["gameId", "playId"], how="inner")
    df["position_group"] = df["rusher_nflId"].map(rusher_position_lookup).fillna("Other")
    df["play_action"] = df["playAction"].astype(str).str.lower().eq("true").astype(int)
    return df


def _to_design(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    num = df[PLAY_FEATURES + ["play_action"]].fillna(0).to_numpy(dtype=float)
    cats = pd.get_dummies(df[CATEGORICAL_FEATURES + ["position_group"]].astype(str), dummy_na=True)
    X = np.concatenate([num, cats.to_numpy(dtype=float)], axis=1)
    cols = PLAY_FEATURES + ["play_action"] + list(cats.columns)
    return X, cols


def fit_linear_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame):
    from sklearn.linear_model import Ridge

    X_tr, cols = _to_design(train_df)
    y_tr = train_df["avg_attention_score"].to_numpy()
    X_val_raw, val_cols = _to_design(val_df)
    # align columns to train
    val_idx_map = {c: i for i, c in enumerate(val_cols)}
    X_val = np.zeros((X_val_raw.shape[0], len(cols)), dtype=float)
    for i, c in enumerate(cols):
        if c in val_idx_map:
            X_val[:, i] = X_val_raw[:, val_idx_map[c]]
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    return model, cols, val_pred


def fit_tree_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingRegressor

    X_tr, cols = _to_design(train_df)
    y_tr = train_df["avg_attention_score"].to_numpy()
    X_val_raw, val_cols = _to_design(val_df)
    val_idx_map = {c: i for i, c in enumerate(val_cols)}
    X_val = np.zeros((X_val_raw.shape[0], len(cols)), dtype=float)
    for i, c in enumerate(cols):
        if c in val_idx_map:
            X_val[:, i] = X_val_raw[:, val_idx_map[c]]
    model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=400)
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    return model, cols, val_pred
