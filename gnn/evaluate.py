"""Evaluation utilities for the expected-attention GNN.

Operates on the play-rusher level: for each (gameId, playId, rusher_nflId)
average frame-level predictions and compare to the observed
`avg_attention_score` from play_attention_scores.csv.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import pick_device


def predict_play_rusher(
    model,
    data_list: Sequence,
    device: str = "auto",
    batch_size: int = 256,
) -> pd.DataFrame:
    """Run the model on every graph and aggregate to play-rusher level.

    Returns DataFrame with columns:
        gameId, playId, rusher_nflId, expected_attention_gnn,
        actual_attention_gnn (= mean of y target across frames; equal to play
        target by construction).
    """
    import torch
    from torch_geometric.loader import DataLoader

    resolved = pick_device(device)
    model = model.to(resolved)
    model.eval()
    loader = DataLoader(list(data_list), batch_size=batch_size, shuffle=False)

    bucket: Dict[Tuple[int, int, int], List[float]] = defaultdict(list)
    target_bucket: Dict[Tuple[int, int, int], List[float]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(resolved)
            pred = model(batch).detach().cpu().numpy()
            mask = batch.rusher_mask.detach().cpu().numpy().astype(bool)
            y = batch.y.detach().cpu().numpy()
            nfl_id = batch.nfl_id.detach().cpu().numpy()
            graph_idx = batch.batch.detach().cpu().numpy()

            game_ids = batch.game_id if isinstance(batch.game_id, list) else [batch.game_id]
            play_ids = batch.play_id if isinstance(batch.play_id, list) else [batch.play_id]

            for n_idx in np.where(mask)[0]:
                g = int(graph_idx[n_idx])
                key = (int(game_ids[g]), int(play_ids[g]), int(nfl_id[n_idx]))
                bucket[key].append(float(pred[n_idx]))
                target_bucket[key].append(float(y[n_idx]))

    rows = []
    for (gid, pid, nid), preds in bucket.items():
        targets = target_bucket[(gid, pid, nid)]
        rows.append({
            "gameId": gid,
            "playId": pid,
            "rusher_nflId": nid,
            "expected_attention_gnn": float(np.mean(preds)),
            "actual_attention_gnn": float(np.mean(targets)),
            "n_frames": len(preds),
        })
    return pd.DataFrame(rows)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    ss_res = float(np.sum(diff * diff))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) if len(y_true) else 1.0
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    if len(y_true) >= 2:
        spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
        pearson = float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson"))
    else:
        spearman = float("nan")
        pearson = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": spearman, "pearson": pearson, "n": int(len(y_true))}


def metrics_by_position_group(
    pred_df: pd.DataFrame,
    rusher_position_lookup: Dict[int, str],
) -> pd.DataFrame:
    df = pred_df.copy()
    df["position_group"] = df["rusher_nflId"].map(rusher_position_lookup).fillna("Other")
    rows = []
    for group, sub in df.groupby("position_group"):
        m = regression_metrics(
            sub["actual_attention_gnn"].to_numpy(),
            sub["expected_attention_gnn"].to_numpy(),
        )
        m["position_group"] = group
        rows.append(m)
    return pd.DataFrame(rows).set_index("position_group")


def split_half_stability(
    pred_df: pd.DataFrame,
    seed: int = 0,
    min_plays: int = 8,
) -> float:
    """Per-player split-half correlation of gravity (actual - expected).

    Returns Spearman rho between player means on two random halves; mirrors
    the STRAIN paper's stability check.
    """
    rng = np.random.RandomState(seed)
    df = pred_df.copy()
    df["gravity_gnn"] = df["actual_attention_gnn"] - df["expected_attention_gnn"]
    df["half"] = rng.randint(0, 2, size=len(df))

    half_means = df.groupby(["rusher_nflId", "half"])["gravity_gnn"].mean().unstack(fill_value=np.nan)
    counts = df.groupby(["rusher_nflId", "half"]).size().unstack(fill_value=0)
    keep = (counts >= (min_plays // 2)).all(axis=1)
    half_means = half_means.loc[keep].dropna()
    if len(half_means) < 5:
        return float("nan")
    return float(half_means[0].corr(half_means[1], method="spearman"))
