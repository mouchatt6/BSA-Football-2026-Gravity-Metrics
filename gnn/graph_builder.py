"""Build PyTorch-Geometric graphs from gravity_base frames.

One graph per (gameId, playId, frameID). Nodes are the players present in the
frame (typically 22). Edges are fully connected with edge features describing
spatial relationships. The play-level globals and the per-rusher attention
target are attached so that downstream models can read them off the Data
object.

The resulting Data has:
    x:               [N, NODE_FEATURE_DIM]
    edge_index:      [2, N*(N-1)]
    edge_attr:       [N*(N-1), EDGE_FEATURE_DIM]
    u:               [1, GLOBAL_FEATURE_DIM]              (play globals)
    rusher_mask:     [N] bool                              (which nodes are rushers)
    y:               [N] float                             (per-rusher attention; 0 elsewhere)
    nfl_id:          [N] long                              (player nflId)
    game_id, play_id, frame_id (python ints stored as Data attributes)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_filters import find_pass_rush_window, sample_window_frames
from .features import (
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    edge_feature_vector,
    global_feature_vector,
    node_feature_vector,
)


@dataclass
class FrameGraph:
    """Light, framework-agnostic representation produced by the builder."""
    x: np.ndarray            # [N, F]
    edge_index: np.ndarray   # [2, E]
    edge_attr: np.ndarray    # [E, EF]
    u: np.ndarray            # [GF]
    rusher_mask: np.ndarray  # [N] bool
    y: np.ndarray            # [N] float (target attention, 0 outside rusher_mask)
    nfl_ids: np.ndarray      # [N] int
    game_id: int
    play_id: int
    frame_id: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _team_label(row: pd.Series) -> str:
    """Return a team string for the player (offense/defense based on possessionTeam)."""
    pos_team = row.get("possessionTeam")
    def_team = row.get("defensiveTeam")
    if str(row.get("officialPosition")) in {
        "QB", "RB", "FB", "WR", "TE", "T", "G", "C",
    }:
        return str(pos_team)
    return str(def_team)


def _enrich_team_column(frame_df: pd.DataFrame) -> pd.DataFrame:
    """Add a `team` column derived from official position so that team-equality
    checks in edge features have a stable value."""
    df = frame_df.copy()
    df["team"] = df.apply(_team_label, axis=1)
    return df


def _build_fully_connected_edges(n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64)
    src = np.repeat(np.arange(n), n)
    dst = np.tile(np.arange(n), n)
    mask = src != dst
    return np.stack([src[mask], dst[mask]], axis=0).astype(np.int64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attention_lookup(play_attention: pd.DataFrame) -> Dict[Tuple[int, int], Dict[int, float]]:
    """{(gameId, playId): {rusher_nflId: avg_attention_score}}."""
    grouped: Dict[Tuple[int, int], Dict[int, float]] = {}
    for (gid, pid), sub in play_attention.groupby(["gameId", "playId"], sort=False):
        grouped[(int(gid), int(pid))] = dict(zip(sub["rusher_nflId"].astype(int), sub["avg_attention_score"].astype(float)))
    return grouped


def iter_play_frames(
    gravity_base_iter: Iterable[pd.DataFrame],
    eligible_keys: set,
) -> Iterable[Tuple[Tuple[int, int], pd.DataFrame]]:
    """Yield (key, play_df) groups limited to eligible plays.

    Accepts either the full frame as one DataFrame or chunks; both are yielded
    grouped by (gameId, playId).
    """
    for chunk in gravity_base_iter:
        chunk = chunk[chunk[["gameId", "playId"]].apply(tuple, axis=1).isin(eligible_keys)]
        if chunk.empty:
            continue
        for key, sub in chunk.groupby(["gameId", "playId"], sort=False):
            yield (int(key[0]), int(key[1])), sub


def build_play_graphs(
    play_df: pd.DataFrame,
    play_context_row: pd.Series,
    targets_for_play: Dict[int, float],
    frames_per_play: int,
) -> List[FrameGraph]:
    """Build evenly-spaced FrameGraph objects for a single play."""
    window = find_pass_rush_window(play_df)
    if window is None:
        return []
    start, end = window
    frame_ids = sample_window_frames(start, end, frames_per_play)

    out: List[FrameGraph] = []
    play_df = play_df[play_df["frameID"].isin(frame_ids)]
    if play_df.empty:
        return []
    play_df = _enrich_team_column(play_df)

    num_rushers = sum(
        1 for v in (
            play_df.drop_duplicates("nflId").set_index("nflId")["pff_role"].astype(str) == "Pass Rush"
        ).tolist() if v
    )
    u = global_feature_vector(play_context_row, num_rushers=num_rushers)

    for fid in frame_ids:
        frame_df = play_df[play_df["frameID"] == fid]
        if frame_df.empty:
            continue
        frame_df = frame_df.drop_duplicates(subset="nflId").reset_index(drop=True)
        n = len(frame_df)
        if n < 4:  # not enough players present
            continue

        x_center = float(frame_df["x"].mean())
        y_center = float(frame_df["y"].mean())

        x_arr = np.stack(
            [node_feature_vector(row, x_center, y_center) for _, row in frame_df.iterrows()],
            axis=0,
        )
        edge_index = _build_fully_connected_edges(n)

        edge_attr = np.zeros((edge_index.shape[1], EDGE_FEATURE_DIM), dtype=np.float32)
        if edge_index.shape[1] > 0:
            x_vals = frame_df["x"].astype(float).to_numpy()
            y_vals = frame_df["y"].astype(float).to_numpy()
            for k in range(edge_index.shape[1]):
                i, j = edge_index[0, k], edge_index[1, k]
                dx = float(x_vals[i] - x_vals[j])
                dy = float(y_vals[i] - y_vals[j])
                dist = float(np.hypot(dx, dy))
                edge_attr[k] = edge_feature_vector(frame_df.iloc[i], frame_df.iloc[j], dx, dy, dist)

        nfl_ids = frame_df["nflId"].astype(int).to_numpy()
        rusher_mask = np.zeros(n, dtype=bool)
        y_arr = np.zeros(n, dtype=np.float32)
        for idx, nid in enumerate(nfl_ids):
            if int(nid) in targets_for_play:
                rusher_mask[idx] = True
                y_arr[idx] = float(targets_for_play[int(nid)])

        out.append(FrameGraph(
            x=x_arr.astype(np.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            u=u.astype(np.float32),
            rusher_mask=rusher_mask,
            y=y_arr,
            nfl_ids=nfl_ids,
            game_id=int(play_context_row["gameId"]),
            play_id=int(play_context_row["playId"]),
            frame_id=int(fid),
        ))
    return out


def to_pyg_data(fg: FrameGraph):
    """Convert a FrameGraph to torch_geometric.data.Data. Imported lazily so
    that the rest of the module can be used without torch installed."""
    import torch
    from torch_geometric.data import Data

    data = Data(
        x=torch.from_numpy(fg.x),
        edge_index=torch.from_numpy(fg.edge_index),
        edge_attr=torch.from_numpy(fg.edge_attr),
        y=torch.from_numpy(fg.y),
    )
    data.u = torch.from_numpy(fg.u).unsqueeze(0)  # [1, GF]
    data.rusher_mask = torch.from_numpy(fg.rusher_mask)
    data.nfl_id = torch.from_numpy(fg.nfl_ids).long()
    data.game_id = fg.game_id
    data.play_id = fg.play_id
    data.frame_id = fg.frame_id
    return data


# Re-export feature dims so callers don't need to know about features.py
__all__ = [
    "FrameGraph",
    "attention_lookup",
    "build_play_graphs",
    "to_pyg_data",
    "NODE_FEATURE_DIM",
    "EDGE_FEATURE_DIM",
    "GLOBAL_FEATURE_DIM",
]
