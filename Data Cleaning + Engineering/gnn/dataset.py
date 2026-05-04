"""PyTorch-Geometric dataset wrapping the frame-graph builder.

The dataset materializes graphs once to a cache directory; subsequent loads
read directly from the cache. This is important because gravity_base.csv is
~2 GB.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    CACHE_DIR,
    EACH_RUSHER_CSV,
    FRAMES_PER_PLAY,
    GRAVITY_BASE_CSV,
    PLAY_ATTENTION_CSV,
    PLAY_CONTEXT_CSV,
)
from .data_filters import filter_eligible_plays
from .graph_builder import (
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    FrameGraph,
    attention_lookup,
    build_play_graphs,
    to_pyg_data,
)


GRAVITY_BASE_USECOLS = [
    "gameId", "playId", "frameID", "nflId",
    "x", "y", "s", "a", "o", "dir",
    "officialPosition", "displayName", "height", "weight",
    "pff_role", "pff_positionLinedUp", "pff_nflIdBlockedPlayer",
    "possessionTeam", "defensiveTeam", "event",
]


def _load_play_context() -> pd.DataFrame:
    df = pd.read_csv(PLAY_CONTEXT_CSV)
    df["gameId"] = df["gameId"].astype(int)
    df["playId"] = df["playId"].astype(int)
    return df


def _load_play_attention() -> pd.DataFrame:
    df = pd.read_csv(PLAY_ATTENTION_CSV)
    df["gameId"] = df["gameId"].astype(int)
    df["playId"] = df["playId"].astype(int)
    df["rusher_nflId"] = df["rusher_nflId"].astype(int)
    return df


def _frames_for_eligible(
    eligible_keys: set,
    chunksize: int = 500_000,
) -> Iterable[pd.DataFrame]:
    """Stream gravity_base.csv in chunks, keeping only eligible plays."""
    for chunk in pd.read_csv(
        GRAVITY_BASE_CSV,
        usecols=GRAVITY_BASE_USECOLS,
        chunksize=chunksize,
    ):
        chunk["gameId"] = chunk["gameId"].astype(int)
        chunk["playId"] = chunk["playId"].astype(int)
        keys = list(zip(chunk["gameId"], chunk["playId"]))
        chunk = chunk.loc[[k in eligible_keys for k in keys]]
        if not chunk.empty:
            yield chunk


def build_frame_graphs(
    play_context: Optional[pd.DataFrame] = None,
    play_attention: Optional[pd.DataFrame] = None,
    frames_per_play: int = FRAMES_PER_PLAY,
    max_plays: Optional[int] = None,
    chunksize: int = 500_000,
    verbose: bool = True,
) -> List[FrameGraph]:
    """Materialize FrameGraphs for every eligible play with attention labels."""
    play_context = _load_play_context() if play_context is None else play_context
    play_attention = _load_play_attention() if play_attention is None else play_attention

    eligible = filter_eligible_plays(play_context)
    eligible_keys = set(zip(eligible["gameId"].astype(int), eligible["playId"].astype(int)))

    targets = attention_lookup(play_attention)
    play_keys_with_targets = set(targets.keys())
    eligible_keys = eligible_keys & play_keys_with_targets
    if max_plays is not None:
        eligible_keys = set(list(eligible_keys)[:max_plays])

    play_ctx_index = eligible.set_index(["gameId", "playId"])

    # We must keep all frames for an eligible play together; group on a per-chunk
    # basis and collect partial slices, then process once each play is complete.
    play_buffers: dict[Tuple[int, int], List[pd.DataFrame]] = {k: [] for k in eligible_keys}
    finalized: set = set()
    graphs: List[FrameGraph] = []

    def finalize(key: Tuple[int, int]) -> None:
        parts = play_buffers.pop(key, [])
        if not parts:
            return
        play_df = pd.concat(parts, axis=0, ignore_index=True)
        try:
            row = play_ctx_index.loc[key]
        except KeyError:
            return
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        row = row.copy()
        row["gameId"] = key[0]
        row["playId"] = key[1]
        graphs.extend(build_play_graphs(
            play_df=play_df,
            play_context_row=row,
            targets_for_play=targets.get(key, {}),
            frames_per_play=frames_per_play,
        ))
        finalized.add(key)

    seen_keys_in_chunk: set = set()
    for chunk in _frames_for_eligible(eligible_keys, chunksize=chunksize):
        keys_in_chunk = set(zip(chunk["gameId"], chunk["playId"]))
        # Any previously-buffered key NOT in this chunk is "complete" assuming
        # plays don't span discontinuous chunks; gravity_base is sorted by
        # (gameId, playId, frameID), so once we leave a play we're done with it.
        for prev_key in seen_keys_in_chunk - keys_in_chunk:
            if prev_key not in finalized:
                finalize(prev_key)
        seen_keys_in_chunk = keys_in_chunk

        for key, sub in chunk.groupby(["gameId", "playId"], sort=False):
            key_t = (int(key[0]), int(key[1]))
            if key_t in finalized:
                continue
            play_buffers.setdefault(key_t, []).append(sub)

        if verbose:
            done = len(finalized)
            print(f"  ... {done} plays processed, {len(graphs)} graphs", flush=True)

    # Finalize remaining buffered plays
    for key in list(play_buffers.keys()):
        if key not in finalized:
            finalize(key)

    if verbose:
        print(f"Built {len(graphs)} graphs over {len(finalized)} plays.")
    return graphs


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def save_frame_graphs(graphs: List[FrameGraph], path: Path) -> None:
    """Persist a list of FrameGraphs to disk via numpy savez per-graph."""
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_frame_graphs(path: Path) -> List[FrameGraph]:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def to_pyg_dataset(graphs: List[FrameGraph]):
    """Convert all FrameGraphs to a list of torch_geometric Data objects."""
    return [to_pyg_data(g) for g in graphs]


# Sizes propagated for downstream model wiring
__all__ = [
    "build_frame_graphs",
    "save_frame_graphs",
    "load_frame_graphs",
    "to_pyg_dataset",
    "cache_path",
    "NODE_FEATURE_DIM",
    "EDGE_FEATURE_DIM",
    "GLOBAL_FEATURE_DIM",
]
