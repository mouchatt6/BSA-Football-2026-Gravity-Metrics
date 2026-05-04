"""Feature encoders shared by graph builder and tabular baselines."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .config import FEATURE_SPEC


def _vocab_index(values: Sequence[str]) -> Dict[str, int]:
    return {v: i for i, v in enumerate(values)}


FORMATION_IDX = _vocab_index(FEATURE_SPEC.formations)
COVERAGE_IDX = _vocab_index(FEATURE_SPEC.coverages)
COVERAGE_TYPE_IDX = _vocab_index(FEATURE_SPEC.coverage_types)
DROPBACK_IDX = _vocab_index(FEATURE_SPEC.dropback_types)
PFF_ROLE_IDX = _vocab_index(FEATURE_SPEC.pff_roles)
PFF_POSITION_IDX = _vocab_index(FEATURE_SPEC.pff_positions)
OFFICIAL_POS_IDX = _vocab_index(FEATURE_SPEC.official_positions)


def one_hot(value, lookup: Dict[str, int]) -> np.ndarray:
    out = np.zeros(len(lookup), dtype=np.float32)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        out[lookup["UNKNOWN"]] = 1.0
        return out
    key = str(value).strip()
    if key in lookup:
        out[lookup[key]] = 1.0
    else:
        out[lookup["UNKNOWN"]] = 1.0
    return out


def parse_height_inches(height: object) -> float:
    if height is None:
        return 0.0
    s = str(height).strip()
    if not s or s.lower() == "nan":
        return 0.0
    if "-" in s:
        try:
            ft, inch = s.split("-")
            return float(ft) * 12 + float(inch)
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def global_feature_vector(play_row: pd.Series, num_rushers: int) -> np.ndarray:
    """Play-level globals concatenated for graph readout."""
    feats: List[np.ndarray] = []
    feats.append(one_hot(play_row.get("offenseFormation"), FORMATION_IDX))
    feats.append(one_hot(play_row.get("passCoverage"), COVERAGE_IDX))
    feats.append(one_hot(play_row.get("passCoverageType"), COVERAGE_TYPE_IDX))
    feats.append(one_hot(play_row.get("dropBackType"), DROPBACK_IDX))
    play_action = 1.0 if str(play_row.get("playAction")).lower() == "true" else 0.0
    score_diff = safe_float(play_row.get("preSnapHomeScore")) - safe_float(play_row.get("preSnapVisitorScore"))
    abs_yardline = safe_float(play_row.get("absoluteYardlineNumber"), 50.0)
    feats.append(np.array([
        safe_float(play_row.get("quarter"), 1.0),
        safe_float(play_row.get("down"), 1.0),
        safe_float(play_row.get("yardsToGo"), 10.0) / 10.0,
        safe_float(play_row.get("defendersInBox"), 6.0) / 11.0,
        safe_float(play_row.get("defenseDL"), 4.0) / 11.0,
        safe_float(play_row.get("defenseLB"), 3.0) / 11.0,
        safe_float(play_row.get("defenseDB"), 4.0) / 11.0,
        safe_float(play_row.get("offenseRB"), 1.0) / 4.0,
        safe_float(play_row.get("offenseTE"), 1.0) / 4.0,
        safe_float(play_row.get("offenseWR"), 3.0) / 5.0,
        play_action,
        score_diff / 21.0,
        abs_yardline / 100.0,
        float(num_rushers) / 8.0,
    ], dtype=np.float32))
    return np.concatenate(feats, axis=0)


def node_feature_vector(player_row: pd.Series, x_center: float, y_center: float) -> np.ndarray:
    """Per-player frame features. Coordinates are centered on the play centroid."""
    is_offense = 1.0 if player_row.get("possessionTeam") == player_row.get("team") else 0.0
    pff_role = str(player_row.get("pff_role", "UNKNOWN"))
    is_rusher = 1.0 if pff_role == "Pass Rush" else 0.0
    is_blocker = 1.0 if pff_role == "Pass Block" else 0.0
    is_qb = 1.0 if str(player_row.get("officialPosition")) == "QB" else 0.0

    raw_x = safe_float(player_row.get("x"))
    raw_y = safe_float(player_row.get("y"))
    feats = np.array([
        raw_x - x_center,
        raw_y - y_center,
        safe_float(player_row.get("s")),
        safe_float(player_row.get("a")),
        np.sin(np.deg2rad(safe_float(player_row.get("o")))),
        np.cos(np.deg2rad(safe_float(player_row.get("o")))),
        np.sin(np.deg2rad(safe_float(player_row.get("dir")))),
        np.cos(np.deg2rad(safe_float(player_row.get("dir")))),
        is_offense,
        is_rusher,
        is_blocker,
        is_qb,
        parse_height_inches(player_row.get("height")) / 80.0,
        safe_float(player_row.get("weight"), 220.0) / 350.0,
    ], dtype=np.float32)

    cat = np.concatenate([
        one_hot(player_row.get("officialPosition"), OFFICIAL_POS_IDX),
        one_hot(player_row.get("pff_role"), PFF_ROLE_IDX),
    ], axis=0)
    return np.concatenate([feats, cat], axis=0)


def edge_feature_vector(
    p_i: pd.Series, p_j: pd.Series, dx: float, dy: float, dist: float
) -> np.ndarray:
    same_team = 1.0 if p_i.get("team") == p_j.get("team") else 0.0
    off_def_pair = 1.0 if same_team == 0.0 else 0.0
    role_i = str(p_i.get("pff_role", ""))
    role_j = str(p_j.get("pff_role", ""))
    blocker_rusher_candidate = 1.0 if (
        (role_i == "Pass Block" and role_j == "Pass Rush")
        or (role_j == "Pass Block" and role_i == "Pass Rush")
    ) else 0.0
    o_diff = safe_float(p_i.get("o")) - safe_float(p_j.get("o"))
    return np.array([
        dx,
        dy,
        dist,
        np.exp(-dist / 3.0),
        same_team,
        off_def_pair,
        blocker_rusher_candidate,
        np.sin(np.deg2rad(o_diff)),
        np.cos(np.deg2rad(o_diff)),
    ], dtype=np.float32)


# Sizes used by model definitions. Keep these in sync with the encoders above.
NODE_FEATURE_DIM: int = 14 + len(FEATURE_SPEC.official_positions) + len(FEATURE_SPEC.pff_roles)
EDGE_FEATURE_DIM: int = 9
GLOBAL_FEATURE_DIM: int = (
    len(FEATURE_SPEC.formations)
    + len(FEATURE_SPEC.coverages)
    + len(FEATURE_SPEC.coverage_types)
    + len(FEATURE_SPEC.dropback_types)
    + 14
)
