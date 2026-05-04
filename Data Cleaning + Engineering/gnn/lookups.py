"""Helper lookups for player/position metadata used throughout the pipeline."""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .config import PLAYERS_CSV, position_group


def rusher_position_group_lookup(players_csv = PLAYERS_CSV) -> Dict[int, str]:
    """{nflId: position_group} from the cleaned players file."""
    df = pd.read_csv(players_csv)
    if "nflId" not in df.columns:
        raise ValueError("players_cleaned.csv missing nflId column")
    return {
        int(row["nflId"]): position_group(row.get("officialPosition") or row.get("Position") or "")
        for _, row in df.iterrows()
        if pd.notna(row["nflId"])
    }


def player_name_lookup(players_csv = PLAYERS_CSV) -> Dict[int, str]:
    df = pd.read_csv(players_csv)
    name_col = "displayName" if "displayName" in df.columns else df.columns[1]
    return {int(row["nflId"]): str(row[name_col]) for _, row in df.iterrows() if pd.notna(row["nflId"])}
