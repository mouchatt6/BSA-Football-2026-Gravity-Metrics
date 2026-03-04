from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd


DROP_IMPACT_COLUMNS = [
    "pff_hit",
    "pff_hurry",
    "pff_sack",
    "pff_beatenByDefender",
    "pff_hitAllowed",
    "pff_hurryAllowed",
    "pff_sackAllowed",
]

PFF_KEEP_COLUMNS = [
    "gameId",
    "playId",
    "nflId",
    "pff_role",
    "pff_positionLinedUp",
    "pff_nflIdBlockedPlayer",
    "pff_blockType",
    "pff_backFieldBlock",
]

KEY_COLUMNS = ["gameId", "playId", "nflId"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(col).strip().strip('"') for col in df.columns]
    return df


def clean_object_values(df: pd.DataFrame) -> pd.DataFrame:
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()
    df = df.replace({"NA": pd.NA, "": pd.NA})
    return df


def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True, keep_default_na=False, low_memory=False)
    df = normalize_columns(df)
    df = clean_object_values(df)
    for key in KEY_COLUMNS:
        if key in df.columns:
            df[key] = pd.to_numeric(df[key], errors="coerce").astype("Int64")
    if "gameId" in df.columns:
        df = df[df["gameId"].notna()].copy()
    return df


def normalize_play_action(df: pd.DataFrame) -> pd.DataFrame:
    if "playAction" not in df.columns:
        return df
    mapper = {
        "TRUE": True,
        "FALSE": False,
        "True": True,
        "False": False,
        "1": True,
        "0": False,
    }
    df["playAction"] = df["playAction"].map(mapper).astype("boolean")
    return df


def add_play_filter_flags(df: pd.DataFrame) -> pd.DataFrame:
    description = df["playDescription"].fillna("").str.lower() if "playDescription" in df.columns else ""
    dropback = df["dropBackType"].fillna("").str.lower() if "dropBackType" in df.columns else ""
    df["is_screen"] = description.str.contains(r"\bscreen\b", regex=True)
    df["is_rpo"] = description.str.contains(r"\brpo\b", regex=True) | dropback.str.contains("rpo")
    df["is_qb_spike"] = description.str.contains(r"\bspike\b", regex=True)
    # This requires time-to-throw and ball-location context not available in current base tables.
    df["ttt_le_1_5_and_behind_los"] = pd.NA
    df["is_gravity_candidate_base"] = ~(df["is_screen"] | df["is_rpo"] | df["is_qb_spike"])
    return df


def save_cleaned_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def build_pipeline(work_dir: Path) -> Dict[str, int | str | bool]:
    source_dir = work_dir / "source_copies"
    cleaned_dir = work_dir / "cleaned_csv"
    outputs_dir = work_dir / "outputs"
    outputs_csv_dir = work_dir / "outputs_csv"
    scripts_dir = work_dir / "scripts"

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_csv_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    games = read_csv_clean(source_dir / "games.csv")
    plays_cleaned = read_csv_clean(source_dir / "plays_cleaned.csv")
    players = read_csv_clean(source_dir / "players.csv")
    pff = read_csv_clean(source_dir / "pffScoutingData.csv")

    plays_cleaned = normalize_play_action(plays_cleaned)
    plays_cleaned = add_play_filter_flags(plays_cleaned)

    games_keep = [
        "gameId",
        "season",
        "week",
        "gameDate",
        "gameTimeEastern",
        "homeTeamAbbr",
        "visitorTeamAbbr",
    ]
    games_keep = [c for c in games_keep if c in games.columns]
    games = games[games_keep].copy()

    play_context = plays_cleaned.merge(games, on="gameId", how="left", validate="many_to_one")

    pff = pff.drop(columns=[c for c in DROP_IMPACT_COLUMNS if c in pff.columns])
    pff = pff[[c for c in PFF_KEEP_COLUMNS if c in pff.columns]].copy()
    if "pff_nflIdBlockedPlayer" in pff.columns:
        pff["pff_nflIdBlockedPlayer"] = pd.to_numeric(
            pff["pff_nflIdBlockedPlayer"], errors="coerce"
        ).astype("Int64")

    play_player_roles = play_context.merge(pff, on=["gameId", "playId"], how="left", validate="one_to_many")
    play_player_roles = play_player_roles.merge(players, on="nflId", how="left", validate="many_to_one")

    for name, frame in {
        "games_cleaned.csv": games,
        "plays_cleaned_cleaned.csv": plays_cleaned,
        "players_cleaned.csv": players,
        "pffScoutingData_cleaned.csv": pff,
    }.items():
        save_cleaned_csv(frame, cleaned_dir / name)

    play_context.to_csv(outputs_csv_dir / "play_context.csv", index=False)
    gravity_base = play_player_roles.copy()

    gravity_base.to_csv(outputs_csv_dir / "gravity_base.csv", index=False)

    summary: Dict[str, int | str | bool] = {
        "work_dir": str(work_dir),
        "source_dir": str(source_dir),
        "cleaned_dir": str(cleaned_dir),
        "outputs_dir": str(outputs_dir),
        "outputs_csv_dir": str(outputs_csv_dir),
        "rows_play_context": int(len(play_context)),
        "rows_play_player_roles": int(len(play_player_roles)),
        "rows_gravity_base": int(len(gravity_base)),
    }

    with open(outputs_dir / "pipeline_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


def resolve_repo_root() -> Path:
    if "__file__" in globals():
        script_root = Path(__file__).resolve().parents[2]
        if (script_root / "Data Cleaning + Engineering").exists():
            return script_root

    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "Data Cleaning + Engineering").exists() and (candidate / "games.csv").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate repository root. Run the notebook from inside the repo tree."
    )


def main() -> None:
    repo_root = resolve_repo_root()
    work_dir = repo_root / "Data Cleaning + Engineering"
    summary = build_pipeline(work_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
