from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

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
TRACKING_JOIN_COLUMNS = ["gameId", "playId", "nflId", "frameId"]
TRACKING_BASE_COLUMNS = [
    "gameId",
    "playId",
    "nflId",
    "frameId",
    "time",
    "jerseyNumber",
    "team",
    "playDirection",
    "x",
    "y",
    "s",
    "a",
    "dis",
    "o",
    "dir",
    "event",
]
TRACKING_EXTRACT_COLUMNS = [
    "gameId",
    "playId",
    "frameId",
    "time",
    "jerseyNumber",
    "playDirection",
    "x",
    "y",
    "s",
    "a",
    "dis",
    "o",
    "dir",
    "event",
]
TRACKING_EXTRACT_OUTPUT_RENAMES = {
    "gameId": "gameID",
    "playId": "playID",
    "frameId": "frameID",
}
TRACKING_MERGE_COLUMNS = ["gameId", "playId", "nflId"] + [c for c in TRACKING_EXTRACT_COLUMNS if c not in {"gameId", "playId"}]
GRAVITY_DROP_COLUMNS = [
    "playDescription",
    "collegeName",
    "gameTimeEastern",
    "birthDate",
    "homeTeamAbbr",
    "visitorTeamAbbr",
]
GRAVITY_RENAME_COLUMNS = {
    "time": "play time",
    "frameId": "frameID",
}
GRAVITY_LEFT_PRIORITY_COLUMNS = [
    "gameId",
    "playId",
    "season",
    "week",
    "gameDate",
    "quarter",
    "down",
    "yardsToGo",
    "gameClock",
    "play time",
    "frameID",
    "possessionTeam",
    "defensiveTeam",
    "yardlineSide",
    "yardlineNumber",
    "absoluteYardlineNumber",
    "offenseFormation",
    "offenseRB",
    "offenseTE",
    "offenseWR",
    "defendersInBox",
    "defenseDL",
    "defenseLB",
    "defenseDB",
    "dropBackType",
    "playAction",
    "passCoverage",
    "passCoverageType",
    "is_screen",
    "is_rpo",
    "is_qb_spike",
    "ttt_le_1_5_and_behind_los",
    "is_gravity_candidate_base",
]

NA_TOKENS = {"NA", "N/A", "None", "null", "NULL", "nan", "NaN", ""}
INT_COLUMNS = ["gameId", "playId", "nflId", "frameId", "jerseyNumber", "season", "week"]
FLOAT_COLUMNS = ["x", "y", "s", "a", "dis", "o", "dir", "yardlineNumber", "absoluteYardlineNumber"]
CANONICAL_RENAMES = {
    "gameid": "gameId",
    "playid": "playId",
    "nflid": "nflId",
    "frameid": "frameId",
    "jerseynumber": "jerseyNumber",
    "playdirection": "playDirection",
    "gamedate": "gameDate",
    "gametimeeastern": "gameTimeEastern",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_columns: List[str] = []
    for col in df.columns:
        normalized = str(col).strip().strip('"')
        canonical_key = normalized.lower()
        cleaned_columns.append(CANONICAL_RENAMES.get(canonical_key, normalized))
    df.columns = cleaned_columns
    return df


def clean_object_values(df: pd.DataFrame) -> pd.DataFrame:
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(list(NA_TOKENS), pd.NA)
    return df


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in INT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True, keep_default_na=False, low_memory=False)
    df = normalize_columns(df)
    df = clean_object_values(df)
    df = coerce_numeric_columns(df)
    if "gameId" in df.columns:
        df = df[df["gameId"].notna()].copy()
    return df


def read_week_tracking_clean(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="replace") as file:
        first_line = file.readline()

    malformed_header = '"gameId"' in first_line and not first_line.lstrip().startswith('"gameId"')
    if malformed_header:
        df = pd.read_csv(
            path,
            names=TRACKING_BASE_COLUMNS,
            header=None,
            skiprows=1,
            skipinitialspace=True,
            keep_default_na=False,
            low_memory=False,
        )
    else:
        df = pd.read_csv(path, skipinitialspace=True, keep_default_na=False, low_memory=False)

    df = normalize_columns(df)
    df = clean_object_values(df)
    df = coerce_numeric_columns(df)

    missing_columns = [col for col in TRACKING_BASE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{path.name} missing required tracking columns: {missing_columns}")

    df = df[TRACKING_BASE_COLUMNS].copy()
    if "team" in df.columns:
        df["team"] = df["team"].str.upper()
    if "playDirection" in df.columns:
        df["playDirection"] = df["playDirection"].str.lower()
    if "event" in df.columns:
        df["event"] = df["event"].str.lower()

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
    # Placeholder until time-to-throw and LOS-relative throw location are added.
    df["ttt_le_1_5_and_behind_los"] = pd.NA
    df["is_gravity_candidate_base"] = ~(df["is_screen"] | df["is_rpo"] | df["is_qb_spike"])
    return df


def save_cleaned_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def format_play_time(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    formatted = parsed.dt.strftime("%M:%S.%f").str.slice(0, 9)
    return formatted


def finalize_gravity_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=[c for c in GRAVITY_DROP_COLUMNS if c in df.columns]).copy()
    out = out.rename(columns={k: v for k, v in GRAVITY_RENAME_COLUMNS.items() if k in out.columns})
    if "play time" in out.columns:
        out["play time"] = format_play_time(out["play time"])

    left_cols = [c for c in GRAVITY_LEFT_PRIORITY_COLUMNS if c in out.columns]
    remaining_cols = [c for c in out.columns if c not in left_cols]
    return out[left_cols + remaining_cols]


def resolve_data_dir(repo_root: Path, work_dir: Path) -> Path:
    candidate = repo_root / "datasets"
    if candidate.exists():
        return candidate
    fallback = work_dir / "source_copies"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not locate a data source directory (`datasets` or `source_copies`).")


def resolve_weeks_dir(data_dir: Path) -> Path:
    for folder_name in ["Weeks-data", "weeks-data", "weeks_data", "Weeks_Data"]:
        candidate = data_dir / folder_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate weekly tracking folder under datasets.")


def week_sort_key(path: Path) -> int:
    match = re.search(r"week(\d+)", path.stem.lower())
    if match:
        return int(match.group(1))
    return 10**9


def build_pipeline(work_dir: Path) -> Dict[str, int | str | bool]:
    repo_root = work_dir.parent
    data_dir = resolve_data_dir(repo_root, work_dir)
    weeks_dir = resolve_weeks_dir(data_dir)

    cleaned_dir = work_dir / "cleaned_csv"
    outputs_dir = work_dir / "outputs"
    outputs_csv_dir = work_dir / "outputs_csv"
    scripts_dir = work_dir / "scripts"

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_csv_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    games = read_csv_clean(data_dir / "games.csv")
    plays_cleaned = read_csv_clean(data_dir / "plays_cleaned.csv")
    players = read_csv_clean(data_dir / "players.csv")
    pff = read_csv_clean(data_dir / "pffScoutingData.csv")

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

    week_files = sorted(weeks_dir.glob("week*.csv"), key=week_sort_key)
    if not week_files:
        raise FileNotFoundError(f"No week CSV files found in {weeks_dir}")

    weeks_cleaned_path = cleaned_dir / "weeks_tracking_cleaned.csv"
    weeks_extract_path = cleaned_dir / "weeks_tracking_extracted.csv"
    gravity_base_path = outputs_csv_dir / "gravity_base.csv"

    for file_path in [weeks_cleaned_path, weeks_extract_path, gravity_base_path]:
        if file_path.exists():
            file_path.unlink()

    write_cleaned_header = True
    write_extract_header = True
    write_gravity_header = True

    rows_weeks_cleaned = 0
    rows_weeks_extracted = 0
    rows_tracking_used_for_merge = 0
    rows_gravity_base = 0

    for week_file in week_files:
        week_df = read_week_tracking_clean(week_file)
        week_df["sourceWeekFile"] = week_file.name
        rows_weeks_cleaned += len(week_df)

        week_df.to_csv(weeks_cleaned_path, mode="a", index=False, header=write_cleaned_header)
        write_cleaned_header = False

        week_extract = week_df[TRACKING_EXTRACT_COLUMNS].copy()
        week_extract = week_extract.rename(columns=TRACKING_EXTRACT_OUTPUT_RENAMES)
        rows_weeks_extracted += len(week_extract)
        week_extract.to_csv(weeks_extract_path, mode="a", index=False, header=write_extract_header)
        write_extract_header = False

        week_merge = week_df[TRACKING_MERGE_COLUMNS].copy()
        week_merge = week_merge[week_merge["nflId"].notna()].copy()
        week_merge = week_merge.drop_duplicates(subset=TRACKING_JOIN_COLUMNS)
        rows_tracking_used_for_merge += len(week_merge)

        gravity_week = play_player_roles.merge(
            week_merge, on=KEY_COLUMNS, how="inner", validate="one_to_many"
        )
        gravity_week = finalize_gravity_frame(gravity_week)
        rows_gravity_base += len(gravity_week)
        if len(gravity_week) > 0:
            gravity_week.to_csv(gravity_base_path, mode="a", index=False, header=write_gravity_header)
            write_gravity_header = False

    if write_gravity_header:
        empty_gravity = play_player_roles.head(0).copy()
        for col in [c for c in TRACKING_MERGE_COLUMNS if c not in KEY_COLUMNS]:
            empty_gravity[col] = pd.Series(dtype="float64")
        empty_gravity = finalize_gravity_frame(empty_gravity)
        empty_gravity.to_csv(gravity_base_path, index=False)

    summary: Dict[str, int | str | bool] = {
        "work_dir": str(work_dir),
        "data_dir": str(data_dir),
        "weeks_dir": str(weeks_dir),
        "cleaned_dir": str(cleaned_dir),
        "outputs_dir": str(outputs_dir),
        "outputs_csv_dir": str(outputs_csv_dir),
        "week_files_found": len(week_files),
        "rows_play_context": int(len(play_context)),
        "rows_play_player_roles": int(len(play_player_roles)),
        "rows_weeks_cleaned": int(rows_weeks_cleaned),
        "rows_weeks_extracted": int(rows_weeks_extracted),
        "rows_tracking_used_for_merge": int(rows_tracking_used_for_merge),
        "rows_gravity_base": int(rows_gravity_base),
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
        has_project_root = (candidate / "Data Cleaning + Engineering").exists()
        has_data_root = (candidate / "datasets" / "games.csv").exists() or (candidate / "games.csv").exists()
        if has_project_root and has_data_root:
            return candidate

    raise FileNotFoundError(
        "Could not locate repository root. Run the script/notebook from inside the repo tree."
    )


def main() -> None:
    repo_root = resolve_repo_root()
    work_dir = repo_root / "Data Cleaning + Engineering"
    summary = build_pipeline(work_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
