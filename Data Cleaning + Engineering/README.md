# Data Cleaning + Engineering

This folder is an isolated pandas pipeline workspace for building gravity-model datasets
without modifying original raw files.

## Structure

- `source_copies/`: legacy fallback source location (current primary source is `../datasets`).
- `cleaned_csv/`: cleaned versions of base and weekly tracking source files.
- `outputs/`: pipeline summary metadata.
- `outputs_csv/`: merged modeling outputs.
- `scripts/build_gravity_dataset.ipynb`: notebook version of the pandas build pipeline.
- `scripts/build_gravity_dataset.py`: script version of the same pipeline logic.

## Build

Open and run:

```bash
jupyter notebook "Data Cleaning + Engineering/scripts/build_gravity_dataset.ipynb"
```

Or run the script directly:

```bash
python3 "Data Cleaning + Engineering/scripts/build_gravity_dataset.py"
```

## Current outputs

- `outputs/pipeline_summary.json`
- `outputs_csv/play_context.csv`
- `outputs_csv/gravity_base.csv`
- `cleaned_csv/weeks_tracking_cleaned.csv`
- `cleaned_csv/weeks_tracking_extracted.csv`

## Weekly tracking handling

The pipeline now ingests weekly tracking files from `../datasets/Weeks-data` (or `weeks-data`)
and performs:

1. Header/schema cleanup (including malformed header recovery in week files).
2. Missing-value normalization (`NA`, `None`, blank values -> null).
3. Value formatting normalization (column names, numeric coercions, casing).
4. Extraction of:
   - `gameID`, `playID`, `frameID`, `time`, `jerseyNumber`, `playDirection`,
     `x`, `y`, `s`, `a`, `dis`, `o`, `dir`, `event`
5. Frame-level merge into `gravity_base.csv` using one-to-many key join on:
   - `gameId`, `playId`, `nflId`
  


## Outputted Datasets (Michael)

I outputted two datasets:

- `blocker_rusher_matchups.csv`
- `base_filtered.csv`

Both of these files are in the shared Google Drive folder under the subfolder **"Output Datasets"** and under the zip file **michael_datasets.zip**.


### `blocker_rusher_matchups.csv`

This dataset stores the frame-by-frame blocker-versus-rusher matchup assignments for each filtered play during the defined pass rush window.

#### What it represents

Each row corresponds to a specific blocker at a specific frame of a specific play, along with the defensive rusher that blocker is assigned or matched to at that moment.

In other words, this file answers the question:

> For this frame of this play, which defender is this blocker primarily engaged with?

#### Granularity

The dataset is at the level of:

- **one row per `gameId`, `playId`, `frameId`, and `blockerId`**

#### Filters

This dataset only includes plays where:

- dropbackType isn't a rollout or designed QB scramble
- play wasn't nullified by penalty (dropbackType != 'Unknown')
- pass rush window is not <= 10 frames

Pass rush window is defined by:

- Lower bound: 0.5 seconds after ball_snap event happens.
- Upper bound: Whichever happens first - Time-to-throw reaches 3.0 seconds after ball_snap, QB throws pass, QB gets sacked/strip-sacked, QB scrambles outside tackle box




### `base_filtered.csv`

This dataset contains the base tracking data for **all players** on the filtered set of plays that are included in the defensive attention analysis.

#### What it represents

Each row corresponds to a player at a specific frame of a specific filtered play, along with the tracking and contextual information needed to analyze movement, alignment, and interaction structure.

This file is the broader frame-level backbone of the project. Unlike `blocker_rusher_matchups.csv`, which focuses specifically on blocker-to-rusher assignments, `base_filtered.csv` includes the full set of relevant player tracking rows for the plays that passed the project’s filtering criteria.

#### Granularity

The dataset is at the level of:

- **one row per `gameId`, `playId`, `frameId`, and `nflId`**

This means the file contains frame-by-frame observations for all tracked players on each included play.

#### Filters

This dataset only includes plays where:

- dropbackType isn't a rollout or designed QB scramble
- play wasn't nullified by penalty (dropbackType != 'Unknown')
- pass rush window is not <= 10 frames

Frame filters:

- Includes all frames below the upper bound of the frame window for each play, as these are the only possible relevant frames.



