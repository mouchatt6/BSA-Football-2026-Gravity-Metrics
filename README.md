# BSA Football 2026: Pass Rusher Gravity Metrics

This repository is for Bruin Sports Analytics football research on a pass rusher "gravity" metric.
The goal is to measure which pass rushers draw more blocking attention than expected, then quantify
that excess attention as a residual.

## Project Scope

Inspired by NBA gravity concepts, this project translates "defensive attention" to pass-rush
situations in football.

High-level modeling concept:

1. Build a play and player-level pass protection context dataset.
2. Compute actual attention from matchup behavior.
3. Model expected attention from context.
4. Define gravity as:

`gravity = actual_attention - expected_attention`

Current repository work is focused on Step 1 (data engineering and merged dataset construction).

## Current Repository Structure

```text
BSA-Football-2026-Gravity-Metrics/
├── README.md
├── datasets/
│   ├── games.csv
│   ├── plays.csv
│   ├── plays_cleaned.csv
│   ├── players.csv
│   └── pffScoutingData.csv
├── Gravity Metrics EDA/
│   ├── Scouting Data EDA.ipynb
│   └── bsa_research_test.ipynb
└── Data Cleaning + Engineering/
    ├── README.md
    ├── source_copies/
    ├── cleaned_csv/
    ├── outputs/
    ├── outputs_csv/
    └── scripts/
        ├── build_gravity_dataset.py
        └── build_gravity_dataset.ipynb
```

## Data Engineering Workspace

`Data Cleaning + Engineering/` is an isolated pipeline workspace.

- `source_copies/`: duplicated CSV inputs used by the pipeline (original files remain untouched).
- `cleaned_csv/`: cleaned versions of duplicated source tables.
- `outputs_csv/`: final merged datasets for analysis in pandas.
- `outputs/pipeline_summary.json`: row counts and run metadata.
- `scripts/build_gravity_dataset.py`: script version of the pipeline.
- `scripts/build_gravity_dataset.ipynb`: notebook walkthrough version of the same flow.

## Current Output Datasets

The pipeline currently writes two main CSV outputs:

1. `play_context.csv`
   - Grain: one row per play (`gameId`, `playId`)
   - Source merge: `plays_cleaned + games`
   - Includes contextual and filter-flag columns (screen/rpo/spike proxies)

2. `gravity_base.csv`
   - Grain: one row per play-player (`gameId`, `playId`, `nflId`)
   - Source merge: `play_context + pffScoutingData + players`
   - Main table for downstream pandas indexing and gravity metric development

## How to Run the Current Pipeline

From repository root:

```bash
python3 "Data Cleaning + Engineering/scripts/build_gravity_dataset.py"
```

Or run the notebook:

```bash
jupyter notebook "Data Cleaning + Engineering/scripts/build_gravity_dataset.ipynb"
```

## Merge Logic (Current)

1. Clean CSV formatting issues (column whitespace, NA strings, key typing).
2. Build `play_context` by left-joining `plays_cleaned` to `games` on `gameId`.
3. Shape PFF table by dropping impact stats not needed for attention/gravity engineering.
4. Build player-level table by joining `play_context` to shaped PFF on (`gameId`, `playId`).
5. Join player bio metadata from `players` on `nflId`.
6. Save `play_context.csv` and `gravity_base.csv`.

## Future Plan

### Near Term

1. Align pipeline input to the `datasets/` folder as source-of-truth (or keep it synced into `source_copies/`).
2. Improve pass-play eligibility logic for:
   - non-screen
   - non-RPO
   - non-QB spike
   - TTT <= 1.5s and behind LOS exclusions (once TTT and LOS fields are integrated)
3. Add validation checks on key uniqueness and merge coverage in pipeline outputs.

### Metric Development

1. Build frame-level blocker-rusher matchup assignment logic.
2. Compute play-level actual attention scores by pass rusher.
3. Add stunt-aware adjustments (screen setter vs looper behavior).
4. Train expected-attention model using contextual features (alignment, personnel, down-distance, formations, play action).
5. Compute residual gravity and normalize with percentiles/z-scores.

### Modeling and Analysis

1. Create evaluation tables for player-level stability and signal quality.
2. Build visual reports for ranking pass rushers by gravity.
3. Compare gravity against conventional pressure/sack metrics to identify unique signal.

## Notes

- Project workflow is pandas-first (no SQL dependency in current pipeline design).
- Original source files should remain unchanged; data engineering should run on copies.
- Large CSV outputs may trigger GitHub large-file warnings; consider Git LFS if output files grow further.
