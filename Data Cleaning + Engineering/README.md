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
