# Data Cleaning + Engineering

This folder is an isolated pandas pipeline workspace for building gravity-model datasets
without modifying any original repository files.

## Structure

- `source_copies/`: duplicated source CSV files used by the pipeline.
- `cleaned_csv/`: cleaned versions of duplicated source files.
- `outputs/`: pipeline summary metadata.
- `outputs_csv/`: Git-friendly text CSV versions of output datasets.
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
