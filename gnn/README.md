# Pass-Rusher Gravity GNN — Branch `GNN-V1`

This branch adds a Graph Neural Network for **expected pass-rush attention**, paired with the existing deterministic actual-attention pipeline to produce a **gravity** metric per pass rusher:

```
gravity_gnn = actual_attention_gnn − expected_attention_gnn
```

Inspired by NBA gravity statistics, where gravity is defined as the difference between the *attention* a player draws and the *expected attention* given spacing. We replicate the residual idea for NFL pass rushers using frame-level player tracking and PFF blocking assignments from the 2021 BDB dataset.

The plan, model architecture, and validation strategy are described in the [PR description / planning doc](#plan-overview). All code lives at the repo-root [`gnn/`](.) folder, independent of `Data Cleaning + Engineering/`.

---

## 1. What's in this branch

```
<repo-root>/
└── gnn/
    ├── __init__.py
    ├── README.md                     # ← you are here
    ├── config.py                     # paths, hyperparams, vocabularies
    ├── data_filters.py               # play eligibility + pass-rush window
    ├── features.py                   # node / edge / global feature encoders
    ├── graph_builder.py              # frame → FrameGraph (numpy)
    ├── dataset.py                    # streams gravity_base.csv → FrameGraph list
    ├── splits.py                     # grouped (gameId) splits
    ├── models.py                     # GraphSAGE, GATv2 with global feature concat
    ├── train.py                      # masked-MSE training loop, early stopping
    ├── evaluate.py                   # play-rusher aggregation + metrics
    ├── infer.py                      # writes the two output CSVs
    ├── baselines.py                  # Ridge + HistGradientBoosting baselines
    ├── lookups.py                    # players_cleaned-derived helpers
    ├── run_pipeline.py               # headless CLI driver
    └── run_gnn_pipeline.ipynb        # canonical end-to-end notebook
```

`gnn/` reads inputs from `Data Cleaning + Engineering/outputs_csv/` and `Data Cleaning + Engineering/cleaned_csv/` but is otherwise self-contained.

Repo-root additions:

- `requirements.txt`, `environment.txt` — pip install manifests
- `environment.yml` — conda environment (`pyg`, `pytorch`, `conda-forge`)
- Updated `.gitignore` to keep cached graph tensors and checkpoints local

---

## 2. Quick start

### Install

The PyG stack needs PyTorch installed first. Pick **one** path:

**Conda (recommended, especially on macOS):**
```bash
conda env create -f environment.yml
conda activate bsa-gravity-gnn
```

**Pip (CPU-only):**
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Pip (CUDA 12.1):**
```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify:
```bash
python -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)"
```

### Make sure the data is in place

The notebook expects:
- `Data Cleaning + Engineering/outputs_csv/gravity_base.csv` (rebuild via `scripts/build_gravity_dataset.py` if missing — this file is gitignored)
- `Data Cleaning + Engineering/outputs_csv/play_context.csv`
- `Data Cleaning + Engineering/outputs_csv/play_attention_scores.csv`
- `Data Cleaning + Engineering/cleaned_csv/players_cleaned.csv`

### Run

```bash
jupyter lab gnn/run_gnn_pipeline.ipynb
```

Or headless (from repo root):
```bash
python -m gnn.run_pipeline --max-plays 400 --model graphsage --device cpu
```

Outputs land in `Data Cleaning + Engineering/outputs_csv/`:
- `gnn_attention_scores.csv` — one row per (gameId, playId, rusher_nflId)
- `gnn_player_gravity.csv` — qualified-player rollup with z-scores and percentiles

---

## 3. Plan overview

We frame the problem as a per-rusher regression on top of frame-level graphs.

### Step 1 — Filter
`data_filters.filter_eligible_plays` keeps:
- `is_gravity_candidate_base = True` (no screens, RPOs, QB spikes)
- `dropBackType ∈ {Traditional, Scramble}`
- not `ttt_le_1_5_and_behind_los`

### Step 2 — Identify rushers / blockers
PFF roles from `gravity_base` (`pff_role`, `pff_positionLinedUp`, `pff_nflIdBlockedPlayer`).

### Step 3 — Actual attention
We use the existing deterministic blocker→nearest-rusher matching score from `play_attention_scores.csv` as the supervised label and as the realized actual attention. This honors the assumption that gravity needs an **observed** quantity before residualization.

### Step 4 — Expected attention via GNN
For each eligible play we identify the pass-rush window:
- start = first `ball_snap` event + 5 frames (≈ 0.5 s)
- end = first of {pass_forward, qb_sack, qb_strip_sack, scramble events} or snap + 30 frames
- window must contain ≥ 5 frames

We sample `FRAMES_PER_PLAY = 6` evenly spaced frames per play and build one fully connected graph per frame:

| Component | Content |
| --- | --- |
| Nodes | All players present in the frame (≤ 22) |
| Node features | centered x/y, s, a, sin/cos of o and dir, team flags, rusher/blocker/QB flags, height, weight, official-position one-hot, pff_role one-hot |
| Edge features | dx, dy, distance, exp(−d/3), same-team, off-vs-def pair, blocker-rusher candidate flag, sin/cos of orientation diff |
| Global features | down, ydsToGo, quarter, defendersInBox, OL/DL counts, formation/coverage one-hots, play-action, score diff, field position, rusher count |

Two models in `models.py`:
- **GraphSAGE** (`SAGEConv`) — primary; robust on fully connected graphs
- **GATv2** (`GATv2Conv` with edge features) — interpretable attention weights

The prediction head reads each node's embedding plus the broadcast global features and outputs a scalar per node. Loss is masked MSE on rusher nodes; the play-level `avg_attention_score` is broadcast as the per-frame target so frame predictions averaged over the window recover the play-rusher prediction.

### Step 5 — Gravity residual
At inference we average frame predictions per (gameId, playId, rusher_nflId), subtract from the observed actual attention, then z-score and percentile-rank players with ≥ 25 plays. Output goes to `gnn_attention_scores.csv` and `gnn_player_gravity.csv`.

---

## 4. Modeling strategy

Baselines:
1. Existing linear-regression expected-attention notebook (Ridge in `baselines.py`)
2. Tree baseline (`HistGradientBoostingRegressor`)
3. **GraphSAGE GNN** on fully connected frame graphs
4. **GATv2 GNN** with edge features

Splits:
- Primary: random by `gameId` (15 % val / 15 % test in `splits.grouped_split`)
- Secondary: leave-one-week-out via `splits.leave_one_week_out`

---

## 5. Evaluation

Implemented in `evaluate.py` and surfaced in the notebook:

| Check | Reason |
| --- | --- |
| MAE / RMSE / R² on actual attention | Reconstruction quality |
| Spearman / Pearson at play-rusher level | Rank fidelity |
| Position-group breakdown | Edge / DI / LB residuals should be plausibly distributed |
| Split-half stability of `gravity_gnn` | Echoes the STRAIN paper's stability test |
| Calibration scatter (notebook) | Detect systematic miscalibration |
| Tabular vs GNN comparison | The GNN must beat tabular on test MAE / RMSE OR provide better player-level stability |

---

## 6. Assumptions & caveats

- Actual attention is anchored to the deterministic blocker-rusher matching from the existing pipeline.
- Expected attention conditions on play context plus the **early** frame graph; we sample within the pass-rush window rather than only the snap to keep coordinate features informative without leaking final blocking outcomes. v2 should restrict to pre-snap or only the first 1–2 frames if leakage shows up in residual diagnostics.
- Stunt screeners are **not** yet handled separately. Their attention will be inflated; expect noisy gravity for stunt-screener-heavy DTs. v2 will add a stunt classifier.
- v1 uses static frame graphs. Temporal modeling (EvolveGCN / small transformer over frames) is deferred.
- Large generated artifacts (graph pickles, model checkpoints, `gravity_base.csv`) stay local — see `.gitignore`.

---

## 7. References

- Stanford CS224W blog — *Leveraging Graph Neural Networks to Predict NFL Players' Pass Rush*
- Xenopoulos & Silva, *Graph Neural Networks to Predict Sports Outcomes*
- *Making Offensive Play Predictable* — soccer defensive predictability via frame graphs
- *Analyzing Defensive Pass Rush in Football* — STRAIN methodology and stability framework
