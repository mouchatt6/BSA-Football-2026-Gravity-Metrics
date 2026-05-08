"""Central configuration for the pass-rusher gravity GNN pipeline.

All paths are resolved relative to the repo root so the module can be imported
from notebooks or scripts living in different folders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _repo_root() -> Path:
    # gnn/ lives at: <repo>/gnn/
    return Path(__file__).resolve().parents[1]


REPO_ROOT: Path = _repo_root()
DCE_ROOT: Path = REPO_ROOT / "Data Cleaning + Engineering"

# Inputs
GRAVITY_BASE_CSV: Path = DCE_ROOT / "outputs_csv" / "gravity_base.csv"
PLAY_CONTEXT_CSV: Path = DCE_ROOT / "outputs_csv" / "play_context.csv"
PLAY_ATTENTION_CSV: Path = DCE_ROOT / "outputs_csv" / "play_attention_scores.csv"
PLAYER_ATTENTION_CSV: Path = DCE_ROOT / "outputs_csv" / "player_attention_scores.csv"
EACH_RUSHER_CSV: Path = DCE_ROOT / "outputs_csv" / "each_pass_rusher.csv"
PFF_POSITION_CSV: Path = DCE_ROOT / "cleaned_csv" / "pff_position_data.csv"
PLAYERS_CSV: Path = DCE_ROOT / "cleaned_csv" / "players_cleaned.csv"

# Outputs
OUTPUT_DIR: Path = DCE_ROOT / "outputs_csv"
GNN_ATTENTION_CSV: Path = OUTPUT_DIR / "gnn_attention_scores.csv"
GNN_PLAYER_GRAVITY_CSV: Path = OUTPUT_DIR / "gnn_player_gravity.csv"

# Cache / checkpoints (gitignored)
GNN_DIR: Path = REPO_ROOT / "gnn"
CACHE_DIR: Path = GNN_DIR / "cache"
CHECKPOINT_DIR: Path = GNN_DIR / "checkpoints"


# Pass-rush window
SNAP_LAG_FRAMES: int = 5         # 0.5 s @ 10 Hz, inclusive lower bound
WINDOW_HARD_CAP_FRAMES: int = 30  # 3.0 s @ 10 Hz, non-inclusive upper bound
WINDOW_END_EVENTS = {
    "pass_forward",
    "autoevent_passforward",
    "qb_sack",
    "qb_strip_sack",
    "fumble",
    "handoff",
    "run",
    "pass_outcome_caught",
    "pass_outcome_incomplete",
}
SNAP_EVENTS = {"ball_snap", "autoevent_ballsnap"}
MIN_WINDOW_LENGTH: int = 5  # minimum number of frames after lower bound

# Sampling
FRAMES_PER_PLAY: int = 6  # evenly spaced frames per play to bound graph count


def pick_device(preferred: str = "auto") -> str:
    """Return the best torch device string available on this machine.

    `preferred` can be "auto", "cuda", "mps", or "cpu". If a non-auto choice
    is unavailable on this machine (e.g. "cuda" on a Mac without NVIDIA
    drivers, or "mps" on Linux), we fall back to the next best option and
    emit a one-line note so the caller knows what they got.

    Order of preference for "auto":  cuda -> mps (Apple Silicon) -> cpu.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    cuda_ok = torch.cuda.is_available()
    mps_ok = (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )

    requested = (preferred or "auto").lower()
    if requested == "cuda":
        if cuda_ok:
            return "cuda"
        fallback = "mps" if mps_ok else "cpu"
        print(f"[pick_device] CUDA not available on this build; using {fallback}.")
        return fallback
    if requested == "mps":
        if mps_ok:
            return "mps"
        fallback = "cuda" if cuda_ok else "cpu"
        print(f"[pick_device] MPS not available; using {fallback}.")
        return fallback
    if requested == "cpu":
        return "cpu"

    # auto
    if cuda_ok:
        return "cuda"
    if mps_ok:
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 25
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 5e-4
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.2
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    device: str = "auto"  # auto-detect cuda / mps / cpu
    early_stopping_patience: int = 5


# Categorical vocabularies (derived once and frozen for reproducibility)
@dataclass(frozen=True)
class FeatureSpec:
    formations: List[str] = field(default_factory=lambda: [
        "SHOTGUN", "EMPTY", "I_FORM", "PISTOL", "SINGLEBACK", "JUMBO", "WILDCAT", "UNKNOWN",
    ])
    coverages: List[str] = field(default_factory=lambda: [
        "Cover 0", "Cover 1", "Cover 2", "Cover 3", "Cover 4", "Cover 6",
        "2-Man", "Quarters", "Bracket", "Goal Line", "Prevent", "Miscellaneous", "UNKNOWN",
    ])
    coverage_types: List[str] = field(default_factory=lambda: ["Man", "Zone", "Other", "UNKNOWN"])
    dropback_types: List[str] = field(default_factory=lambda: ["Traditional", "Scramble", "UNKNOWN"])
    pff_roles: List[str] = field(default_factory=lambda: [
        "Pass Rush", "Pass Block", "Pass Route", "Pass", "Coverage", "UNKNOWN",
    ])
    pff_positions: List[str] = field(default_factory=lambda: [
        "LE", "LEO", "DLT", "NT", "NLT", "NRT", "DRT", "REO", "RE",
        "ROLB", "LOLB", "LILB", "RILB", "MLB",
        "LCB", "RCB", "SCBL", "SCBR", "FS", "SS",
        "QB", "FB", "HB",
        "LT", "LG", "C", "RG", "RT", "TE-L", "TE-R", "TE-iL", "TE-iR",
        "WR", "SLWR", "SRWR", "FLL", "FLR", "UNKNOWN",
    ])
    official_positions: List[str] = field(default_factory=lambda: [
        "QB", "RB", "FB", "WR", "TE", "T", "G", "C",
        "DE", "DT", "NT", "OLB", "ILB", "MLB", "LB",
        "CB", "FS", "SS", "DB", "UNKNOWN",
    ])
    position_groups: List[str] = field(default_factory=lambda: [
        "Edge", "DI", "LB", "DB", "QB", "OL", "RB", "TE", "WR", "Other",
    ])


FEATURE_SPEC = FeatureSpec()


def position_group(off_pos: str) -> str:
    """Coarse position group used for residual breakdowns."""
    if off_pos is None:
        return "Other"
    p = str(off_pos).upper().strip()
    if p in {"DE", "OLB"}:
        return "Edge"
    if p in {"DT", "NT"}:
        return "DI"
    if p in {"ILB", "MLB", "LB"}:
        return "LB"
    if p in {"CB", "FS", "SS", "DB"}:
        return "DB"
    if p in {"QB"}:
        return "QB"
    if p in {"T", "G", "C"}:
        return "OL"
    if p in {"RB", "FB"}:
        return "RB"
    if p in {"TE"}:
        return "TE"
    if p in {"WR"}:
        return "WR"
    return "Other"
