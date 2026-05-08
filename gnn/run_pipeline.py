"""End-to-end CLI runner: build graphs, train, evaluate, write outputs.

Usage (run from repo root):
    python -m gnn.run_pipeline --max-plays 200

For the canonical run, prefer the Jupyter notebook `gnn/run_gnn_pipeline.ipynb`.
This script exists for reproducibility and headless execution.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import CACHE_DIR, FRAMES_PER_PLAY, TrainConfig
from .dataset import build_frame_graphs, save_frame_graphs, load_frame_graphs, to_pyg_dataset
from .infer import run_inference_and_write
from .lookups import rusher_position_group_lookup
from .models import build_model
from .splits import grouped_split
from .train import train_model


def main():
    parser = argparse.ArgumentParser(description="Build and train the pass-rusher gravity GNN.")
    parser.add_argument("--max-plays", type=int, default=None,
                        help="Cap number of eligible plays processed (debugging).")
    parser.add_argument("--frames-per-play", type=int, default=FRAMES_PER_PLAY)
    parser.add_argument("--model", choices=["graphsage", "gatv2"], default="graphsage")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="auto = pick best available (cuda > mps > cpu)")
    parser.add_argument("--cache", default="frame_graphs.pkl")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    import pandas as pd

    cache_file = CACHE_DIR / args.cache
    if cache_file.exists() and not args.no_cache:
        print(f"Loading cached frame graphs from {cache_file}")
        graphs = load_frame_graphs(cache_file)
    else:
        print("Building frame graphs from gravity_base.csv ...")
        graphs = build_frame_graphs(
            frames_per_play=args.frames_per_play,
            max_plays=args.max_plays,
        )
        if not args.no_cache:
            save_frame_graphs(graphs, cache_file)

    if not graphs:
        raise RuntimeError("No graphs produced. Check filters and inputs.")

    data_list = to_pyg_dataset(graphs)

    game_ids = [int(g.game_id) for g in graphs]
    split = grouped_split(game_ids)
    train_data = [data_list[i] for i in split.train_idx]
    val_data = [data_list[i] for i in split.val_idx]
    test_data = [data_list[i] for i in split.test_idx]
    print(f"split sizes: train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, device=args.device)
    model = build_model(args.model, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, dropout=cfg.dropout)
    history = train_model(model, train_data, val_data, cfg=cfg,
                          checkpoint_name=f"{args.model}_best.pt")
    print(f"best val loss = {history.best_val_loss:.4f} at epoch {history.best_epoch}")

    play_attention = pd.read_csv(__import__("gnn.config", fromlist=["PLAY_ATTENTION_CSV"]).PLAY_ATTENTION_CSV)
    rusher_pos = rusher_position_group_lookup()

    inference_data = test_data if test_data else val_data
    play_rusher_df, player_df = run_inference_and_write(
        model=model,
        data_list=inference_data,
        play_attention=play_attention,
        rusher_position_lookup=rusher_pos,
        device=cfg.device,
        batch_size=256,
    )
    print(f"wrote {len(play_rusher_df)} play-rusher rows and {len(player_df)} qualified player rows")


if __name__ == "__main__":
    main()
