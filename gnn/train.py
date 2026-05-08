"""Training loop for the expected-attention GNN.

Loss is masked MSE on rusher nodes (non-rusher nodes carry y=0 but are not
included in the loss). The play-level target is broadcast across all frames
of a play during graph construction, so frame-level prediction averaged at
inference recovers the play-rusher prediction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .config import CHECKPOINT_DIR, TrainConfig, pick_device


@dataclass
class TrainHistory:
    train_losses: List[float]
    val_losses: List[float]
    best_epoch: int
    best_val_loss: float
    config: dict


def _masked_mse(pred, y, mask):
    import torch
    mask = mask.bool()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    diff = (pred[mask] - y[mask])
    return (diff * diff).mean()


def train_model(
    model,
    train_data: Sequence,
    val_data: Sequence,
    cfg: TrainConfig = TrainConfig(),
    checkpoint_name: str = "gnn_best.pt",
    log_every: int = 1,
) -> TrainHistory:
    import torch
    from torch_geometric.loader import DataLoader

    resolved = pick_device(cfg.device)
    device = torch.device(resolved)
    print(f"[train] using device: {resolved}")
    model = model.to(device)

    train_loader = DataLoader(list(train_data), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(list(val_data), batch_size=cfg.batch_size, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / checkpoint_name

    for epoch in range(cfg.epochs):
        model.train()
        epoch_train = 0.0
        n_train = 0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            pred = model(batch)
            loss = _masked_mse(pred, batch.y, batch.rusher_mask)
            loss.backward()
            optim.step()
            epoch_train += float(loss.item()) * batch.num_graphs
            n_train += batch.num_graphs

        train_loss = epoch_train / max(n_train, 1)
        train_losses.append(train_loss)

        model.eval()
        epoch_val = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = _masked_mse(pred, batch.y, batch.rusher_mask)
                epoch_val += float(loss.item()) * batch.num_graphs
                n_val += batch.num_graphs
        val_loss = epoch_val / max(n_val, 1)
        val_losses.append(val_loss)

        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(),
                         "epoch": epoch,
                         "val_loss": val_loss,
                         "config": asdict(cfg)}, ckpt_path)
        else:
            bad_epochs += 1

        if log_every and (epoch % log_every == 0):
            print(f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}{' *' if improved else ''}")

        if bad_epochs >= cfg.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best val {best_val:.4f} @ epoch {best_epoch}")
            break

    # Reload best
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    return TrainHistory(
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        config=asdict(cfg),
    )
