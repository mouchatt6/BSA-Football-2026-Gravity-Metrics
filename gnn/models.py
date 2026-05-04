"""GNN architectures for expected pass-rusher attention.

Both models share a common skeleton:
    - per-node feature MLP
    - K message-passing layers (SAGEConv or GATv2Conv)
    - global-feature broadcast concat with each node embedding
    - per-rusher scalar regression head

We deliberately do NOT use plain GCN as the primary model; on fully connected
sports graphs vanilla GCN tends to over-smooth.

Imports are lazy (inside class bodies) so that this module can be parsed when
torch is not installed (useful for static checks and notebook outline cells).
"""
from __future__ import annotations

from typing import Optional

from .features import EDGE_FEATURE_DIM, GLOBAL_FEATURE_DIM, NODE_FEATURE_DIM


def build_graphsage(
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.2,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv

    class GraphSAGEAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_in = nn.Sequential(
                nn.Linear(NODE_FEATURE_DIM, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers = nn.ModuleList([
                SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])
            self.global_proj = nn.Linear(GLOBAL_FEATURE_DIM, hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.dropout = dropout

        def forward(self, data):
            x = self.node_in(data.x)
            for layer in self.layers:
                x = F.relu(layer(x, data.edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)
            u = self.global_proj(data.u)
            # Map each node to its graph's global features
            u_per_node = u[data.batch] if hasattr(data, "batch") and data.batch is not None else u.expand(x.size(0), -1)
            h = torch.cat([x, u_per_node], dim=-1)
            return self.head(h).squeeze(-1)

    return GraphSAGEAttention()


def build_gatv2(
    hidden_dim: int = 64,
    num_layers: int = 3,
    heads: int = 4,
    dropout: float = 0.2,
    use_edge_features: bool = True,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv

    class GATv2Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_in = nn.Sequential(
                nn.Linear(NODE_FEATURE_DIM, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers = nn.ModuleList()
            for i in range(num_layers):
                conv = GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    edge_dim=EDGE_FEATURE_DIM if use_edge_features else None,
                    dropout=dropout,
                )
                self.layers.append(conv)
            self.global_proj = nn.Linear(GLOBAL_FEATURE_DIM, hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.use_edge_features = use_edge_features
            self.dropout = dropout

        def forward(self, data):
            x = self.node_in(data.x)
            edge_attr = data.edge_attr if self.use_edge_features else None
            for layer in self.layers:
                x = F.elu(layer(x, data.edge_index, edge_attr=edge_attr))
                x = F.dropout(x, p=self.dropout, training=self.training)
            u = self.global_proj(data.u)
            u_per_node = u[data.batch] if hasattr(data, "batch") and data.batch is not None else u.expand(x.size(0), -1)
            h = torch.cat([x, u_per_node], dim=-1)
            return self.head(h).squeeze(-1)

    return GATv2Attention()


def build_model(name: str, **kwargs):
    name = name.lower()
    if name in {"graphsage", "sage"}:
        return build_graphsage(**{k: v for k, v in kwargs.items() if k in {"hidden_dim", "num_layers", "dropout"}})
    if name in {"gatv2", "gat"}:
        return build_gatv2(**{k: v for k, v in kwargs.items() if k in {"hidden_dim", "num_layers", "heads", "dropout", "use_edge_features"}})
    raise ValueError(f"Unknown model: {name}")
