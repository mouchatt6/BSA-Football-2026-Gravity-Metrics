"""Pass-rusher Gravity GNN pipeline.

Modules:
    config         path/feature/hyperparam constants
    data_filters   play eligibility and pass-rush window logic
    features       categorical encoders and feature lists
    graph_builder  pandas frame -> torch_geometric.data.Data
    dataset        InMemoryDataset wrapping the builder
    splits         grouped (by gameId) train/val/test split
    models         GraphSAGE and GATv2 architectures
    train          training loop
    evaluate       metrics
    infer          writes gnn_attention_scores.csv / gnn_player_gravity.csv
"""

__all__ = [
    "config",
    "data_filters",
    "features",
    "graph_builder",
    "dataset",
    "splits",
    "models",
    "train",
    "evaluate",
    "infer",
]
