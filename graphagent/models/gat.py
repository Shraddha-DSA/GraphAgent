"""
Graph Attention Network (GAT) for graph classification.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GAT(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
        )

        self.conv2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.dropout = torch.nn.Dropout(dropout)

        self.classifier = torch.nn.Linear(
            hidden_channels,
            num_classes,
        )

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index

        if hasattr(data, "batch"):
            batch = data.batch
        else:
            batch = torch.zeros(
                x.size(0),
                dtype=torch.long,
                device=x.device,
            )

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, batch)

        x = self.classifier(x)

        return x