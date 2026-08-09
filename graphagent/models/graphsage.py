from __future__ import annotations

import torch
import torch.nn.functional as F

from torch_geometric.nn import (
    SAGEConv,
    global_mean_pool,
)


class GraphSAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            hidden_channels,
        )

        self.dropout = torch.nn.Dropout(
            p=dropout,
        )

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

        x = self.conv1(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = global_mean_pool(
            x,
            batch,
        )

        x = self.classifier(x)

        return x