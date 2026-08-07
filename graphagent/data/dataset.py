"""
PyTorch Geometric Dataset for GraphAgent.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data, InMemoryDataset

from graphagent.graphs.builder import GraphBuilder
from graphagent.graphs.edge_features import EdgeFeatureBuilder
from graphagent.graphs.node_features import NodeFeatureBuilder

from graphagent.core.constants import AGENT_TO_ID

class GraphDataset:
    """
    Converts a NetworkX graph into a PyTorch Geometric Data object.
    """

    def __init__(self):
        self.node_builder = NodeFeatureBuilder()
        self.edge_builder = EdgeFeatureBuilder()

    def graph_to_data(self, graph):

        node_to_idx = {
            node: idx
            for idx, node in enumerate(graph.nodes())
        }

        # Node features
        x = self.node_builder.build(graph)

        # Edge index
        edges = [
            [node_to_idx[s], node_to_idx[t]]
            for s, t in graph.edges()
        ]

        edge_index = (
            torch.tensor(edges, dtype=torch.long)
            .t()
            .contiguous()
        )

        edge_attr = self.edge_builder.build(graph)

        y = torch.tensor(
        [int(graph.graph["success"])],
        dtype=torch.long,
)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )


class WorkflowDataset(InMemoryDataset):
    """
    Dataset of workflow graphs.
    """

    def __init__(self, workflows):

        super().__init__()

        self.builder = GraphBuilder()
        self.converter = GraphDataset()

        self.data_list = []

        for workflow in workflows:

            graph = self.builder.build_graph(workflow)

            data = self.converter.graph_to_data(graph)

            self.data_list.append(data)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]