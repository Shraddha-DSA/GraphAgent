from __future__ import annotations

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
)

from graphagent.graphs.builder import GraphBuilder
from graphagent.graphs.edge_features import EdgeFeatureBuilder
from graphagent.graphs.node_features import NodeFeatureBuilder

from graphagent.core.constants import AGENT_TO_ID


class GraphDataset:
    def __init__(self):

        self.node_builder = NodeFeatureBuilder()
        self.edge_builder = EdgeFeatureBuilder()

    def graph_to_data(self, graph):
        node_to_idx = {
            node: idx
            for idx, node in enumerate(
                graph.nodes()
            )
        }

        x = self.node_builder.build(graph)
        edges = [
            [
                node_to_idx[source],
                node_to_idx[target],
            ]
            for source, target in graph.edges()
        ]

        if edges:

            edge_index = (
                torch.tensor(
                    edges,
                    dtype=torch.long,
                )
                .t()
                .contiguous()
            )

        else:

            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
            )
        edge_attr = self.edge_builder.build(
            graph
        )

        label = int(
            graph.graph["success"]
        )

        y = torch.tensor(
            [label],
            dtype=torch.long,
        )

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )


class WorkflowDataset(InMemoryDataset):
    def __init__(self, workflows):

        super().__init__()

        self.builder = GraphBuilder()

        self.converter = GraphDataset()

        self.data_list = []

        for workflow in workflows:

            graph = self.builder.build_graph(
                workflow
            )


            graph.graph["task"] = workflow[
                "task"
            ]

            graph.graph["latency"] = workflow[
                "latency"
            ]

            graph.graph["token_usage"] = workflow[
                "token_usage"
            ]

            graph.graph["cost"] = workflow[
                "cost"
            ]

            graph.graph["success"] = workflow[
                "success"
            ]

            data = self.converter.graph_to_data(
                graph
            )

            self.data_list.append(data)

    def len(self):

        return len(
            self.data_list
        )

    def get(self, idx):

        return self.data_list[idx]