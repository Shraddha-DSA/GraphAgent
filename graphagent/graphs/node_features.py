from __future__ import annotations

import torch

from graphagent.core.constants import (
    PLANNER,
    RESEARCHER,
    CODER,
    REVIEWER,
    TESTER,
)


class NodeFeatureBuilder:
    def __init__(self):
        self.agent_encoding = {
            PLANNER: [1, 0, 0, 0, 0],
            RESEARCHER: [0, 1, 0, 0, 0],
            CODER: [0, 0, 1, 0, 0],
            REVIEWER: [0, 0, 0, 1, 0],
            TESTER: [0, 0, 0, 0, 1],
        }

    def build(self, graph):
        features = []

        for node in graph.nodes():
            feature = self.agent_encoding[node]

            features.append(feature)

        return torch.tensor(
            features,
            dtype=torch.float,
        )