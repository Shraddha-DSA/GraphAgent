from __future__ import annotations

import torch


class EdgeFeatureBuilder:
    def build(self, graph):
        features = []

        for source, target in graph.edges():

            feature = [
                1.0,   # Communication happened once
                1.0,   # Default latency weight
                1.0,   # Default confidence
            ]

            features.append(feature)

        return torch.tensor(
            features,
            dtype=torch.float,
        )