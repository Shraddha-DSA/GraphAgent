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

        self.task_encoding = {
            "code_generation": [1, 0, 0, 0, 0, 0, 0, 0],
            "bug_fixing": [0, 1, 0, 0, 0, 0, 0, 0],
            "documentation": [0, 0, 1, 0, 0, 0, 0, 0],
            "research": [0, 0, 0, 1, 0, 0, 0, 0],
            "summarization": [0, 0, 0, 0, 1, 0, 0, 0],
            "translation": [0, 0, 0, 0, 0, 1, 0, 0],
            "question_answering": [0, 0, 0, 0, 0, 0, 1, 0],
            "data_analysis": [0, 0, 0, 0, 0, 0, 0, 1],
        }

    def build(self, graph):
        
        task = graph.graph.get(
            "task",
            "code_generation",
        )

        latency = float(
            graph.graph.get(
                "latency",
                0.0,
            )
        )

        token_usage = float(
            graph.graph.get(
                "token_usage",
                0.0,
            )
        )

        cost = float(
            graph.graph.get(
                "cost",
                0.0,
            )
        )

        task_feature = self.task_encoding.get(
            task,
            self.task_encoding["code_generation"],
        )

        
        normalized_latency = min(
            latency / 100.0,
            1.0,
        )

        normalized_tokens = min(
            token_usage / 50000.0,
            1.0,
        )

        normalized_cost = min(
            cost / 2.0,
            1.0,
        )

        features = []

        # Build a feature vector for every node.
        for node in graph.nodes():

            agent_feature = self.agent_encoding.get(
                node,
                [0, 0, 0, 0, 0],
            )

            feature = (
                agent_feature
                + task_feature
                + [
                    normalized_latency,
                    normalized_tokens,
                    normalized_cost,
                ]
            )

            features.append(feature)

        return torch.tensor(
            features,
            dtype=torch.float,
        )