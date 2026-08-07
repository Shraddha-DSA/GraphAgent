"""
Inference engine for GraphAgent.
"""

from pathlib import Path

import torch

from graphagent.graphs.builder import GraphBuilder
from graphagent.data.dataset import GraphDataset
from graphagent.models.graphsage import GraphSAGE


class Predictor:

    def __init__(self, checkpoint="checkpoints/best_model.pt"):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.builder = GraphBuilder()

        self.converter = GraphDataset()

        self.model = GraphSAGE(
            in_channels=5,
            hidden_channels=32,
            num_classes=2,
        )

        checkpoint = Path(checkpoint)

        if checkpoint.exists():

            self.model.load_state_dict(
                torch.load(
                    checkpoint,
                    map_location=self.device,
                )
            )

        self.model.to(self.device)

        self.model.eval()

    @torch.no_grad()
    def predict(self, workflow):

        graph = self.builder.build_graph(workflow)

        data = self.converter.graph_to_data(graph)

        data = data.to(self.device)

        logits = self.model(data)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence = probabilities.max().item()

        prediction = logits.argmax(dim=1).item()

        return {
            "prediction": "Success" if prediction else "Failure",
            "success_probability": round(confidence, 4),
        }