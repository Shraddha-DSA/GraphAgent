from pathlib import Path

import torch

from graphagent.data.dataset import GraphDataset
from graphagent.graphs.builder import GraphBuilder
from graphagent.models.graphsage import GraphSAGE


class Predictor:

    def __init__(
        self,
        checkpoint="checkpoints/best_model.pt",
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.builder = GraphBuilder()
        self.converter = GraphDataset()

        self.model = GraphSAGE(
            in_channels=16,
            hidden_channels=32,
            num_classes=2,
        )

        checkpoint_path = Path(checkpoint)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}"
            )

        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        self.model.eval()

    @torch.no_grad()
    def predict(self, workflow):
        graph = self.builder.build_graph(workflow)

        graph.graph["task"] = workflow["task"]
        graph.graph["latency"] = workflow["latency"]
        graph.graph["token_usage"] = workflow["token_usage"]
        graph.graph["cost"] = workflow["cost"]
        graph.graph["success"] = False

        data = self.converter.graph_to_data(graph)

        data = data.to(self.device)

        logits = self.model(data)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        success_probability = probabilities[0, 1].item()

        prediction = logits.argmax(
            dim=1
        ).item()

        return {
            "prediction": (
                "Success"
                if prediction == 1
                else "Failure"
            ),
            "success_probability": round(
                success_probability,
                4,
            ),
        }