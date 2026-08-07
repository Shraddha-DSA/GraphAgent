"""
Production trainer for Graph Neural Networks.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        val_loader=None,
        device=None,
        checkpoint_dir="checkpoints",
    ):

        self.device = (
            device
            if device
            else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        self.model = model.to(self.device)

        self.optimizer = optimizer

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.best_accuracy = 0.0

        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def train_epoch(self):

        self.model.train()

        total_loss = 0

        correct = 0

        total = 0

        progress = tqdm(
            self.train_loader,
            desc="Training",
        )

        for batch in progress:

            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(batch)

            loss = F.cross_entropy(
                output,
                batch.y,
            )

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            prediction = output.argmax(dim=1)

            correct += (
                prediction == batch.y
            ).sum().item()

            total += batch.y.size(0)

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
            )

        accuracy = correct / total

        return (
            total_loss / len(self.train_loader),
            accuracy,
        )


    @torch.no_grad()
    def evaluate(self, loader):

        self.model.eval()

        total_loss = 0

        correct = 0

        total = 0

        for batch in loader:

            batch = batch.to(self.device)

            output = self.model(batch)

            loss = F.cross_entropy(
                output,
                batch.y,
            )

            total_loss += loss.item()

            prediction = output.argmax(dim=1)

            correct += (
                prediction == batch.y
            ).sum().item()

            total += batch.y.size(0)

        accuracy = correct / total

        return (
            total_loss / len(loader),
            accuracy,
        )


    def save_best(self, accuracy):

        if accuracy <= self.best_accuracy:
            return

        self.best_accuracy = accuracy

        torch.save(
            self.model.state_dict(),
            self.checkpoint_dir / "best_model.pt",
        )

        print("\nBest Model Saved\n")