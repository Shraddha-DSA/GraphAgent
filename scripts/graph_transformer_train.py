from torch.optim import Adam
from torch_geometric.loader import DataLoader

from graphagent.data.synthetic_generator import SyntheticWorkflowGenerator
from graphagent.data.splitter import DatasetSplitter
from graphagent.data.dataset import WorkflowDataset

from graphagent.models.graph_transformer import GraphTransformer
from graphagent.models.trainer import Trainer


def main():

    generator = SyntheticWorkflowGenerator()

    workflows = generator.generate_dataset(1000)

    splitter = DatasetSplitter()

    train, val, test = splitter.split(workflows)

    train_dataset = WorkflowDataset(train)
    val_dataset = WorkflowDataset(val)
    test_dataset = WorkflowDataset(test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = GraphTransformer(
        in_channels=5,
        hidden_channels=32,
        num_classes=2,
    )

    optimizer = Adam(
        model.parameters(),
        lr=0.001,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    EPOCHS = 20

    for epoch in range(EPOCHS):

        train_loss, train_acc = trainer.train_epoch()

        val_loss, val_acc = trainer.evaluate(val_loader)

        trainer.save_best(val_acc)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f}"
        )

    test_loss, test_acc = trainer.evaluate(test_loader)

    print(f"\nTest Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()