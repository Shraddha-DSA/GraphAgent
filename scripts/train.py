"""
Training script for GraphAgent.
"""

from torch.optim import Adam
from torch_geometric.loader import DataLoader

from graphagent.data.synthetic_generator import SyntheticWorkflowGenerator
from graphagent.data.splitter import DatasetSplitter
from graphagent.data.dataset import WorkflowDataset

from graphagent.models.graphsage import GraphSAGE
from graphagent.models.trainer import Trainer

from graphagent.utils.mlflow_logger import MLFlowLogger


def main():
    

    EPOCHS = 20
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    HIDDEN_CHANNELS = 32


    generator = SyntheticWorkflowGenerator()

    workflows = generator.generate_dataset(1000)
    splitter = DatasetSplitter()

    train_workflows, val_workflows, test_workflows = splitter.split(
        workflows
    )

    train_dataset = WorkflowDataset(train_workflows)

    val_dataset = WorkflowDataset(val_workflows)

    test_dataset = WorkflowDataset(test_workflows)


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = GraphSAGE(
        in_channels=5,
        hidden_channels=HIDDEN_CHANNELS,
        num_classes=2,
    )

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    logger = MLFlowLogger(
        experiment_name="GraphAgent"
    )

    logger.start_run(
        run_name="GraphSAGE"
    )

    logger.log_params(
        {
            "model": "GraphSAGE",
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "hidden_channels": HIDDEN_CHANNELS,
        }
    )


    for epoch in range(EPOCHS):

        train_loss, train_acc = trainer.train_epoch()

        val_loss, val_acc = trainer.evaluate(val_loader)

        trainer.save_best(val_acc)

        logger.log_metrics(
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
            },
            step=epoch,
        )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )


    test_loss, test_acc = trainer.evaluate(test_loader)

    logger.log_metrics(
        {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
    )

    logger.log_model(
        model,
        artifact_path="graphsage_model",
    )

    logger.end_run()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()