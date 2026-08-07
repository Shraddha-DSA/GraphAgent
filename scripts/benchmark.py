from time import perf_counter
from graphagent.models.graph_transformer import GraphTransformer
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from graphagent.data.synthetic_generator import SyntheticWorkflowGenerator
from graphagent.data.splitter import DatasetSplitter
from graphagent.data.dataset import WorkflowDataset

from graphagent.models.gcn import GCN
from graphagent.models.graphsage import GraphSAGE
from graphagent.models.gat import GAT
from graphagent.models.trainer import Trainer


def build_loaders():

    generator = SyntheticWorkflowGenerator()

    workflows = generator.generate_dataset(1000)

    splitter = DatasetSplitter()

    train, val, test = splitter.split(workflows)

    train_dataset = WorkflowDataset(train)
    val_dataset = WorkflowDataset(val)
    test_dataset = WorkflowDataset(test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
    )

    return train_loader, val_loader, test_loader


def evaluate_model(model_class, name):

    train_loader, val_loader, test_loader = build_loaders()

    model = model_class(
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

    start = perf_counter()

    for _ in range(20):
        trainer.train_epoch()

    _, accuracy = trainer.evaluate(test_loader)

    elapsed = perf_counter() - start

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Time": elapsed,
    }


def main():

    models = [
        (GCN, "GCN"),
        (GraphSAGE, "GraphSAGE"),
        (GAT, "GAT"),
        (GraphTransformer, "GraphTransformer"),
    ]

    results = []

    for model, name in models:

        print(f"\nTraining {name}...\n")

        results.append(
            evaluate_model(
                model,
                name,
            )
        )

    print("\n" + "=" * 60)

    print(
        f"{'Model':<15}"
        f"{'Accuracy':<15}"
        f"{'Time(s)':<15}"
    )

    print("-" * 60)

    for result in results:

        print(
            f"{result['Model']:<15}"
            f"{result['Accuracy']:<15.4f}"
            f"{result['Time']:<15.2f}"
        )


if __name__ == "__main__":
    main()