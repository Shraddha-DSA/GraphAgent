from torch_geometric.loader import DataLoader

from graphagent.data.synthetic_generator import SyntheticWorkflowGenerator
from graphagent.data.dataset import WorkflowDataset
from graphagent.data.splitter import DatasetSplitter


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

print(train_dataset[0])

print()

print(f"Train Graphs : {len(train_dataset)}")
print(f"Validation   : {len(val_dataset)}")
print(f"Test         : {len(test_dataset)}")

print()

batch = next(iter(train_loader))

print(batch)