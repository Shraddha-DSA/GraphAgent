from graphagent.data.synthetic_generator import SyntheticWorkflowGenerator
from graphagent.graphs.builder import GraphBuilder
from graphagent.data.dataset import GraphDataset
from graphagent.models.gcn import GCN

generator = SyntheticWorkflowGenerator()

sample = generator.generate()

graph = GraphBuilder().build_graph(sample.__dict__)

data = GraphDataset().graph_to_data(graph)

model = GCN(
    in_channels=data.num_node_features,
    hidden_channels=16,
    num_classes=2,
)

output = model(data)

print(output)