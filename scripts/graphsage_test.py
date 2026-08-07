from graphagent.data.synthetic_generator import SyntheticWorkflowGenerator
from graphagent.graphs.builder import GraphBuilder
from graphagent.data.dataset import GraphDataset
from graphagent.models.graphsage import GraphSAGE

generator = SyntheticWorkflowGenerator()

workflow = generator.generate()

graph = GraphBuilder().build_graph(
    workflow.__dict__
)

data = GraphDataset().graph_to_data(
    graph
)

model = GraphSAGE(
    in_channels=data.num_node_features,
    hidden_channels=32,
    num_classes=2,
)

output = model(data)

print(output)