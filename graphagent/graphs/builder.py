from __future__ import annotations

import networkx as nx


class GraphBuilder:

    def build_graph(self, workflow: dict) -> nx.DiGraph:
        graph = nx.DiGraph()

        for agent in workflow["agents"]:
            graph.add_node(agent)
        for source, target in workflow["edges"]:
            graph.add_edge(source, target)

        graph.graph["task"] = workflow["task"]
        graph.graph["latency"] = workflow["latency"]
        graph.graph["token_usage"] = workflow["token_usage"]
        graph.graph["cost"] = workflow["cost"]
        graph.graph["success"] = workflow.get("success", False)

        if "next_agent" in workflow:
            graph.graph["next_agent"] = workflow["next_agent"]

        return graph