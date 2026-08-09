from graphagent.api.predictor import Predictor


class WorkflowRecommender:

    def __init__(self):
        self.predictor = Predictor()

    def recommend(self, workflow):
        prediction = self.predictor.predict(workflow)

        recommendations = []

        agents = workflow["agents"]
        edges = workflow["edges"]
        estimated_success = prediction["success_probability"]


        if "Planner" not in agents:
            recommendations.append(
                "Add a Planner at the beginning of the workflow."
            )

        if "Researcher" not in agents:
            recommendations.append(
                "Add a Researcher when the task requires external context."
            )

        if "Reviewer" not in agents:
            recommendations.append(
                "Add a Reviewer before the final testing stage."
            )

        if "Tester" not in agents:
            recommendations.append(
                "Add a Tester before completing the workflow."
            )

        if (
            "Reviewer" in agents
            and "Tester" in agents
            and agents.index("Reviewer")
            > agents.index("Tester")
        ):
            recommendations.append(
                "Move Reviewer before Tester."
            )

        if len(agents) != len(set(agents)):
            recommendations.append(
                "Reduce unnecessary repeated agent executions."
            )

        if len(edges) >= len(agents):
            recommendations.append(
                "Workflow contains loops; review repeated execution paths."
            )

        if workflow["latency"] > 30:
            recommendations.append(
                "Reduce workflow latency by removing unnecessary stages."
            )

        if workflow["token_usage"] > 20000:
            recommendations.append(
                "Reduce token usage by shortening unnecessary agent calls."
            )

        if workflow["cost"] > 0.50:
            recommendations.append(
                "Reduce workflow cost by minimizing expensive agent calls."
            )

        if not recommendations:
            recommendations.append(
                "Workflow structure looks strong."
            )

        return {
            "recommendation": recommendations,
            "estimated_success": round(
                estimated_success,
                4,
            ),
        }