"""
Simple workflow recommendation engine.
"""


class WorkflowRecommender:

    def recommend(self, workflow):

        agents = workflow["agents"]

        recommendations = []

        estimated_success = 0.80

        if "Reviewer" not in agents:

            recommendations.append(
                "Add a Reviewer before testing."
            )

            estimated_success += 0.08

        if "Researcher" not in agents:

            recommendations.append(
                "Add a Researcher for better context."
            )

            estimated_success += 0.05

        if agents.count("Coder") > 2:

            recommendations.append(
                "Reduce repeated coding iterations."
            )

        if "Tester" not in agents:

            recommendations.append(
                "Add a Tester before deployment."
            )

            estimated_success += 0.05

        if not recommendations:

            recommendations.append(
                "Workflow looks well structured."
            )

        estimated_success = min(
            estimated_success,
            0.99,
        )

        return {
            "recommendation": recommendations,
            "estimated_success": round(
                estimated_success,
                2,
            ),
        }