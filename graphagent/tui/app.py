"""
GraphAgent Textual terminal UI.
"""

import requests

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
)


# FastAPI backend URL.
API_URL = "http://localhost:8000"


class GraphAgentApp(App):
    CSS = """
    Screen {
        align: center middle;
    }

    #main {
        width: 90%;
        height: 90%;
        border: solid cyan;
        padding: 2;
        overflow-y: scroll;
    }

    #title {
        width: 100%;
        text-align: center;
        text-style: bold;
        margin-bottom: 2;
    }

    .label {
        margin-top: 1;
        margin-bottom: 1;
    }

    Input {
        width: 100%;
        margin-bottom: 1;
    }

    Select {
        width: 100%;
        margin-bottom: 1;
    }

    #actions {
        width: 100%;
        height: 5;
        margin-top: 2;
        margin-bottom: 2;
        align: center middle;
    }

    Button {
        width: 20;
        margin: 0 2;
    }

    #result {
        width: 100%;
        min-height: 10;
        border: solid green;
        padding: 1;
        margin-top: 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="main"):

            yield Label(
                "GRAPHAGENT",
                id="title",
            )

            yield Label(
                "Graph Neural Network based Multi-Agent Workflow Analyzer"
            )

            yield Label(
                "Task",
                classes="label",
            )

            yield Select(
                [
                    ("Code Generation", "code_generation"),
                    ("Bug Fixing", "bug_fixing"),
                    ("Research", "research"),
                    ("Documentation", "documentation"),
                    ("Summarization", "summarization"),
                    ("Translation", "translation"),
                ],
                value="code_generation",
                id="task",
            )

            yield Label(
                "Agents (comma separated)",
                classes="label",
            )

            yield Input(
                value="Planner,Researcher,Coder,Reviewer,Tester",
                id="agents",
            )

            yield Label(
                "Latency",
                classes="label",
            )

            yield Input(
                value="20.0",
                id="latency",
            )

            yield Label(
                "Token Usage",
                classes="label",
            )

            yield Input(
                value="12000",
                id="tokens",
            )

            yield Label(
                "Cost",
                classes="label",
            )

            yield Input(
                value="0.25",
                id="cost",
            )


            with Horizontal(id="actions"):

                yield Button(
                    "Predict",
                    id="predict",
                    variant="success",
                )

                yield Button(
                    "Recommend",
                    id="recommend",
                    variant="primary",
                )


            yield Static(
                "Result will appear here...",
                id="result",
            )

        yield Footer()

    def build_workflow(self):


        task = self.query_one(
            "#task",
            Select,
        ).value

        agents_text = self.query_one(
            "#agents",
            Input,
        ).value

        agents = [
            agent.strip()
            for agent in agents_text.split(",")
            if agent.strip()
        ]

        edges = [
            [agents[i], agents[i + 1]]
            for i in range(len(agents) - 1)
        ]

        latency = float(
            self.query_one("#latency", Input).value
        )

        tokens = int(
            self.query_one("#tokens", Input).value
        )

        cost = float(
            self.query_one("#cost", Input).value
        )

        return {
            "task": task,
            "agents": agents,
            "edges": edges,
            "latency": latency,
            "token_usage": tokens,
            "cost": cost,
        }

    def show_result(self, message):


        self.query_one(
            "#result",
            Static,
        ).update(message)

    def on_button_pressed(
        self,
        event: Button.Pressed,
    ) -> None:

        try:

            workflow = self.build_workflow()


            if event.button.id == "predict":

                response = requests.post(
                    f"{API_URL}/predict",
                    json=workflow,
                    timeout=30,
                )

                response.raise_for_status()

                result = response.json()

                self.show_result(
                    f"Prediction: {result['prediction']}\n\n"
                    f"Success Probability: "
                    f"{result['success_probability']:.2%}"
                )

            elif event.button.id == "recommend":

                response = requests.post(
                    f"{API_URL}/recommend",
                    json=workflow,
                    timeout=30,
                )

                response.raise_for_status()

                result = response.json()

                recommendations = "\n".join(
                    f"• {item}"
                    for item in result["recommendation"]
                )

                self.show_result(
                    f"Estimated Success: "
                    f"{result['estimated_success']:.2%}\n\n"
                    f"Recommendations:\n"
                    f"{recommendations}"
                )

        except requests.RequestException as error:

            self.show_result(
                f"API Error:\n{error}"
            )

        except (ValueError, TypeError) as error:

            self.show_result(
                f"Input Error:\n{error}"
            )


if __name__ == "__main__":

    GraphAgentApp().run()