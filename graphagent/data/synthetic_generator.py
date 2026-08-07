"""
Generate realistic synthetic multi-agent workflows.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass


@dataclass
class WorkflowSample:
    task: str

    agents: list[str]

    edges: list[tuple[str, str]]

    latency: float

    token_usage: int

    cost: float

    success: bool
    next_agent: str


class SyntheticWorkflowGenerator:

    def __init__(self):

        self.tasks = [
            "code_generation",
            "bug_fixing",
            "documentation",
            "research",
            "summarization",
            "translation",
            "question_answering",
            "data_analysis",
        ]

    def generate(self):

        task = random.choice(self.tasks)

        workflow = self._generate_workflow()

        latency = self._latency(workflow)

        tokens = self._tokens(workflow)

        cost = round(tokens * 0.00002, 4)

        success = self._success_probability(workflow)

        next_agent = workflow[-1][1]

        return WorkflowSample(
            task=task,
            agents=self._extract_agents(workflow),
            edges=workflow,
            latency=latency,
            token_usage=tokens,
            cost=cost,
            success=success,
            next_agent=next_agent,
        )
    def generate_dataset(self, n):

        return [
            asdict(self.generate())
            for _ in range(n)
        ]

    def _generate_workflow(self):

        templates = [

            [
                ("Planner", "Researcher"),
                ("Researcher", "Coder"),
                ("Coder", "Reviewer"),
                ("Reviewer", "Tester"),
            ],

            [
                ("Planner", "Coder"),
                ("Coder", "Reviewer"),
                ("Reviewer", "Tester"),
            ],

            [
                ("Planner", "Researcher"),
                ("Researcher", "Coder"),
                ("Coder", "Reviewer"),
                ("Reviewer", "Coder"),
                ("Coder", "Tester"),
            ],

            [
                ("Planner", "Researcher"),
                ("Researcher", "Coder"),
                ("Coder", "Reviewer"),
                ("Reviewer", "Coder"),
                ("Coder", "Reviewer"),
                ("Reviewer", "Tester"),
            ],

            [
                ("Planner", "Researcher"),
                ("Researcher", "Reviewer"),
                ("Reviewer", "Tester"),
            ],
        ]

        return random.choice(templates)
    def _extract_agents(self, edges):

        agents = []

        for s, t in edges:

            if s not in agents:
                agents.append(s)

            if t not in agents:
                agents.append(t)

        return agents



    def _latency(self, workflow):

        return round(
            5 + len(workflow) * random.uniform(3, 7),
            2,
        )

    def _tokens(self, workflow):

        return random.randint(
            3000,
            5000 * len(workflow),
        )

    def _success_probability(self, workflow):

        score = 0.6

        if ("Researcher", "Coder") in workflow:
            score += 0.1

        if ("Reviewer", "Coder") in workflow:
            score += 0.15

        if ("Reviewer", "Tester") in workflow:
            score += 0.1

        score = min(score, 0.95)

        return random.random() < score