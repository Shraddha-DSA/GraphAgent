from graphagent.api.schemas import WorkflowRequest


request = WorkflowRequest(
    task="code_generation",
    agents=[
        "Planner",
        "Researcher",
        "Coder",
        "Reviewer",
    ],
    edges=[
        ["Planner", "Researcher"],
        ["Researcher", "Coder"],
        ["Coder", "Reviewer"],
    ],
    latency=18.2,
    token_usage=12000,
    cost=0.28,
)

print(request)