from graphagent.api.predictor import Predictor

predictor = Predictor()

workflow = {
    "task": "code_generation",
    "agents": [
        "Planner",
        "Researcher",
        "Coder",
        "Reviewer",
        "Tester",
    ],
    "edges": [
        ("Planner", "Researcher"),
        ("Researcher", "Coder"),
        ("Coder", "Reviewer"),
        ("Reviewer", "Tester"),
    ],
    "latency": 19.8,
    "token_usage": 14000,
    "cost": 0.31,
    "success": True,
}

result = predictor.predict(workflow)

print(result)