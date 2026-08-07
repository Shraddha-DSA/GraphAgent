"""
Application-wide constants.

Keep only values here that are not expected to change frequently.
Configuration values that users may want to modify should go in
the YAML files inside the `configs/` directory.
"""

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT_DIR / "configs"

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
GRAPH_DATA_DIR = DATA_DIR / "graphs"

LOG_DIR = ROOT_DIR / "logs"

MLFLOW_DIR = ROOT_DIR / "mlruns"

PLANNER = "Planner"
RESEARCHER = "Researcher"
CODER = "Coder"
REVIEWER = "Reviewer"
TESTER = "Tester"

AGENT_TYPES = [
    PLANNER,
    RESEARCHER,
    CODER,
    REVIEWER,
    TESTER,
]

NODE_FEATURE_DIM = 16

EDGE_FEATURE_DIM = 8

TASK_TYPES = [
    "code_generation",
    "bug_fixing",
    "documentation",
    "research",
    "summarization",
    "translation",
    "question_answering",
    "data_analysis",
]
SUCCESS = "SUCCESS"
FAILED = "FAILED"
RUNNING = "RUNNING"

WORKFLOW_STATUS = [
    SUCCESS,
    FAILED,
    RUNNING,
]

DEFAULT_SEED = 42

AGENT_TO_ID = {
    "Planner": 0,
    "Researcher": 1,
    "Coder": 2,
    "Reviewer": 3,
    "Tester": 4,
}

ID_TO_AGENT = {
    value: key
    for key, value in AGENT_TO_ID.items()
}

NUM_AGENT_CLASSES = len(AGENT_TO_ID)