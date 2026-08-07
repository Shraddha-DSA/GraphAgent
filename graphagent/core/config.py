from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT_DIR / "configs"
def load_yaml(file_path: Path) -> dict[str, Any]:

    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
class Config:

    def __init__(self):

        self.config = load_yaml(CONFIG_DIR / "config.yaml")
        self.model = load_yaml(CONFIG_DIR / "model.yaml")
        self.agents = load_yaml(CONFIG_DIR / "agents.yaml")
        self.training = load_yaml(CONFIG_DIR / "training.yaml")

    def reload(self):
        self.__init__()

config = Config()