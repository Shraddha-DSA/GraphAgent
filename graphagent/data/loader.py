from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class DatasetLoader:

    @staticmethod
    def save(dataset: List[Dict], filepath: str | Path) -> None:

        filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(dataset, file, indent=4)

    @staticmethod
    def load(filepath: str | Path) -> List[Dict]:
        filepath = Path(filepath)

        with open(filepath, "r", encoding="utf-8") as file:
            dataset = json.load(file)

        return dataset