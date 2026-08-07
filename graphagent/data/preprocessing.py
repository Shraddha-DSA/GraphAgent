from __future__ import annotations

from typing import List, Dict


class DataPreprocessor:
    def __init__(self):
        pass

    def validate_sample(self, sample: Dict) -> bool:
        required_keys = [
            "task",
            "agents",
            "edges",
            "latency",
            "token_usage",
            "cost",
            "success",
        ]

        return all(key in sample for key in required_keys)

    def preprocess(self, dataset: List[Dict]) -> List[Dict]:

        cleaned_dataset = []

        for sample in dataset:

            if not self.validate_sample(sample):
                continue

            cleaned_dataset.append(sample)

        return cleaned_dataset