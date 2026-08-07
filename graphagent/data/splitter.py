"""
Dataset splitting utilities.
"""

from __future__ import annotations

from sklearn.model_selection import train_test_split


class DatasetSplitter:
    def __init__(
        self,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42,
    ):

        assert abs(train_size + val_size + test_size - 1.0) < 1e-6

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

    def split(self, dataset):

        train_data, temp_data = train_test_split(
            dataset,
            train_size=self.train_size,
            random_state=self.random_state,
            shuffle=True,
        )

        val_ratio = self.val_size / (self.val_size + self.test_size)

        val_data, test_data = train_test_split(
            temp_data,
            train_size=val_ratio,
            random_state=self.random_state,
            shuffle=True,
        )

        return train_data, val_data, test_data