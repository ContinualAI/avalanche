from typing import SupportsFloat

from avalanche.evaluation import Metric


class Sum(Metric[float]):
    def __init__(self):
        super().__init__()
        self.summed: float = 0.0

    def update(self, value: SupportsFloat) -> None:
        self.summed += float(value)

    def result(self) -> float:
        return self.summed

    def reset(self) -> None:
        self.summed = 0.0


__all__ = ['Sum']
