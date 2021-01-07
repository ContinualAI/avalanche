from typing import SupportsFloat

from avalanche.evaluation import Metric


class Mean(Metric[float]):
    def __init__(self):
        super().__init__()
        self.summed: float = 0.0
        self.weight: float = 0.0

    def update(self, value: SupportsFloat, weight: SupportsFloat = 1.0) -> None:
        value = float(value)
        weight = float(weight)
        self.summed += value * weight
        self.weight += weight

    def result(self) -> float:
        if self.weight == 0.0:
            return 0.0
        return self.summed / self.weight

    def reset(self) -> None:
        self.summed = 0.0
        self.weight = 0.0


__all__ = ['Mean']
