from abc import abstractmethod
from typing import TypeVar, SupportsFloat, List

import torch
from torch import Tensor
from typing_extensions import Protocol

TAccumulator = TypeVar('TAccumulator', bound='RawAccumulator')
TRawData = TypeVar('TRawData')
TAccumulated = TypeVar('TAccumulated')


class RawAccumulator(Protocol[TRawData, TAccumulated]):
    def __call__(self: TAccumulator, data: TRawData) -> TAccumulator:
        ...

    def reset(self: TAccumulator) -> TAccumulator:
        ...

    @property
    @abstractmethod
    def value(self) -> TAccumulated:
        ...


class SumAccumulator(RawAccumulator[SupportsFloat, float]):

    def __init__(self):
        self._accumulator: float = 0.0

    def reset(self):
        self._accumulator = 0.0
        return self

    def __call__(self, data: SupportsFloat):
        self._accumulator += float(data)
        return self

    @property
    def value(self) -> float:
        return self._accumulator


class AverageAccumulator(RawAccumulator[Tensor, float]):

    def __init__(self):
        self._accumulator: float = 0.0
        self._count: float = 0.0

    def reset(self):
        self._accumulator = 0.0
        self._count = 0.0
        return self

    def __call__(self, data: Tensor, weight: SupportsFloat = None):
        if weight is None:
            self._count += float(torch.numel(data))
            self._accumulator += float(torch.sum(data))
        else:
            weight = float(weight)
            self._count += weight
            self._accumulator += weight * float(torch.sum(data))

        return self

    @property
    def value(self) -> float:
        if self._count == 0.0:
            return 0.0
        return self._accumulator / self._count


class TensorAccumulator(RawAccumulator[Tensor, Tensor]):

    def __init__(self):
        self._accumulator: List[Tensor] = []
        self._reference_shape = None

    def reset(self):
        self._accumulator = []
        self._reference_shape = None
        return self

    def __call__(self, data: Tensor):
        if self._reference_shape is None:
            self._reference_shape = list(data.size())
        else:
            self._check_shape(self._reference_shape, data)
        self._accumulator.append(data)
        return self

    @property
    def value(self) -> Tensor:
        if len(self._accumulator) == 0:
            return torch.empty(0)
        return torch.cat(self._accumulator)

    @staticmethod
    def _check_shape(reference_shape: List[int], data: Tensor):
        data_shape = list(data.size())
        if len(data_shape) != len(reference_shape):
            raise ValueError(
                'Incompatible rank. Shape ' + str(data_shape) +
                ' is not compatible with ' + str(reference_shape))

        for i in range(1, len(reference_shape)):
            if data_shape[i] != reference_shape[1]:
                raise ValueError(
                    'Incompatible shape. Shape ' + str(data_shape) +
                    ' is not compatible with ' + str(reference_shape))


__all__ = [
    'TRawData',
    'TAccumulated',
    'RawAccumulator',
    'SumAccumulator',
    'AverageAccumulator',
    'TensorAccumulator']
