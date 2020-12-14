#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from abc import abstractmethod
from typing import TypeVar, SupportsFloat, List

import torch
from torch import Tensor
from typing_extensions import Protocol

TAccumulator = TypeVar('TAccumulator', bound='RawAccumulator')
TRawData = TypeVar('TRawData')
TAccumulated = TypeVar('TAccumulated')


class RawAccumulator(Protocol[TRawData, TAccumulated]):
    """
    Protocol definition of an accumulator.

    An accumulator receives the values to accumulate (__call__ method)
    and lazily aggregates them when the value property is accessed.

    The reset method can be used to revert the accumulator to its initial state.
    """

    def __call__(self: TAccumulator, data: TRawData) -> TAccumulator:
        """
        Accumulates the given values and returns self.

        :param data: The data to accumulate.
        :return: Self.
        """
        ...

    def reset(self: TAccumulator) -> TAccumulator:
        """
        Reverts this accumulator to its initial state.

        :return: Self.
        """
        ...

    @property
    @abstractmethod
    def value(self) -> TAccumulated:
        """
        Returns the value as the aggregation of previously accumulated values.

        Accessing this property doesn't affect the state of the accumulator.

        :return: The aggregated value.
        """
        ...


class SumAccumulator(RawAccumulator[SupportsFloat, float]):
    """
    A simple accumulator used to sum floating point values.
    """

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
    """
    A simple accumulator used to average over Tensor numeric values.
    """

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
    """
    A simple accumulator used to concatenate Tensors.
    """

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
