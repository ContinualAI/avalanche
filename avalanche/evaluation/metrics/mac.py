################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-01-2021                                                             #
# Author(s): Vincenzo Lomonaco, Lorenzo Pellegrini                             #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import Optional

from torch import Tensor
from torch.nn import Module

from avalanche.evaluation import Metric


class MAC(Metric[int]):
    """
    Multiply-and-accumulate metric. Provides a lower bound of the
    computational cost of a model in a hardware-independent way by
    computing the number of multiplications. Currently supports only
    Linear or Conv2d modules. Other operations are ignored.
    """
    def __init__(self):
        """
        Creates an instance of the MAC metric.
        """
        self.hooks = []
        self._compute_cost: Optional[int] = None

    def update(self, model: Module, dummy_input: Tensor):
        """
        Computes the MAC metric.

        :param model: current model.
        :param dummy_input: A tensor of the correct size to feed as input
            to model.
        :return: MAC metric.
        """

        for mod in model.modules():
            if MAC.is_recognized_module(mod):
                def foo(a, b, c):
                    return self.update_compute_cost(a, b, c)
                handle = mod.register_forward_hook(foo)
                self.hooks.append(handle)

        self._compute_cost = 0
        model(dummy_input)  # trigger forward hooks

        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def result(self) -> Optional[int]:
        """
        Return the number of MAC operations as computed in the previous call
        to the `update` method.

        :return: The number of MAC operations or None if `update` has not been
            called yet.
        """
        return self._compute_cost

    def update_compute_cost(self, module, dummy_input, output):
        modname = module.__class__.__name__
        if modname == 'Linear':
            self._compute_cost += dummy_input[0].shape[1] * output.shape[1]
        elif modname == 'Conv2d':
            n, cout, hout, wout = output.shape  # Batch, Channels, Height, Width
            ksize = module.kernel_size[0] * module.kernel_size[1]
            self._compute_cost += cout * hout * wout * ksize

    @staticmethod
    def is_recognized_module(mod):
        modname = mod.__class__.__name__
        return modname == 'Linear' or modname == 'Conv2d'


__all__ = [
    'MAC'
]
