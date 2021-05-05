################################################################################
# Copyright (c) 2021. ContinualAI. All rights reserved.                        #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 5-5-2021                                                               #
# Author: Antonio Carta, Vincenzo Lomonaco                                     #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

"""
The core module offers fundamental utilities (classes and data structures) that
can be used by inner Avalanche modules. As for now, it contains only the
Strategy Callbacks definition that can be used by the :py:mod:`training`
module for defining new continual learning strategies and by the
:py:mod:`evaluation` module for defining new evaluation plugin metrics.
"""

from abc import ABC
from typing import Generic, TypeVar

CallbackResult = TypeVar('CallbackResult')


class StrategyCallbacks(Generic[CallbackResult], ABC):
    """
    Strategy callbacks provide access before/after each phase of the training
    and evaluation loops. Subclasses can override the desired callbacks to
    customize the loops. In Avalanche, callbacks are used by
    :class:`StrategyPlugin` to implement continual strategies, and
    :class:`StrategyLogger` for automatic logging.

    For each method of the training and evaluation loops, `StrategyCallbacks`
    provide two functions `before_{method}` and `after_{method}`, called
    before and after the method, respectively.

    As a reminder, `BaseStrategy` loops follow the structure shown below:

    **Training loop**
    The training loop is organized as follows::
        train
            train_exp  # for each experience
                adapt_train_dataset
                train_dataset_adaptation
                make_train_dataloader
                train_epoch  # for each epoch
                    # forward
                    # backward
                    # model update

    **Evaluation loop**
    The evaluation loop is organized as follows::
        eval
            eval_exp  # for each experience
                adapt_eval_dataset
                eval_dataset_adaptation
                make_eval_dataloader
                eval_epoch  # for each epoch
                    # forward
                    # backward
                    # model update
    """

    def __init__(self):
        pass

    def before_training(self, *args, **kwargs) -> CallbackResult:
        """ Called before `train` by the `BaseStrategy`. """
        pass

    def before_training_exp(self, *args, **kwargs) -> CallbackResult:
        """ Called before `train_exp` by the `BaseStrategy`. """
        pass

    def before_train_dataset_adaptation(self, *args,
                                        **kwargs) -> CallbackResult:
        """ Called before `train_dataset_adapatation` by the `BaseStrategy`. """
        pass

    def after_train_dataset_adaptation(self, *args, **kwargs) -> CallbackResult:
        """ Called after `train_dataset_adapatation` by the `BaseStrategy`. """
        pass

    def before_training_epoch(self, *args, **kwargs) -> CallbackResult:
        """ Called before `train_epoch` by the `BaseStrategy`. """
        pass

    def before_training_iteration(self, *args, **kwargs) -> CallbackResult:
        """ Called before the start of a training iteration by the
        `BaseStrategy`. """
        pass

    def before_forward(self, *args, **kwargs) -> CallbackResult:
        """ Called before `model.forward()` by the `BaseStrategy`. """
        pass

    def after_forward(self, *args, **kwargs) -> CallbackResult:
        """ Called after `model.forward()` by the `BaseStrategy`. """
        pass

    def before_backward(self, *args, **kwargs) -> CallbackResult:
        """ Called before `criterion.backward()` by the `BaseStrategy`. """
        pass

    def after_backward(self, *args, **kwargs) -> CallbackResult:
        """ Called after `criterion.backward()` by the `BaseStrategy`. """
        pass

    def after_training_iteration(self, *args, **kwargs) -> CallbackResult:
        """ Called after the end of a training iteration by the
        `BaseStrategy`. """
        pass

    def before_update(self, *args, **kwargs) -> CallbackResult:
        """ Called before `optimizer.update()` by the `BaseStrategy`. """
        pass

    def after_update(self, *args, **kwargs) -> CallbackResult:
        """ Called after `optimizer.update()` by the `BaseStrategy`. """
        pass

    def after_training_epoch(self, *args, **kwargs) -> CallbackResult:
        """ Called after `train_epoch` by the `BaseStrategy`. """
        pass

    def after_training_exp(self, *args, **kwargs) -> CallbackResult:
        """ Called after `train_exp` by the `BaseStrategy`. """
        pass

    def after_training(self, *args, **kwargs) -> CallbackResult:
        """ Called after `train` by the `BaseStrategy`. """
        pass

    def before_eval(self, *args, **kwargs) -> CallbackResult:
        """ Called before `eval` by the `BaseStrategy`. """
        pass

    def before_eval_dataset_adaptation(self, *args, **kwargs) -> CallbackResult:
        """ Called before `eval_dataset_adaptation` by the `BaseStrategy`. """
        pass

    def after_eval_dataset_adaptation(self, *args, **kwargs) -> CallbackResult:
        """ Called after `eval_dataset_adaptation` by the `BaseStrategy`. """
        pass

    def before_eval_exp(self, *args, **kwargs) -> CallbackResult:
        """ Called before `eval_exp` by the `BaseStrategy`. """
        pass

    def after_eval_exp(self, *args, **kwargs) -> CallbackResult:
        """ Called after `eval_exp` by the `BaseStrategy`. """
        pass

    def after_eval(self, *args, **kwargs) -> CallbackResult:
        """ Called after `eval` by the `BaseStrategy`. """
        pass

    def before_eval_iteration(self, *args, **kwargs) -> CallbackResult:
        """ Called before the start of a training iteration by the
        `BaseStrategy`. """
        pass

    def before_eval_forward(self, *args, **kwargs) -> CallbackResult:
        """ Called before `model.forward()` by the `BaseStrategy`. """
        pass

    def after_eval_forward(self, *args, **kwargs) -> CallbackResult:
        """ Called after `model.forward()` by the `BaseStrategy`. """
        pass

    def after_eval_iteration(self, *args, **kwargs) -> CallbackResult:
        """ Called after the end of an iteration by the
        `BaseStrategy`. """
        pass
