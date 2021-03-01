################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 2020-01-25                                                             #
# Author(s): Antonio Carta, Lorenzo Pellegrini                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
import sys
from typing import List

from avalanche.evaluation.metric_results import MetricValue
from avalanche.logging import TextLogger

from avalanche.training.plugins import PluggableStrategy

from tqdm import tqdm


class InteractiveLogger(TextLogger):
    def __init__(self):
        """
        Logger for interactive output to the standard output. Shows a progress
        bar and prints metric values.
        """
        super().__init__(file=sys.stdout)
        self._pbar = None

    def before_training_epoch(self, strategy: PluggableStrategy,
                              metric_values: List['MetricValue'], **kwargs):
        super().before_training_epoch(strategy, metric_values, **kwargs)
        self._progress.total = len(strategy.current_dataloader)

    def after_training_epoch(self, strategy: PluggableStrategy,
                             metric_values: List['MetricValue'], **kwargs):
        self._end_progress()
        super().after_training_epoch(strategy, metric_values, **kwargs)

    def before_eval_exp(self, strategy: PluggableStrategy,
                        metric_values: List['MetricValue'], **kwargs):
        super().before_eval_exp(strategy, metric_values, **kwargs)
        self._progress.total = len(strategy.current_dataloader)

    def after_eval_exp(self, strategy: 'PluggableStrategy',
                       metric_values: List['MetricValue'], **kwargs):
        self._end_progress()
        super().after_eval_exp(strategy, metric_values, **kwargs)

    def after_training_iteration(self, strategy: 'PluggableStrategy',
                                 metric_values: List['MetricValue'], **kwargs):
        self._progress.update()
        self._progress.refresh()
        super().after_training_iteration(strategy, metric_values, **kwargs)

    def after_eval_iteration(self, strategy: 'PluggableStrategy',
                             metric_values: List['MetricValue'], **kwargs):
        self._progress.update()
        self._progress.refresh()
        super().after_eval_iteration(strategy, metric_values, **kwargs)

    @property
    def _progress(self):
        if self._pbar is None:
            self._pbar = tqdm(leave=True, position=0, file=sys.stdout)
        return self._pbar

    def _end_progress(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
