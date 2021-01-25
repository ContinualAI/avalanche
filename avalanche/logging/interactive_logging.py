################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 2020-01-25                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
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
        self.pbar = tqdm()

    def before_training_epoch(self, strategy: PluggableStrategy,
                              metric_values: List['MetricValue'], **kwargs):
        print(strategy.mb_it)
        super().before_training_epoch(strategy, metric_values, **kwargs)
        self.pbar.reset()
        self.pbar.total = len(strategy.current_dataloader)

    def before_test_step(self, strategy: PluggableStrategy,
                         metric_values: List['MetricValue'], **kwargs):
        super().before_test_step(strategy, metric_values, **kwargs)
        self.pbar.reset()
        self.pbar.total = len(strategy.current_dataloader)

    def after_training_iteration(self, strategy: 'PluggableStrategy',
                                 metric_values: List['MetricValue'], **kwargs):
        super().after_training_iteration(strategy, metric_values, **kwargs)
        self.pbar.update()
        self.pbar.refresh()

    def after_test_iteration(self, strategy: 'PluggableStrategy',
                             metric_values: List['MetricValue'], **kwargs):
        super().after_test_iteration(strategy, metric_values, **kwargs)
        self.pbar.update()
        self.pbar.refresh()
