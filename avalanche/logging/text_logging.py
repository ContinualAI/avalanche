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

import torch

from avalanche.evaluation.metric_results import MetricValue
from avalanche.logging import StrategyLogger
from avalanche.training.plugins import PluggableStrategy


class TextLogger(StrategyLogger):
    def __init__(self, file=sys.stdout):
        """
        Text-based logger that logs metrics in a file.
        By default it prints to the standard output.

        :param file: destination file (default=sys.stdout).
        """
        super().__init__()
        self.file = file
        self.metric_vals = {}

    def log_metric(self, metric_value: 'MetricValue', callback: str) -> None:
        m_orig = metric_value.origin
        name = metric_value.name
        x = metric_value.x_plot
        val = metric_value.value
        self.metric_vals[m_orig] = (name, x, val)

    def _val_to_str(self, m_val):
        if isinstance(m_val, torch.Tensor):
            return '\n' + str(m_val)
        elif isinstance(m_val, float):
            return f'{m_val:.4f}'
        else:
            return str(m_val)

    def print_current_metrics(self):
        sorted_vals = sorted(self.metric_vals.values(),
                        key=lambda x: x[0])
        for name, x, val in sorted_vals:
            val = self._val_to_str(val)
            print(f'\t{name} = {val}', file=self.file, flush=True)

    def before_training_step(self, strategy: 'PluggableStrategy',
                             metric_values: List['MetricValue'], **kwargs):
        super().before_training_step(strategy, metric_values, **kwargs)
        self._on_step_start(strategy)

    def before_test_step(self, strategy: PluggableStrategy,
                         metric_values: List['MetricValue'], **kwargs):
        super().before_test_step(strategy, metric_values, **kwargs)
        self._on_step_start(strategy)

    def after_training_epoch(self, strategy: 'PluggableStrategy',
                             metric_values: List['MetricValue'], **kwargs):
        super().after_training_epoch(strategy, metric_values, **kwargs)
        print(f'Epoch {strategy.epoch} ended.', file=self.file, flush=True)
        self.print_current_metrics()

    def after_test_step(self, strategy: 'PluggableStrategy',
                        metric_values: List['MetricValue'], **kwargs):
        super().after_test_step(strategy, metric_values, **kwargs)
        print(f'> Test on step {strategy.step_id} (Task '
              f'{strategy.test_task_label}) ended.')
        self.print_current_metrics()

    def before_training(self, strategy: 'PluggableStrategy',
                        metric_values: List['MetricValue'], **kwargs):
        super().before_training(strategy, metric_values, **kwargs)
        print('-- >> Start of training phase << --', file=self.file, flush=True)

    def before_test(self, strategy: 'PluggableStrategy',
                    metric_values: List['MetricValue'], **kwargs):
        super().before_test(strategy, metric_values, **kwargs)
        print('-- >> Start of test phase << --', file=self.file, flush=True)

    def after_training(self, strategy: 'PluggableStrategy',
                       metric_values: List['MetricValue'], **kwargs):
        super().after_training(strategy, metric_values, **kwargs)
        print('-- >> End of training phase << --', file=self.file, flush=True)

    def after_test(self, strategy: 'PluggableStrategy',
                   metric_values: List['MetricValue'], **kwargs):
        super().after_test(strategy, metric_values, **kwargs)
        print('-- >> End of test phase << --', file=self.file, flush=True)

    def _on_step_start(self, strategy: 'PluggableStrategy'):
        action_name = 'training' if strategy.is_training else 'test'
        step_id = strategy.step_id
        task_id = strategy.train_task_label if strategy.is_training \
            else strategy.test_task_label
        print('-- Starting {} on step {} (Task {}) --'.format(
              action_name, step_id, task_id), file=self.file, flush=True)
