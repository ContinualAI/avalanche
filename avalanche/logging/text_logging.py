################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 2020-01-25                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
import sys
from typing import List

import torch

from avalanche.evaluation.metric_results import MetricValue
from avalanche.logging import StrategyLogger
from avalanche.training.plugins import PluggableStrategy
from avalanche.evaluation.metric_utils import stream_type


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
        name = metric_value.name
        x = metric_value.x_plot
        val = metric_value.value
        self.metric_vals[name] = (name, x, val)

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

    def before_training_exp(self, strategy: 'PluggableStrategy',
                            metric_values: List['MetricValue'], **kwargs):
        super().before_training_exp(strategy, metric_values, **kwargs)
        self._on_exp_start(strategy)

    def before_eval_exp(self, strategy: PluggableStrategy,
                        metric_values: List['MetricValue'], **kwargs):
        super().before_eval_exp(strategy, metric_values, **kwargs)
        self._on_exp_start(strategy)

    def after_training_epoch(self, strategy: 'PluggableStrategy',
                             metric_values: List['MetricValue'], **kwargs):
        super().after_training_epoch(strategy, metric_values, **kwargs)
        print(f'Epoch {strategy.epoch} ended.', file=self.file, flush=True)
        self.print_current_metrics()
        self.metric_vals = {}

    def after_eval_exp(self, strategy: 'PluggableStrategy',
                       metric_values: List['MetricValue'], **kwargs):
        super().after_eval_exp(strategy, metric_values, **kwargs)
        print(f'> Eval on experience {strategy.eval_exp_id} (Task '
              f'{strategy.eval_task_label}) '
              f'from {stream_type(strategy.experience)} stream ended.',
              file=self.file, flush=True)
        self.print_current_metrics()
        self.metric_vals = {}

    def before_training(self, strategy: 'PluggableStrategy',
                        metric_values: List['MetricValue'], **kwargs):
        super().before_training(strategy, metric_values, **kwargs)
        print('-- >> Start of training phase << --', file=self.file, flush=True)

    def before_eval(self, strategy: 'PluggableStrategy',
                    metric_values: List['MetricValue'], **kwargs):
        super().before_eval(strategy, metric_values, **kwargs)
        print('-- >> Start of eval phase << --', file=self.file, flush=True)

    def after_training(self, strategy: 'PluggableStrategy',
                       metric_values: List['MetricValue'], **kwargs):
        super().after_training(strategy, metric_values, **kwargs)
        print('-- >> End of training phase << --', file=self.file, flush=True)

    def after_eval(self, strategy: 'PluggableStrategy',
                   metric_values: List['MetricValue'], **kwargs):
        super().after_eval(strategy, metric_values, **kwargs)
        print('-- >> End of eval phase << --', file=self.file, flush=True)
        self.print_current_metrics()
        self.metric_vals = {}

    def _on_exp_start(self, strategy: 'PluggableStrategy'):
        action_name = 'training' if strategy.is_training else 'eval'
        exp_id = strategy.training_exp_counter if strategy.is_training \
            else strategy.eval_exp_id
        task_id = strategy.train_task_label if strategy.is_training \
            else strategy.eval_task_label
        stream = stream_type(strategy.experience)
        print('-- Starting {} on experience {} (Task {}) from {} stream --'
              .format(action_name, exp_id, task_id, stream), file=self.file,
              flush=True)
