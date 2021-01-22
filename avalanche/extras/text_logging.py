import sys

from avalanche.evaluation.metrics import Loss, Accuracy
from avalanche.extras.interactive_logging import BaseLogger
from avalanche.training.plugins import PluggableStrategy


class TextLogger(BaseLogger):
    def __init__(self, metrics=None, file=sys.stdout):
        if metrics is None:
            metrics = [Loss(), Accuracy()]
        super().__init__(metrics)
        self.file = file

    def before_training_step(self, strategy: 'PluggableStrategy', **kwargs):
        super().before_training_step(strategy, **kwargs)
        self._on_step_start(strategy)

    def before_test_step(self, strategy: PluggableStrategy, **kwargs):
        super().before_test_step(strategy, **kwargs)
        self._on_step_start(strategy)

    def after_training_epoch(self, strategy: 'PluggableStrategy', **kwargs):
        super().after_training_epoch(strategy, **kwargs)
        print(f'Epoch {strategy.epoch} ended.', file=self.file, flush=True)
        for name, x, val in self.metric_vals.values():
            print(f'\t{name} = {val}', file=self.file, flush=True)

    def after_test_step(self, strategy: 'PluggableStrategy', **kwargs):
        super().after_test_step(strategy, **kwargs)
        print(f'> Test on step {strategy.step_id} (Task '
              f'{strategy.test_task_label}) ended.')
        for name, x, val in self.metric_vals.values():
            print(f'\t{name} = {val}', file=self.file, flush=True)

    def before_training(self, strategy: 'PluggableStrategy', **kwargs):
        super().before_training(strategy, **kwargs)
        print('-- >> Start of training phase << --', file=self.file, flush=True)

    def before_test(self, strategy: 'PluggableStrategy', **kwargs):
        super().before_test(strategy, **kwargs)
        print('-- >> Start of test phase << --', file=self.file, flush=True)

    def after_training(self, strategy: 'PluggableStrategy', **kwargs):
        super().after_training(strategy, **kwargs)
        print('-- >> End of training phase << --', file=self.file, flush=True)

    def after_test(self, strategy: 'PluggableStrategy', **kwargs):
        super().after_test(strategy, **kwargs)
        print('-- >> End of test phase << --', file=self.file, flush=True)

    def _on_step_start(self, strategy: 'PluggableStrategy'):
        action_name = 'training' if strategy.is_training else 'test'
        step_id = strategy.step_id
        task_id = strategy.train_task_label if strategy.is_training \
            else strategy.test_task_label
        print('-- Starting {} on step {} (Task {}) --'.format(
              action_name, step_id, task_id), file=self.file, flush=True)
