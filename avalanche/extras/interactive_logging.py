import sys

from avalanche.evaluation.metrics import Loss
from avalanche.evaluation.metrics.accuracy import Accuracy

from avalanche.training.plugins import PluggableStrategy, StrategyPlugin

from tqdmX import TqdmWrapper
from tqdm import tqdm


class BaseLogger(StrategyPlugin):
    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics
        self.metric_vals = {}

    def _update_metrics(self, strategy: PluggableStrategy, callback: str):
        metric_values = []
        for metric in self.metrics:
            metric_result = getattr(metric, callback)(strategy)
            if metric_result is not None:
                metric_values.extend(metric_result)

        for metric_val in metric_values:
            m_orig = metric_val.origin
            name = metric_val.name
            x = metric_val.x_plot
            val = metric_val.value
            self.metric_vals[m_orig] = (name, x, val)
        return metric_values

    def before_training(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'before_training')

    def before_training_step(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'before_training_step')

    def adapt_train_dataset(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'adapt_train_dataset')

    def before_training_epoch(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'before_training_epoch')

    def before_training_iteration(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'before_training_iteration')

    def before_forward(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'before_forward')

    def after_forward(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'after_forward')

    def before_backward(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'before_backward')

    def after_backward(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'after_backward')

    def after_training_iteration(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'after_training_iteration')

    def before_update(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'before_update')

    def after_update(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'after_update')

    def after_training_epoch(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'after_training_epoch')

    def after_training_step(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'after_training_step')

    def after_training(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'after_training')

    def before_test(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'before_test')

    def adapt_test_dataset(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'adapt_test_dataset')

    def before_test_step(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'before_test_step')

    def after_test_step(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'after_test_step')

    def after_test(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'after_test')

    def before_test_iteration(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'before_test_iteration')

    def before_test_forward(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'before_test_forward')

    def after_test_forward(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'after_test_forward')

    def after_test_iteration(self, strategy: PluggableStrategy, **kwargs):
        return self._update_metrics(strategy, 'after_test_iteration')


class InteractiveLogger(BaseLogger):
    def __init__(self, metrics=None, file=sys.stdout,
                 update_frequency=10):
        if metrics is None:
            metrics = [Loss(), Accuracy()]
        super().__init__(metrics)
        self.file = file
        self.pbar = TqdmWrapper(tqdm())
        self.update_frequency = update_frequency

    def before_training_step(self, strategy: 'PluggableStrategy', **kwargs):
        super().before_training_step(strategy, **kwargs)
        self._on_step_start(strategy)

    def before_training_epoch(self, strategy: PluggableStrategy, **kwargs):
        super().before_training_epoch(strategy, **kwargs)
        self.pbar.tqdm.reset()
        self.pbar.tqdm.total = len(strategy.current_dataloader)

    def before_test_step(self, strategy: PluggableStrategy, **kwargs):
        super().before_test_step(strategy, **kwargs)
        self._on_step_start(strategy)
        self.pbar.tqdm.reset()
        self.pbar.tqdm.total = len(strategy.current_dataloader)

    def after_training_iteration(self, strategy: 'PluggableStrategy', **kwargs):
        super().after_training_iteration(strategy, **kwargs)
        self._pbar_update(strategy.mb_it)

    def after_test_iteration(self, strategy: 'PluggableStrategy', **kwargs):
        super().after_test_iteration(strategy, **kwargs)
        self._pbar_update(strategy.mb_it)

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

    def _pbar_update(self, it):
        # Update progress bar
        self.pbar.update()
        if it % self.update_frequency == 0:
            for name, x, val in self.metric_vals.values():
                self.pbar.add(f'\t{name} = {val}')

    def _on_step_start(self, strategy: 'PluggableStrategy'):
        action_name = 'training' if strategy.is_training else 'test'
        step_id = strategy.step_id
        task_id = strategy.train_task_label if strategy.is_training \
            else strategy.test_task_label
        print('-- Starting {} on step {} (Task {}) --'.format(
              action_name, step_id, task_id), file=self.file, flush=True)


__all__ = ['BaseLogger', 'InteractiveLogger']
