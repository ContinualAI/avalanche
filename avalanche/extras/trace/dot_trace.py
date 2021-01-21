import sys
import warnings
from pathlib import Path
from typing import Union, Optional, TextIO, TYPE_CHECKING, List

from avalanche.evaluation.metrics import RunningEpochLoss, RunningEpochAccuracy
from avalanche.extras.trace import StrategyTrace

if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy
    from avalanche.evaluation.metric_results import MetricValue
    from avalanche.evaluation import PluginMetric


class DotTrace(StrategyTrace):

    NO_LOSS_METRIC = 'No proper loss metric found. For the loss to be logged,' \
                     ' the "running epoch" loss metric must be added with ' \
                     'log_to_board=True on both the train and test phases.'

    NO_ACCURACY_METRIC = 'No proper loss metric found. For the accuracy to be' \
                         ' logged, the "running epoch" accuracy metric must ' \
                         'be added with log_to_board=True on both the train ' \
                         'and test phases.'

    def __init__(self,
                 stdout=True,
                 stderr=False,
                 trace_file: Union[str, Path] = None,
                 *,
                 iterations_per_line: int = 50,
                 lines_between_summaries: int = 4):
        super().__init__()

        if stdout and stderr:
            warnings.warn('Logging trace on both standard output and error. '
                          'This may lead to a duplicate output')

        if trace_file is not None:
            trace_file = Path(trace_file)
        self._trace_file_path: Optional[Path] = trace_file
        trace_handle = None
        if self._trace_file_path is not None:
            trace_handle = open(str(self._trace_file_path), "a+")
        self._trace_file_handle: Optional[TextIO] = trace_handle
        self._stdout: bool = stdout
        self._stderr: bool = stderr

        # Trace warnings
        self._loss_warned = False
        self._accuracy_warned = False

        # Last values
        self._running_loss = None
        self._running_accuracy = None

        # Presentation internals
        self._last_iter = 0
        self._iter_line = iterations_per_line
        self._lines_summary = lines_between_summaries

    def _update_metrics(self, metric_values: List['MetricValue']):
        loss_metric = StrategyTrace._find_metric(
            metric_values, (RunningEpochLoss,))

        accuracy_metric = StrategyTrace._find_metric(
            metric_values, (RunningEpochAccuracy,))

        # Store the _has flag so that we don't show the warning again
        if loss_metric is None and not self._loss_warned:
            warnings.warn(DotTrace.NO_LOSS_METRIC)
            self._loss_warned = True

        if accuracy_metric is None and not self._accuracy_warned:
            warnings.warn(DotTrace.NO_ACCURACY_METRIC)
            self._accuracy_warned = True

        if loss_metric is not None:
            self._running_loss = float(loss_metric.value)
        else:
            self._running_loss = None

        if accuracy_metric is not None:
            self._running_accuracy = float(accuracy_metric.value)
        else:
            self._running_accuracy = None

    def _new_line(self):
        if self._stdout:
            print(flush=True)
        if self._stderr:
            print(flush=True, file=sys.stderr)
        try:
            if self._trace_file_handle is not None:
                self._trace_file_handle.write('\n')
                self._trace_file_handle.flush()
        except IOError as err:
            self._file_error(err, new_line=False)

    def _message(self, msg: str):
        if self._stdout:
            print(msg, end='', flush=True)
        if self._stderr:
            print(end='', flush=True, file=sys.stderr)
        try:
            if self._trace_file_handle is not None:
                self._trace_file_handle.write(msg)
                self._trace_file_handle.flush()
        except IOError as err:
            self._file_error(err)

    def _file_error(self, err, new_line=True):
        try:
            self._trace_file_handle.close()
        except IOError:
            pass
        self._trace_file_handle = None
        if new_line and self._stderr:
            # Add a new line to avoid appending the message
            # just after the dot
            print(flush=True, file=sys.stderr)
        print('An error occurred while writing trace info on file. '
              'File logging will be disabled.', file=sys.stderr)
        print(err, flush=True, file=sys.stderr)

    def after_training_iteration(self, strategy: 'PluggableStrategy',
                                 metric_values: List['MetricValue'], **kwargs):
        self._on_iteration(strategy, metric_values)

    def after_test_iteration(self, strategy: 'PluggableStrategy',
                             metric_values: List['MetricValue'], **kwarg):
        self._on_iteration(strategy, metric_values)

    def before_training_step(self, strategy: 'PluggableStrategy',
                             metric_values: List['MetricValue'], **kwarg):
        self._on_step_start(strategy)

    def before_test_step(self, strategy: 'PluggableStrategy',
                         metric_values: List['MetricValue'], **kwarg):
        self._on_step_start(strategy)

    def after_training_epoch(self, strategy: 'PluggableStrategy',
                             metric_values: List['MetricValue'], **kwarg):
        self._on_epoch_end(strategy)

    def after_test_step(self, strategy: 'PluggableStrategy',
                        metric_values: List['MetricValue'], **kwarg):
        self._on_epoch_end(strategy)

    def before_training(self, strategy: 'PluggableStrategy',
                        metric_values: List['MetricValue'], **kwarg):
        self._on_phase_start(strategy)

    def before_test(self, strategy: 'PluggableStrategy',
                    metric_values: List['MetricValue'], **kwarg):
        self._on_phase_start(strategy)

    def after_training(self, strategy: 'PluggableStrategy',
                       metric_values: List['MetricValue'], **kwarg):
        self._on_phase_end(strategy)

    def after_test(self, strategy: 'PluggableStrategy',
                   metric_values: List['MetricValue'], **kwarg):
        self._on_phase_end(strategy)

    def _on_iteration(self, strategy: 'PluggableStrategy',
                      metric_values: List['MetricValue']):
        self._update_metrics(metric_values)
        self._last_iter = strategy.mb_it

        new_line = (self._last_iter + 1) % self._iter_line == 0
        print_summary = ((self._last_iter + 1) %
                         (self._iter_line * self._lines_summary)) == 0
        self._message('.' if strategy.is_training else '+')
        if new_line:
            self._message(' [{: >5} iterations]'.format(self._last_iter + 1))
            self._new_line()

        if self._running_loss is None and self._running_accuracy is None:
            # Nothing to log
            return

        if not print_summary or strategy.is_testing:
            # Wrong phase or not reached a multiple of
            # self._iter_line * self._lines_summary iterations
            return

        if print_summary and strategy.is_training:
            if self._running_loss is not None and \
                    self._running_accuracy is not None:
                # Complete message!
                msg = '> Running average loss: {:.6f}, accuracy {:.4f}%'.format(
                    self._running_loss, self._running_accuracy * 100.0)
            elif self._running_loss is not None:
                msg = '> Running average loss: {:.6f}'.format(
                    self._running_loss)
            else:
                msg = '> Running accuracy {:.4f}%'.format(
                    self._running_accuracy)
            self._message(msg)
            self._new_line()

    def _on_step_start(self, strategy: 'PluggableStrategy'):
        action_name = 'training' if strategy.is_training else 'test'
        step_id = strategy.step_id
        task_id = strategy.train_task_label if strategy.is_training \
            else strategy.test_task_label
        msg = '-- Starting {} on step {} (Task {}) --'.format(
            action_name, step_id, task_id)
        self._message(msg)
        self._new_line()

    def _on_epoch_end(self, strategy: 'PluggableStrategy'):
        if strategy.is_training:
            if self._running_loss is not None and \
                    self._running_accuracy is not None:
                # Complete message!
                msg = 'Epoch {} ended. Loss: {:.6f}, accuracy {:.4f}%'.format(
                    strategy.epoch, self._running_loss,
                    self._running_accuracy * 100.0)
            elif self._running_loss is not None:
                msg = 'Epoch {} ended. Loss: {:.6f}'.format(
                    strategy.epoch, self._running_loss)
            else:
                msg = 'Epoch {} ended. Accuracy {:.4f}%'.format(
                    strategy.epoch, self._running_accuracy * 100.0)
        else:
            if self._running_loss is not None and \
                    self._running_accuracy is not None:
                # Complete message!
                msg = '> Test on step {} (Task {}) ended. Loss: {:.6f}, ' \
                      'accuracy {:.4f}%' \
                    .format(strategy.step_id, strategy.test_task_label,
                            self._running_loss,
                            self._running_accuracy * 100.0)
            elif self._running_loss is not None:
                msg = '> Test on step {} (Task {}) ended. Loss: {:.6f}' \
                    .format(strategy.step_id, strategy.test_task_label,
                            self._running_loss)
            else:
                msg = '> Test on step {} (Task {}) ended. ' \
                      'Accuracy {:.4f}%' \
                    .format(strategy.step_id, strategy.test_task_label,
                            self._running_accuracy * 100.0)

        if (self._last_iter + 1) % self._iter_line != 0:
            self._new_line()
        self._message(msg)
        self._new_line()

    def _on_phase_start(self, strategy: 'PluggableStrategy'):
        phase_name = 'training' if strategy.is_training else 'test'
        msg = '-- >> Start of {} phase << --'.format(phase_name)
        self._new_line()
        self._message(msg)
        self._new_line()

    def _on_phase_end(self, strategy: 'PluggableStrategy'):
        phase_name = 'training' if strategy.is_training else 'test'
        msg = '-- >> End of {} phase << --'.format(phase_name)
        self._new_line()
        self._message(msg)
        self._new_line()

    def log_metric(self, metric_value: 'MetricValue'):
        pass

    @staticmethod
    def recommended_metrics() -> List['PluginMetric']:
        # TODO: continue
        return [RunningEpochLoss(train=True, test=True),
                RunningEpochAccuracy(train=True, test=True,
                                     log_to_board=False, log_to_text=True)]


__all__ = [
    'DotTrace'
]
