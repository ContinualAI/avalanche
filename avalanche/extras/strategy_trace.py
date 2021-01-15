import sys
from abc import ABC
from pathlib import Path
from typing import TextIO, Optional, Union, TYPE_CHECKING

import warnings

from avalanche.evaluation.metrics import Loss
from avalanche.evaluation.metrics.accuracy import Accuracy

if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy


class StrategyTrace(ABC):
    def __init__(self):
        pass

    def before_training(self, strategy: 'PluggableStrategy'):
        return None

    def before_training_step(self, strategy: 'PluggableStrategy'):
        return None

    def adapt_train_dataset(self, strategy: 'PluggableStrategy'):
        return None

    def before_training_epoch(self, strategy: 'PluggableStrategy'):
        return None

    def before_training_iteration(self, strategy: 'PluggableStrategy'):
        return None

    def before_forward(self, strategy: 'PluggableStrategy'):
        return None

    def after_forward(self, strategy: 'PluggableStrategy'):
        return None

    def before_backward(self, strategy: 'PluggableStrategy'):
        return None

    def after_backward(self, strategy: 'PluggableStrategy'):
        return None

    def after_training_iteration(self, strategy: 'PluggableStrategy'):
        return None

    def before_update(self, strategy: 'PluggableStrategy'):
        return None

    def after_update(self, strategy: 'PluggableStrategy'):
        return None

    def after_training_epoch(self, strategy: 'PluggableStrategy'):
        return None

    def after_training_step(self, strategy: 'PluggableStrategy'):
        return None

    def after_training(self, strategy: 'PluggableStrategy'):
        return None

    def before_test(self, strategy: 'PluggableStrategy'):
        return None

    def adapt_test_dataset(self, strategy: 'PluggableStrategy'):
        return None

    def before_test_step(self, strategy: 'PluggableStrategy'):
        return None

    def after_test_step(self, strategy: 'PluggableStrategy'):
        return None

    def after_test(self, strategy: 'PluggableStrategy'):
        return None

    def before_test_iteration(self, strategy: 'PluggableStrategy'):
        return None

    def before_test_forward(self, strategy: 'PluggableStrategy'):
        return None

    def after_test_forward(self, strategy: 'PluggableStrategy'):
        return None

    def after_test_iteration(self, strategy: 'PluggableStrategy'):
        return None


class DotTrace(StrategyTrace):
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

        # Trace metrics
        self._running_loss: Loss = Loss()
        self._running_accuracy: Accuracy = Accuracy()

        # Presentation internals
        self._last_iter = 0
        self._iter_line = iterations_per_line
        self._lines_summary = lines_between_summaries

    def _update_metrics(self, strategy: 'PluggableStrategy'):
        self._running_loss.update(strategy.loss, len(strategy.mb_y))
        self._running_accuracy.update(strategy.mb_y, strategy.logits)

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

    def after_training_iteration(self, strategy: 'PluggableStrategy'):
        self._on_iteration(strategy)

    def after_test_iteration(self, strategy: 'PluggableStrategy'):
        self._on_iteration(strategy)

    def before_training_step(self, strategy: 'PluggableStrategy'):
        self._on_step_start(strategy)

    def before_test_step(self, strategy: 'PluggableStrategy'):
        self._on_step_start(strategy)

    def after_training_epoch(self, strategy: 'PluggableStrategy'):
        self._on_epoch_end(strategy)

    def after_test_step(self, strategy: 'PluggableStrategy'):
        self._on_epoch_end(strategy)

    def before_training(self, strategy: 'PluggableStrategy'):
        self._on_phase_start(strategy)

    def before_test(self, strategy: 'PluggableStrategy'):
        self._on_phase_start(strategy)

    def after_training(self, strategy: 'PluggableStrategy'):
        self._on_phase_end(strategy)

    def after_test(self, strategy: 'PluggableStrategy'):
        self._on_phase_end(strategy)

    def _on_iteration(self, strategy: 'PluggableStrategy'):
        self._update_metrics(strategy)
        self._last_iter = strategy.mb_it

        new_line = (self._last_iter + 1) % self._iter_line == 0
        print_summary = ((self._last_iter + 1) %
                         (self._iter_line * self._lines_summary)) == 0
        self._message('.' if strategy.is_training else '+')
        if new_line:
            self._message(' [{: >5} iterations]'.format(self._last_iter + 1))
            self._new_line()

        if print_summary and strategy.is_training:
            msg = '> Running average loss: {:.6f}, accuracy {:.4f}%'.format(
                self._running_loss.result(),
                self._running_accuracy.result() * 100.0)
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
            msg = 'Epoch {} ended. Loss: {:.6f}, accuracy {:.4f}%'.format(
                strategy.epoch, self._running_loss.result(),
                self._running_accuracy.result() * 100.0)
        else:
            msg = '> Test on step {} (Task {}) ended. Loss: {:.6f}, ' \
                  'accuracy {:.4f}%'\
                .format(strategy.step_id, strategy.test_task_label,
                        self._running_loss.result(),
                        self._running_accuracy.result() * 100.0)

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


DefaultStrategyTrace = DotTrace

__all__ = ['StrategyTrace', 'DotTrace', 'DefaultStrategyTrace']
