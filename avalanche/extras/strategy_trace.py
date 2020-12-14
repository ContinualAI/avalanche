import sys
from pathlib import Path
from typing import TextIO, Optional, Union

import warnings
from typing_extensions import Protocol

from avalanche.evaluation import EvalData, OnTrainIteration, OnTestIteration, \
    OnTrainEpochEnd, OnTestStepEnd, OnTestStepStart, OnTrainStepStart, Metric, \
    OnTrainPhaseEnd, OnTestPhaseEnd, OnTrainPhaseStart, OnTestPhaseStart
from avalanche.evaluation.metrics import RunningEpochLoss
from avalanche.evaluation.metrics.accuracy import RunningEpochAccuracy


class StrategyTrace(Protocol):
    def __call__(self, eval_data: EvalData) -> None:
        ...


class DotTrace(StrategyTrace):
    def __init__(self, stdout=True,
                 stderr=False,
                 trace_file: Union[str, Path] = None,
                 *,
                 iterations_per_line: int = 50,
                 lines_between_summaries: int = 4):

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
        self._iteration_loss_metric: Metric = RunningEpochLoss()
        self._iteration_accuracy_metric: Metric = RunningEpochAccuracy()

        # Presentation internals
        self._last_iter = 0
        self._iter_line = iterations_per_line
        self._lines_summary = lines_between_summaries

    def _update_metrics(self, eval_data: EvalData):
        iterations_loss = self._iteration_loss_metric(eval_data)
        iterations_accuracy = self._iteration_accuracy_metric(eval_data)
        loss: Optional[float] = None
        if len(iterations_loss) > 0:
            loss = iterations_loss[0].value

        accuracy: Optional[float] = None
        if len(iterations_accuracy) > 0:
            accuracy = iterations_accuracy[0].value
        return dict(
            iterations_loss=loss,
            iterations_accuracy=accuracy)

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

    def __call__(self, eval_data: EvalData):
        metric_values = self._update_metrics(eval_data)
        running_loss = metric_values['iterations_loss']
        running_accuracy = metric_values['iterations_accuracy']

        if isinstance(eval_data, (OnTrainIteration, OnTestIteration)):
            self._last_iter = eval_data.iteration
            new_line = (self._last_iter + 1) % self._iter_line == 0
            print_summary = ((self._last_iter + 1) %
                             (self._iter_line * self._lines_summary)) == 0
            self._message('.' if eval_data.train_phase else '+')
            if new_line:
                self._message(' [{: >5} iterations]'.format(self._last_iter+1))
                self._new_line()

            if print_summary and eval_data.train_phase:
                msg = '> Running average loss: {:.6f}, accuracy {:.4f}%'.format(
                        running_loss, running_accuracy*100.0)
                self._message(msg)
                self._new_line()

        elif isinstance(eval_data, (OnTestStepStart, OnTrainStepStart)):
            action_name = 'training' if eval_data.train_phase else 'test'
            step_id = eval_data.training_step_id if eval_data.train_phase \
                else eval_data.test_step_id
            task_id = eval_data.training_task_label if eval_data.train_phase \
                else eval_data.test_task_label
            msg = '-- Starting {} on step {} (Task {}) --'.format(
                action_name, step_id, task_id)
            self._message(msg)
            self._new_line()

        elif isinstance(eval_data, (OnTrainEpochEnd, OnTestStepEnd)):
            is_train = eval_data.train_phase
            if is_train:
                msg = 'Epoch {} ended. Loss: {:.6f}, accuracy {:.4f}%'.format(
                    eval_data.epoch, running_loss, running_accuracy*100.0)
            else:
                msg = '> Test on step {} (Task {}) ended. Loss: {:.6f}, ' \
                      'accuracy {:.4f}%' .format(eval_data.test_step_id,
                                                 eval_data.test_task_label,
                                                 running_loss,
                                                 running_accuracy*100.0)

            if (self._last_iter + 1) % self._iter_line != 0:
                self._new_line()
            self._message(msg)
            self._new_line()
        elif isinstance(eval_data, (OnTrainPhaseEnd, OnTestPhaseEnd)):
            phase_name = 'training' if eval_data.train_phase else 'test'
            msg = '-- >> End of {} phase << --'.format(phase_name)
            self._new_line()
            self._message(msg)
            self._new_line()

        elif isinstance(eval_data, (OnTrainPhaseStart, OnTestPhaseStart)):
            phase_name = 'training' if eval_data.train_phase else 'test'
            msg = '-- >> Start of {} phase << --'.format(phase_name)
            self._new_line()
            self._message(msg)
            self._new_line()


DefaultStrategyTrace = DotTrace

__all__ = ['StrategyTrace', 'DotTrace', 'DefaultStrategyTrace']
