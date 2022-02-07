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
from typing import List, TYPE_CHECKING, Tuple, Type

import torch

from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metric_results import MetricValue, TensorImage
from avalanche.logging import BaseLogger
from avalanche.evaluation.metric_utils import stream_type, phase_and_task

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

UNSUPPORTED_TYPES: Tuple[Type] = (TensorImage,)


class TextLogger(BaseLogger, SupervisedPlugin):
    """
    The `TextLogger` class provides logging facilities
    printed to a user specified file. The logger writes
    metric results after each training epoch, evaluation
    experience and at the end of the entire evaluation stream.

    .. note::
        To avoid an excessive amount of printed lines,
        this logger will **not** print results after
        each iteration. If the user is monitoring
        metrics which emit results after each minibatch
        (e.g., `MinibatchAccuracy`), only the last recorded
        value of such metrics will be reported at the end
        of the epoch.

    .. note::
        Since this logger works on the standard output,
        metrics producing images or more complex visualizations
        will be converted to a textual format suitable for
        console printing. You may want to add more loggers
        to your `EvaluationPlugin` to better support
        different formats.
    """

    def __init__(self, file=sys.stdout):
        """
        Creates an instance of `TextLogger` class.

        :param file: destination file to which print metrics
            (default=sys.stdout).
        """
        super().__init__()
        self.file = file
        self.metric_vals = {}

    def log_single_metric(self, name, value, x_plot) -> None:
        # We only keep track of the last value for each metric
        self.metric_vals[name] = (name, x_plot, value)

    def _val_to_str(self, m_val):
        if isinstance(m_val, torch.Tensor):
            return "\n" + str(m_val)
        elif isinstance(m_val, float):
            return f"{m_val:.4f}"
        else:
            return str(m_val)

    def print_current_metrics(self):
        sorted_vals = sorted(self.metric_vals.values(), key=lambda x: x[0])
        for name, x, val in sorted_vals:
            if isinstance(val, UNSUPPORTED_TYPES):
                continue
            val = self._val_to_str(val)
            print(f"\t{name} = {val}", file=self.file, flush=True)

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super().before_training_exp(strategy, metric_values, **kwargs)
        self._on_exp_start(strategy)

    def before_eval_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super().before_eval_exp(strategy, metric_values, **kwargs)
        self._on_exp_start(strategy)

    def after_training_epoch(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super().after_training_epoch(strategy, metric_values, **kwargs)
        print(
            f"Epoch {strategy.clock.train_exp_epochs} ended.",
            file=self.file,
            flush=True,
        )
        self.print_current_metrics()
        self.metric_vals = {}

    def after_eval_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super().after_eval_exp(strategy, metric_values, **kwargs)
        exp_id = strategy.experience.current_experience
        task_id = phase_and_task(strategy)[1]
        if task_id is None:
            print(
                f"> Eval on experience {exp_id} "
                f"from {stream_type(strategy.experience)} stream ended.",
                file=self.file,
                flush=True,
            )
        else:
            print(
                f"> Eval on experience {exp_id} (Task "
                f"{task_id}) "
                f"from {stream_type(strategy.experience)} stream ended.",
                file=self.file,
                flush=True,
            )
        self.print_current_metrics()
        self.metric_vals = {}

    def before_training(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super().before_training(strategy, metric_values, **kwargs)
        print("-- >> Start of training phase << --", file=self.file, flush=True)

    def before_eval(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super().before_eval(strategy, metric_values, **kwargs)
        print("-- >> Start of eval phase << --", file=self.file, flush=True)

    def after_training(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super().after_training(strategy, metric_values, **kwargs)
        print("-- >> End of training phase << --", file=self.file, flush=True)

    def after_eval(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super().after_eval(strategy, metric_values, **kwargs)
        print("-- >> End of eval phase << --", file=self.file, flush=True)
        self.print_current_metrics()
        self.metric_vals = {}

    def _on_exp_start(self, strategy: "SupervisedTemplate"):
        action_name = "training" if strategy.is_training else "eval"
        exp_id = strategy.experience.current_experience
        task_id = phase_and_task(strategy)[1]
        stream = stream_type(strategy.experience)
        if task_id is None:
            print(
                "-- Starting {} on experience {} from {} stream --".format(
                    action_name, exp_id, stream
                ),
                file=self.file,
                flush=True,
            )
        else:
            print(
                "-- Starting {} on experience {} (Task {}) from {}"
                " stream --".format(action_name, exp_id, task_id, stream),
                file=self.file,
                flush=True,
            )
