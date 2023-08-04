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
from typing import List, TYPE_CHECKING

from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metric_results import MetricValue
from avalanche.logging import TextLogger
from avalanche.benchmarks.scenarios import OnlineCLExperience

from tqdm import tqdm

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class InteractiveLogger(TextLogger, SupervisedPlugin):
    """
    The `InteractiveLogger` class provides logging facilities
    for the console standard output. The logger shows
    a progress bar during training and evaluation flows and
    interactively display metric results as soon as they
    become available. The logger writes metric results after
    each training epoch, evaluation experience and at the
    end of the entire evaluation stream.

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

    def __init__(self):
        super().__init__(file=sys.stdout)
        self._pbar = None

    def before_training_epoch(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs
    ):
        if isinstance(strategy.experience, OnlineCLExperience):
            return
        super().before_training_epoch(strategy, metric_values, **kwargs)
        self._progress.total = len(strategy.dataloader)

    def after_training_epoch(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs
    ):
        if isinstance(strategy.experience, OnlineCLExperience):
            return
        self._end_progress()
        super().after_training_epoch(strategy, metric_values, **kwargs)

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs
    ):
        if isinstance(strategy.experience, OnlineCLExperience):
            experience = strategy.experience.logging()
            if experience.is_first_subexp:
                super().before_training_exp(strategy, metric_values, **kwargs)
                self._progress.total = experience.sub_stream_length

    def after_training_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs
    ):
        if isinstance(strategy.experience, OnlineCLExperience):
            experience = strategy.experience.logging()
            if experience.is_last_subexp:
                self._end_progress()
                super().after_training_exp(strategy, metric_values, **kwargs)
            else:
                self._progress.update()
                self._progress.refresh()

    def before_eval_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs
    ):
        super().before_eval_exp(strategy, metric_values, **kwargs)
        self._progress.total = len(strategy.dataloader)

    def after_eval_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs
    ):
        self._end_progress()
        super().after_eval_exp(strategy, metric_values, **kwargs)

    def after_training_iteration(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs
    ):
        if isinstance(strategy.experience, OnlineCLExperience):
            return

        self._progress.update()
        self._progress.refresh()
        super().after_training_iteration(strategy, metric_values, **kwargs)

    def after_eval_iteration(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs
    ):
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

    def __getstate__(self):
        out = super().__getstate__()
        del out["_pbar"]
        return out

    def __setstate__(self, state):
        state["_pbar"] = None
        super().__setstate__(state)


__all__ = ["InteractiveLogger"]
