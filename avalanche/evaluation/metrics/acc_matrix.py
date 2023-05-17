import torch
from typing import List

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.benchmarks.scenarios.online_scenario import (
    OnlineCLExperience, 
    OnlineCLScenario,
)
from avalanche.training.templates import SupervisedTemplate


class AccuracyMatrixPluginMetric(PluginMetric[float]):
    """
    Base class for all accuracies plugin metrics
    """

    def __init__(self, mode="eval"):
        """Creates the Accuracy plugin

        :param reset_at:
        :param emit_at:
        :param mode:
        """
        self._accuracy = Accuracy()
        self._mode = mode
        super(AccuracyMatrixPluginMetric, self).__init__()

        self.count = 0
        self.matrix = torch.zeros((1, 1))
        self.last_task_id = 0
        self.online = False

        self.current_task_id = None

        if self._mode == 'train':
            assert False, "This metric only works in eval"

    def reset(self, strategy=None) -> None:
        self.matrix = torch.zeros((1, 1))
        self.count = 0

    def result(self, strategy=None) -> float:
        return self.matrix

    def add_new_task(self, new_length):
        temp = self.matrix.clone()
        self.matrix = torch.zeros((new_length, new_length))
        self.matrix[:temp.size(0), :temp.size(1)] = temp

    def update(self, current_task_id, eval_task_id):
        if (max(current_task_id, eval_task_id) + 1) > self.matrix.size(0):
            self.add_new_task(max(current_task_id, eval_task_id)+1)

        acc = self._accuracy.result()
        self.matrix[current_task_id, eval_task_id] = acc

        self._accuracy.reset()

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        metric_value = self.result(strategy)
        plot_x_position = strategy.clock.train_iterations

        if self._mode == "train":
            metric_name = "TrainStream/Acc_Matrix"
        else:
            metric_name = "EvalStream/Acc_Matrix"

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def after_eval_iteration(self, strategy: "SupervisedTemplate"):
        super().after_eval_iteration(strategy)
        if self._mode == "eval":
            self._accuracy.update(strategy.mb_output, strategy.mb_y)

    def after_eval_exp(self, strategy: "SupervisedTemplate"):
        super().after_eval_exp(strategy)
        if self._mode == "eval":
            curr_exp = strategy.experience.current_experience
            self.update(self.current_task_id, curr_exp)

    def after_eval(self, strategy: "SupervisedTemplate"):
        super().after_eval_exp(strategy)
        if self._mode == "eval":
            return self._package_result(strategy)

    def before_training(self, strategy: "SupervisedTemplate"):
        super().before_training(strategy)
        if self.current_task_id is not None:
            self.current_task_id += 1
        else:
            self.current_task_id = 0     


def accuracy_matrix_metrics(
) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch accuracy at training time.
    :param epoch: If True, will return a metric able to log
        the epoch accuracy at training time.

    :return: A list of plugin metrics.
    """

    metrics = []
    metrics.append(AccuracyMatrixPluginMetric("eval"))

    return metrics


__all__ = [
    "accuracy_matrix_metrics",
    "AccuracyMatrixPluginMetric",
]
