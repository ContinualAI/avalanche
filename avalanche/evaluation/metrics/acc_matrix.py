import torch
from typing import List

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.training.templates import SupervisedTemplate


class AccuracyMatrixPluginMetric(PluginMetric[float]):
    """
    Class for obtaining an Accuracy Matrix for the evaluation stream
    """

    def __init__(self):
        """
        Creates the Accuracy Matrix plugin
        """
        self._accuracy = Accuracy()
        super(AccuracyMatrixPluginMetric, self).__init__()

        self.count = 0
        self.matrix = torch.zeros((1, 1))
        self.online = False

        self.num_training_steps = None

    def reset(self, strategy=None) -> None:
        """Resets the metric.

        :param strategy: The strategy object associated with the stream.
        """
        self.matrix = torch.zeros((1, 1))
        self.count = 0

    def result(self, strategy=None) -> float:
        """Returns the metric result.

        :param strategy: The strategy object associated with the stream.
        :return: The metric result as a torch tensor.
        """
        return self.matrix

    def add_new_task(self, new_length):
        """Adds a new dimension to the accuracy matrix.

        :param new_length: The new dimension of the matrix. We assume a square
                            matrix
        """
        temp = self.matrix.clone()
        self.matrix = torch.zeros((new_length, new_length))
        self.matrix[: temp.size(0), : temp.size(1)] = temp

    def update(self, num_training_steps, eval_exp_id):
        """Updates the matrix with the accuracy value for a given task pair.

        :param num_training_steps: The ID of the current training experience.
        :param eval_exp_id: The ID of the evaluation experience.
        """
        if (max(num_training_steps, eval_exp_id) + 1) > self.matrix.size(0):
            self.add_new_task(max(num_training_steps, eval_exp_id) + 1)

        acc = self._accuracy.result()
        self.matrix[num_training_steps, eval_exp_id] = acc

        self._accuracy.reset()

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        """Packages the metric result.

        :param strategy: The strategy object associated with the stream.
        :return: The metric result. As a MetricValue object.
        """
        metric_value = self.result(strategy)
        plot_x_position = strategy.clock.train_iterations
        metric_name = "EvalStream/Acc_Matrix"

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def after_eval_iteration(self, strategy: "SupervisedTemplate"):
        """Performs actions after each evaluation iteration.

        :param strategy: The strategy object associated with the stream.
        """
        super().after_eval_iteration(strategy)
        self._accuracy.update(strategy.mb_output, strategy.mb_y)

    def after_eval_exp(self, strategy: "SupervisedTemplate"):
        """Performs actions after evaluating an experience.

        :param strategy: The strategy object associated with the stream.
        """
        super().after_eval_exp(strategy)
        curr_exp = strategy.experience.current_experience
        self.update(self.num_training_steps, curr_exp)

    def after_eval(self, strategy: "SupervisedTemplate"):
        """Performs actions after the evaluation phase.

        :param strategy: The strategy object associated with the stream.
        :return: The metric result.
        """
        super().after_eval_exp(strategy)
        return self._package_result(strategy)

    def before_training(self, strategy: "SupervisedTemplate"):
        """Performs actions before the training phase.

        :param strategy: The strategy object associated with the metric.
        """
        super().before_training(strategy)
        if self.num_training_steps is not None:
            self.num_training_steps += 1
        else:
            self.num_training_steps = 0


def accuracy_matrix_metrics() -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :return: A list of plugin metrics.
    """

    metrics = []
    metrics.append(AccuracyMatrixPluginMetric())

    return metrics


__all__ = [
    "accuracy_matrix_metrics",
    "AccuracyMatrixPluginMetric",
]
