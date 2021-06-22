# Date: 22-07-2021                                                             #
# Author(s): Eli Verwimp                                                       #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import Dict

import torch
from torch import Tensor
from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import Accuracy


class ClassAccuracy(Metric[Dict[int, float]]):
    """
    The Class Accuracy metric. This is a standalone metric
    used to compute more specific ones.

    Instances of this metric keeps the running average accuracy
    over multiple <prediction, target> pairs of Tensors,
    provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average accuracy
    for all classes seen and across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an empty dictionary.
    """

    def __init__(self, classes=None):
        """
        Creates an instance of the standalone Accuracy metric.

        By default this metric in its initial state will return an empty
        dictionary. The metric can be updated by using the `update` method
        while the running accuracies can be retrieved using the `result` method.

        :param classes: The classes to keep track of. If None (default), all
        classes seen are tracked.
        :return: None
        """

        if classes is not None:
            self.classes = set(int(c) for c in classes)
        else:
            self.classes = None
        self._class_accuracies: Dict[int, Accuracy] = {}

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor) -> None:
        """
        Update the running accuracy given the true and predicted labels for each
        class.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :return: None.
        """
        if len(true_y) != len(predicted_y):
            raise ValueError('Size mismatch for true_y and predicted_y tensors')

        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        for label in true_y.unique():
            label = int(label)
            if self.classes is not None and label not in self.classes:
                continue

            mask = torch.eq(true_y, label)
            try:
                self._class_accuracies[label].update(predicted_y[mask], true_y[mask])
            except KeyError:
                self._class_accuracies[label] = Accuracy()
                self._class_accuracies[label].update(predicted_y[mask], true_y[mask])

    def result(self) -> Dict[int, float]:
        """
        Retrieves the running accuracy for each class.

        Calling this method will not change the internal state of the metric.

        :return: The running accuracy, as a float value between 0 and 1.
        """
        return {label: acc.result() for label, acc in self._class_accuracies.items()}

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self._class_accuracies = {}


class ClassAccuracyPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all class accuracies plugin metrics
    """
    def __init__(self, reset_at, emit_at, mode, classes=None):
        self._class_accuracy = ClassAccuracy(classes)
        super(ClassAccuracyPluginMetric, self).__init__(
            self._class_accuracy, reset_at=reset_at, emit_at=emit_at,
            mode=mode)

    def update(self, strategy):
        self._class_accuracy.update(strategy.mb_output, strategy.mb_y)

    def __str__(self):
        return "Top1_ClassAcc"
