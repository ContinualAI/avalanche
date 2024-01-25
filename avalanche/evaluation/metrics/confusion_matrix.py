################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################
from matplotlib.figure import Figure
from numpy import arange
from typing import (
    Any,
    Callable,
    Iterable,
    Union,
    Optional,
    TYPE_CHECKING,
    List,
    Literal,
)

import wandb
import numpy as np
import torch
from PIL.Image import Image
from torch import Tensor
from torch.nn.functional import pad

from avalanche.benchmarks import NCScenario
from avalanche.evaluation import PluginMetric, Metric
from avalanche.evaluation.metric_results import (
    AlternativeValues,
    MetricValue,
    MetricResult,
)
from avalanche.evaluation.metric_utils import (
    default_cm_image_creator,
    phase_and_task,
    stream_type,
)

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class ConfusionMatrix(Metric[Tensor]):
    """The standalone confusion matrix metric.

    Instances of this metric keep track of the confusion matrix by receiving a
    pair of "ground truth" and "prediction" Tensors describing the labels of a
    minibatch. Those two tensors can both contain plain labels or
    one-hot/logit vectors.

    The result is the unnormalized running confusion matrix.

    Beware that by default the confusion matrix size will depend on the value of
    the maximum label as detected by looking at both the ground truth and
    predictions Tensors. When passing one-hot/logit vectors, this
    metric will try to infer the number of classes from the vector sizes.
    Otherwise, the maximum label value encountered in the truth/prediction
    Tensors will be used.

    If the user sets the `num_classes`, then the confusion matrix will always be
    of size `num_classes, num_classes`. Whenever a prediction or label tensor is
    provided as logits, only the first `num_classes` units will be considered in
    the confusion matrix computation. If they are provided as numerical labels,
    each of them has to be smaller than `num_classes`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an empty Tensor.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        normalize: Optional[Literal["true", "pred", "all"]] = None,
    ):
        """Creates an instance of the standalone confusion matrix metric.

        By default this metric in its initial state will return an empty Tensor.
        The metric can be updated by using the `update` method while the running
        confusion matrix can be retrieved using the `result` method.

        :param num_classes: The number of classes. Defaults to None,
            which means that the number of classes will be inferred from
            ground truth and prediction Tensors (see class description for more
            details). If not None, the confusion matrix will always be of size
            `num_classes, num_classes` and only the first `num_classes` values
            of output logits or target logits will be considered in the update.
            If the output or targets are provided as numerical labels,
            there can be no label greater than `num_classes`.
        :param normalize: how to normalize confusion matrix.
            None to not normalize
        """
        self._cm_tensor: Optional[Tensor] = None
        """
        The Tensor where the running confusion matrix is stored.
        """
        self._num_classes: Optional[int] = num_classes

        self.normalize = normalize

    @torch.no_grad()
    def update(self, true_y: Tensor, predicted_y: Tensor) -> None:
        """
        Update the running confusion matrix given the true and predicted labels.

        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param predicted_y: The ground truth. Both labels and logit vectors
            are supported.
        :return: None.
        """
        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        if len(true_y.shape) > 2:
            raise ValueError(
                "Confusion matrix supports labels with at" " most 2 dimensions"
            )
        if len(predicted_y.shape) > 2:
            raise ValueError(
                "Confusion matrix supports predictions with at " "most 2 dimensions"
            )

        max_label = -1 if self._num_classes is None else self._num_classes - 1

        # SELECT VALID PORTION OF TARGET AND PREDICTIONS
        true_y = torch.as_tensor(true_y)
        if len(true_y.shape) == 2 and self._num_classes is not None:
            true_y = true_y[:, : max_label + 1]
        predicted_y = torch.as_tensor(predicted_y)
        if len(predicted_y.shape) == 2 and self._num_classes is not None:
            predicted_y = predicted_y[:, : max_label + 1]

        # COMPUTE MAX LABEL AND CONVERT TARGET AND PREDICTIONS IF NEEDED
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            if self._num_classes is None:
                max_label = max(max_label, predicted_y.shape[1] - 1)
            predicted_y = torch.max(predicted_y, 1)[1]
        else:
            # Labels -> check non-negative
            min_label = torch.min(predicted_y).item()
            if min_label < 0:
                raise ValueError("Label values must be non-negative values")
            if self._num_classes is None:
                max_label = max(max_label, torch.max(predicted_y).item())
            elif torch.max(predicted_y).item() >= self._num_classes:
                raise ValueError(
                    "Encountered predicted label larger than" "num_classes"
                )

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            if self._num_classes is None:
                max_label = max(max_label, true_y.shape[1] - 1)
            true_y = torch.max(true_y, 1)[1]
        else:
            # Labels -> check non-negative
            min_label = torch.min(true_y).item()
            if min_label < 0:
                raise ValueError("Label values must be non-negative values")

            if self._num_classes is None:
                max_label = max(max_label, torch.max(true_y).item())
            elif torch.max(true_y).item() >= self._num_classes:
                raise ValueError("Encountered target label larger than" "num_classes")

        if max_label < 0:
            raise ValueError(
                "The Confusion Matrix metric can only handle " "positive label values"
            )

        if self._cm_tensor is None:
            # Create the confusion matrix
            self._cm_tensor = torch.zeros(
                (max_label + 1, max_label + 1), dtype=torch.long
            )
        elif max_label >= self._cm_tensor.shape[0]:
            # Enlarge the confusion matrix
            size_diff = 1 + max_label - self._cm_tensor.shape[0]
            self._cm_tensor = pad(self._cm_tensor, (0, size_diff, 0, size_diff))

        for pattern_idx in range(len(true_y)):
            self._cm_tensor[true_y[pattern_idx]][predicted_y[pattern_idx]] += 1

    def result(self) -> Tensor:
        """
        Retrieves the unnormalized confusion matrix.

        Calling this method will not change the internal state of the metric.

        :return: The running confusion matrix, as a Tensor.
        """
        if self._cm_tensor is None:
            matrix_shape = (0, 0)
            if self._num_classes is not None:
                matrix_shape = (self._num_classes, self._num_classes)
            return torch.zeros(matrix_shape, dtype=torch.long)
        if self.normalize is not None:
            return ConfusionMatrix._normalize_cm(self._cm_tensor, self.normalize)
        return self._cm_tensor

    def reset(self) -> None:
        """
        Resets the metric.

        Calling this method will *not* reset the default number of classes
        optionally defined in the constructor optional parameter.

        :return: None.
        """
        self._cm_tensor = None

    @staticmethod
    def _normalize_cm(cm: Tensor, normalization: Literal["true", "pred", "all"]):
        if normalization not in ("true", "pred", "all"):
            raise ValueError(
                "Invalid normalization parameter. Can be 'true'," " 'pred' or 'all'"
            )

        if normalization == "true":
            cm = cm / cm.sum(dim=1, keepdim=True, dtype=torch.float64)
        elif normalization == "pred":
            cm = cm / cm.sum(dim=0, keepdim=True, dtype=torch.float64)
        elif normalization == "all":
            cm = cm / cm.sum(dtype=torch.float64)
        cm = ConfusionMatrix.nan_to_num(cm)
        return cm

    @staticmethod
    def nan_to_num(matrix: Tensor) -> Tensor:
        # if version.parse(torch.__version__) >= version.parse("1.8.0"):
        #    # noinspection PyUnresolvedReferences
        #    return torch.nan_to_num(matrix)

        numpy_ndarray = matrix.numpy()
        numpy_ndarray = np.nan_to_num(numpy_ndarray)
        return torch.tensor(numpy_ndarray, dtype=matrix.dtype)


class StreamConfusionMatrix(PluginMetric[Tensor]):
    """
    The Stream Confusion Matrix metric.
    This plugin metric only works on the eval phase.

    Confusion Matrix computation can be slow if you compute it for a large
    number of classes. We recommend to set `save_image=False` if the runtime
    is too large.

    At the end of the eval phase, this metric logs the confusion matrix
    relative to all the patterns seen during eval.

    The metric can log either a Tensor or a PIL Image representing the
    confusion matrix.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        normalize: Optional[Literal["true", "pred", "all"]] = None,
        save_image: bool = True,
        image_creator: Callable[
            [Tensor, Optional[Iterable[Any]]], Union[Figure, Image]
        ] = default_cm_image_creator,
        absolute_class_order: bool = False,
    ):
        """
        Creates an instance of the Stream Confusion Matrix metric.

        We recommend to set `save_image=False` if the runtime is too large.
        In fact, a large number of classes may increase the computation time
        of this metric.

        :param num_classes: The number of classes. Defaults to None,
            which means that the number of classes will be inferred from
            ground truth and prediction Tensors (see class description for more
            details). If not None, the confusion matrix will always be of size
            `num_classes, num_classes` and only the first `num_classes` values
            of output logits or target logits will be considered in the update.
            If the output or targets are provided as numerical labels,
            there can be no label greater than `num_classes`.
        :param normalize: Normalizes confusion matrix over the true (rows),
            predicted (columns) conditions or all the population. If None,
            confusion matrix will not be normalized. Valid values are: 'true',
            'pred' and 'all' or None.
        :param save_image: If True, a graphical representation of the confusion
            matrix will be logged, too. If False, only the Tensor representation
            will be logged. Defaults to True.
        :param image_creator: A callable that, given the tensor representation
            of the confusion matrix and the corresponding labels, returns a
            graphical representation of the matrix as a PIL Image. Defaults to
            `default_cm_image_creator`.
        :param absolute_class_order: If true, the labels in the created image
            will be sorted by id, otherwise they will be sorted by order of
            encounter at training time. This parameter is ignored if
            `save_image` is False, or the scenario is not a NCScenario.
        """
        super().__init__()

        self._save_image: bool = save_image
        self.num_classes = num_classes
        self.normalize = normalize
        self.absolute_class_order = absolute_class_order
        self._matrix: ConfusionMatrix = ConfusionMatrix(
            num_classes=num_classes, normalize=normalize
        )
        self._image_creator = image_creator

    def reset(self) -> None:
        self._matrix = ConfusionMatrix(
            num_classes=self.num_classes, normalize=self.normalize
        )

    def result(self) -> Tensor:
        exp_cm = self._matrix.result()
        return exp_cm

    def update(self, true_y: Tensor, predicted_y: Tensor) -> None:
        self._matrix.update(true_y, predicted_y)

    def before_eval(self, strategy) -> None:
        self.reset()

    def after_eval_iteration(self, strategy: "SupervisedTemplate") -> None:
        super().after_eval_iteration(strategy)
        self.update(strategy.mb_y, strategy.mb_output)

    def after_eval(self, strategy: "SupervisedTemplate") -> MetricResult:
        return self._package_result(strategy)

    def _package_result(self, strategy: "SupervisedTemplate") -> MetricResult:
        assert strategy.experience is not None
        exp_cm = self.result()
        phase_name, _ = phase_and_task(strategy)
        stream = stream_type(strategy.experience)
        metric_name = "{}/{}_phase/{}_stream".format(str(self), phase_name, stream)
        plot_x_position = strategy.clock.train_iterations

        if self._save_image:
            class_order = self._get_display_class_order(exp_cm, strategy)

            cm_image = self._image_creator(
                exp_cm[class_order][:, class_order], class_order
            )
            metric_representation = MetricValue(
                self,
                metric_name,
                AlternativeValues(cm_image, exp_cm),
                plot_x_position,
            )
        else:
            metric_representation = MetricValue(
                self, metric_name, exp_cm, plot_x_position
            )

        return [metric_representation]

    def _get_display_class_order(
        self, exp_cm: Tensor, strategy: "SupervisedTemplate"
    ) -> Iterable[int]:
        assert strategy.experience is not None
        benchmark = strategy.experience.benchmark

        if self.absolute_class_order or not isinstance(benchmark, NCScenario):
            return arange(len(exp_cm))

        return benchmark.classes_order

    def __str__(self):
        return "ConfusionMatrix_Stream"


class WandBStreamConfusionMatrix(PluginMetric):
    """
    Confusion Matrix metric compatible with Weights and Biases logger.
    Differently from the `StreamConfusionMatrix`, this metric will use W&B
    built-in functionalities to log the Confusion Matrix.

    This metric may not produce meaningful outputs with other loggers.

    https://docs.wandb.ai/guides/track/log#custom-charts
    """

    def __init__(self, class_names=None):
        """
        :param class_names: list of names for the classes.
            E.g. ["cat", "dog"] if class 0 == "cat" and class 1 == "dog"
            If None, no class names will be used. Default None.
        """

        super().__init__()

        self.outputs = []  # softmax-ed or logits outputs
        self.targets = []  # target classes
        self.class_names = class_names

    def reset(self) -> None:
        self.outputs = []
        self.targets = []

    def before_eval(self, strategy) -> None:
        self.reset()

    def result(self):
        outputs = torch.cat(self.outputs, dim=0)
        targets = torch.cat(self.targets, dim=0)
        return outputs, targets

    def update(self, output, target):
        self.outputs.append(output)
        self.targets.append(target)

    def after_eval_iteration(self, strategy: "SupervisedTemplate"):
        super(WandBStreamConfusionMatrix, self).after_eval_iteration(strategy)
        self.update(strategy.mb_output, strategy.mb_y)

    def after_eval(self, strategy: "SupervisedTemplate") -> MetricResult:
        return self._package_result(strategy)

    def _package_result(self, strategy: "SupervisedTemplate") -> MetricResult:
        assert strategy.experience is not None
        outputs, targets = self.result()
        phase_name, _ = phase_and_task(strategy)
        stream = stream_type(strategy.experience)
        metric_name = "{}/{}_phase/{}_stream".format(str(self), phase_name, stream)
        plot_x_position = strategy.clock.train_iterations

        # compute predicted classes
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        result = wandb.plot.confusion_matrix(
            preds=preds,
            y_true=targets.cpu().numpy(),
            class_names=self.class_names,
        )

        metric_representation = MetricValue(
            self, metric_name, AlternativeValues(result), plot_x_position
        )

        return [metric_representation]

    def __str__(self):
        return "W&BConfusionMatrix_Stream"


def confusion_matrix_metrics(
    num_classes=None,
    normalize=None,
    save_image=True,
    image_creator=default_cm_image_creator,
    class_names=None,
    stream=False,
    wandb=False,
    absolute_class_order: bool = False,
) -> List[PluginMetric]:
    """Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param num_classes: The number of classes. Defaults to None,
        which means that the number of classes will be inferred from
        ground truth and prediction Tensors (see class description for more
        details). If not None, the confusion matrix will always be of size
        `num_classes, num_classes` and only the first `num_classes` values
        of output logits or target logits will be considered in the update.
        If the output or targets are provided as numerical labels,
        there can be no label greater than `num_classes`.
    :param normalize: Normalizes confusion matrix over the true (rows),
        predicted (columns) conditions or all the population. If None,
        confusion matrix will not be normalized. Valid values are: 'true',
        'pred' and 'all' or None.
    :param save_image: If True, a graphical representation of the confusion
        matrix will be logged, too. If False, only the Tensor representation
        will be logged. Defaults to True.
    :param image_creator: A callable that, given the tensor representation
        of the confusion matrix, returns a graphical representation of the
        matrix as a PIL Image. Defaults to `default_cm_image_creator`.
    :param class_names: W&B only. List of names for the classes.
        E.g. ["cat", "dog"] if class 0 == "cat" and class 1 == "dog"
        If None, no class names will be used. Default None.
    :param stream: If True, will return a metric able to log
        the confusion matrix averaged over the entire evaluation stream
        of experiences.
    :param wandb: if True, will return a Weights and Biases confusion matrix
        together with all the other confusion matrixes requested.
    :param absolute_class_order: Not W&B. If true, the labels in the created
        image will be sorted by id, otherwise they will be sorted by order of
        encounter at training time. This parameter is ignored if `save_image` is
        False, or the scenario is not a NCScenario.

    :return: A list of plugin metrics.
    """

    metrics: List[PluginMetric] = []

    if stream:
        metrics.append(
            StreamConfusionMatrix(
                num_classes=num_classes,
                normalize=normalize,
                save_image=save_image,
                image_creator=image_creator,
                absolute_class_order=absolute_class_order,
            )
        )
        if wandb:
            metrics.append(WandBStreamConfusionMatrix(class_names=class_names))

    return metrics


__all__ = [
    "ConfusionMatrix",
    "StreamConfusionMatrix",
    "WandBStreamConfusionMatrix",
    "confusion_matrix_metrics",
]
