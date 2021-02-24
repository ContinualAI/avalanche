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

from typing_extensions import Literal
from typing import Callable, Union, Optional, Dict, Mapping, TYPE_CHECKING

import numpy as np
import torch
from PIL.Image import Image
from torch import Tensor
from torch.nn.functional import pad

from avalanche.evaluation import PluginMetric, Metric
from avalanche.evaluation.metric_results import AlternativeValues, \
    MetricValue, MetricResult
from avalanche.evaluation.metric_utils import default_cm_image_creator
if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy


class ConfusionMatrix(Metric[Tensor]):
    """
    The confusion matrix metric.

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
    Tensors will be used. It is recommended to set the (initial) number of
    classes in the constructor.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an empty Tensor.
    """

    def __init__(self, num_classes: int = None):
        """
        Creates an instance of the confusion matrix metric.

        By default this metric in its initial state will return an empty Tensor.
        The metric can be updated by using the `update` method while the running
        confusion matrix can be retrieved using the `result` method.

        :param num_classes: The initial number of classes. Defaults to None,
            which means that the number of classes will be inferred from
            ground truth and prediction Tensors (see class description for more
            details).
        """
        self._cm_tensor: Optional[Tensor] = None
        """
        The Tensor where the running confusion matrix is stored.
        """
        self._num_classes: Optional[int] = num_classes

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
            raise ValueError('Size mismatch for true_y and predicted_y tensors')

        max_label = -1
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            max_label = max(max_label, predicted_y.shape[1]-1)
            predicted_y = torch.max(predicted_y, 1)[1]
        else:
            # Labels -> check non-negative
            min_label = torch.min(predicted_y).item()
            if min_label < 0:
                raise ValueError('Label values must be non-negative values')

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            max_label = max(max_label, true_y.shape[1]-1)
            true_y = torch.max(true_y, 1)[1]
        else:
            # Labels -> check non-negative
            min_label = torch.min(true_y).item()
            if min_label < 0:
                raise ValueError('Label values must be non-negative values')

        # Initialize or enlarge the confusion matrix
        max_label = max(max_label, max(torch.max(predicted_y).item(),
                                       torch.max(true_y).item()))
        if self._num_classes is not None:
            max_label = max(max_label, self._num_classes-1)

        if max_label < 0:
            raise ValueError('The Confusion Matrix metric can only handle '
                             'positive label values')

        if self._cm_tensor is None:
            # Create the confusion matrix
            self._cm_tensor = torch.zeros((max_label+1, max_label+1),
                                          dtype=torch.long)
        elif max_label >= self._cm_tensor.shape[0]:
            # Enlarge the confusion matrix
            size_diff = 1 + max_label - self._cm_tensor.shape[0]
            self._cm_tensor = pad(self._cm_tensor,
                                  (0, size_diff, 0, size_diff))

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
        return self._cm_tensor

    def reset(self) -> None:
        """
        Resets the metric.

        Calling this method will *not* reset the default number of classes
        optionally defined in the constructor optional parameter.

        :return: None.
        """
        self._cm_tensor = None


class TaskConfusionMatrix(PluginMetric[Tensor]):
    """
    The Confusion Matrix metric.

    This metric logs the confusion matrix for each task at the end of
    each phase. By default this metric computes the matrix on the eval phase
    (on the eval set) only but this behaviour can be changed by passing
    `train=True` in the constructor.

    The metric will log both a Tensor and PIL Image both representing the
    confusion matrices. The Logger will decide which one to use depending on its
    internal implementation.
    """
    def __init__(self, *,
                 train: bool = False,
                 eval: bool = True,
                 num_classes: Union[int, Mapping[int, int]] = None,
                 normalize: Literal['true', 'pred', 'all'] = None,
                 save_image: bool = True,
                 image_creator: Callable[[Tensor], Image] =
                 default_cm_image_creator):
        """
        Creates an instance of the Confusion Matrix metric.

        The train and eval parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the metric will be computed on the training
            phase. Defaults to False.
        :param eval: When True, the metric will be computed on the eval
            phase. Defaults to True.
        :param num_classes: When not None, is used to properly define the
            amount of rows/columns in the confusion matrix. When None, the
            matrix will have many rows/columns as the maximum value of the
            predicted and true pattern labels. Can be either an int, in which
            case the same value will be used across all tasks, or a dictionary
            defining the amount of classes for each task (key = task label,
            value = amount of classes). Defaults to None.
        :param normalize: Normalizes confusion matrix over the true (rows),
            predicted (columns) conditions or all the population. If None,
            confusion matrix will not be normalized. Valid values are: 'true',
            'pred' and 'all'.
        :param save_image: If True, a graphical representation of the confusion
            matrix will be logged, too. If False, only the Tensor representation
            will be logged. Defaults to True.
        :param image_creator: A callable that, given the tensor representation
            of the confusion matrix, returns a graphical representation of the
            matrix as a PIL Image. Defaults to `default_cm_image_creator`.
        """
        super().__init__()

        if not train and not eval:
            raise ValueError('train and eval can\'t be both False at the same'
                             ' time.')

        self._save_image: bool = save_image
        self._task_matrices: Dict[int, ConfusionMatrix] = dict()
        self._num_classes: Optional[Union[int, Mapping[int, int]]] = num_classes
        self._normalize: Optional[Literal['true', 'pred', 'all']] = normalize

        if image_creator is None:
            image_creator = default_cm_image_creator
        self._image_creator: Callable[[Tensor], Image] = image_creator

        self._keep_train_matrix = train
        self._keep_eval_matrix = eval

    def reset(self) -> None:
        self._task_matrices = dict()

    def result(self) -> Dict[int, Tensor]:
        result_dict = dict()
        for task_id in self._task_matrices:
            task_cm = self._task_matrices[task_id].result()
            if self._normalize is not None:
                task_cm = TaskConfusionMatrix._normalize_cm(task_cm,
                                                            self._normalize)
            result_dict[task_id] = task_cm
        return result_dict

    def update(self, true_y: Tensor, predicted_y: Tensor, task_label: int) \
            -> None:

        if task_label not in self._task_matrices:
            self._task_matrices[task_label] = ConfusionMatrix(
                num_classes=self._class_num_for_task(task_label))
        self._task_matrices[task_label].update(true_y, predicted_y)

    def before_training(self, strategy) -> None:
        if self._keep_train_matrix:
            self.reset()

    def before_eval(self, strategy) -> None:
        if self._keep_eval_matrix:
            self.reset()

    def after_training_iteration(self, strategy: 'PluggableStrategy') -> None:
        if self._keep_train_matrix:
            self.update(strategy.mb_y,
                        strategy.logits,
                        strategy.train_task_label)

    def after_eval_iteration(self, strategy: 'PluggableStrategy') -> None:
        if self._keep_eval_matrix:
            self.update(strategy.mb_y,
                        strategy.logits,
                        strategy.train_task_label)

    def after_training(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        if self._keep_train_matrix:
            return self._package_result(strategy)

    def after_eval(self, strategy: 'PluggableStrategy') -> MetricResult:
        if self._keep_eval_matrix:
            return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        phase_name = 'Eval' if strategy.is_eval else 'Train'
        metric_values = []
        for task_label, task_cm in self.result().items():
            metric_name = 'ConfusionMatrix/{}/Task{:03}'.format(phase_name,
                                                                task_label)
            plot_x_position = self._next_x_position(metric_name)

            if self._save_image:
                cm_image = self._image_creator(task_cm)
                metric_representation = MetricValue(
                    self, metric_name, AlternativeValues(cm_image, task_cm),
                    plot_x_position)
            else:
                metric_representation = MetricValue(
                    self, metric_name, task_cm, plot_x_position)
            metric_values.append(metric_representation)
        return metric_values

    def _class_num_for_task(self, task_label: int) -> Optional[int]:
        if self._num_classes is None or isinstance(self._num_classes, int):
            return self._num_classes

        if task_label not in self._num_classes:
            return None

        return self._num_classes[task_label]

    @staticmethod
    def _normalize_cm(cm: Tensor,
                      normalization: Literal['true', 'pred', 'all']):
        if normalization not in ('true', 'pred', 'all'):
            raise ValueError('Invalid normalization parameter. Can be \'true\','
                             ' \'pred\' or \'all\'')

        if normalization == 'true':
            cm = cm / cm.sum(dim=1, keepdim=True, dtype=torch.float64)
        elif normalization == 'pred':
            cm = cm / cm.sum(dim=0, keepdim=True, dtype=torch.float64)
        elif normalization == 'all':
            cm = cm / cm.sum(dtype=torch.float64)
        cm = TaskConfusionMatrix.nan_to_num(cm)
        return cm

    @staticmethod
    def nan_to_num(matrix: Tensor) -> Tensor:
        # if version.parse(torch.__version__) >= version.parse("1.8.0"):
        #    # noinspection PyUnresolvedReferences
        #    return torch.nan_to_num(matrix)

        numpy_ndarray = matrix.numpy()
        numpy_ndarray = np.nan_to_num(numpy_ndarray)
        return torch.tensor(numpy_ndarray, dtype=matrix.dtype)


__all__ = [
    'ConfusionMatrix',
    'TaskConfusionMatrix'
]
