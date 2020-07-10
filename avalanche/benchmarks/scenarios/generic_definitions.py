################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################
from enum import Enum
from typing import TypeVar, Tuple, List, Protocol, runtime_checkable

from avalanche.training.utils import IDatasetWithTargets, DatasetWithTargets


class DatasetPart(Enum):
    """An enumeration defining the different dataset parts"""
    CURRENT = 1  # Classes in this step only
    CUMULATIVE = 2  # Encountered classes (including classes in this step)
    OLD = 3  # Encountered classes (excluding classes in this step)
    FUTURE = 4  # Future classes
    COMPLETE = 5  # All classes (encountered + not seen yet)


class DatasetType(Enum):
    """An enumeration defining the different dataset types"""
    TRAIN = 1  # Training set
    VALIDATION = 2  # Validation (or test) set


TrainSetWithTargets = TypeVar('TrainSetWithTargets', bound=IDatasetWithTargets)
TestSetWithTargets = TypeVar('TestSetWithTargets', bound=IDatasetWithTargets)
MTSingleSet = Tuple[DatasetWithTargets, int]
MTMultipleSet = List[MTSingleSet]

# https://www.python.org/dev/peps/pep-0544/#self-types-in-protocols
TStepInfo = TypeVar('TStepInfo', bound='IStepInfo')


@runtime_checkable
class IStepInfo(Protocol):
    """
    Definition of a learning step. A learning step contains a set of patterns
    which has become available at a particular time instant. The content and
    size of a Step is defined by the specific benchmark that creates the
    IStepInfo instance.

    For instance, a step of a New Classes scenario will contain all patterns
    belonging to a subset of classes of the original training set. A step of a
    New Instance scenario will contain patterns from previously seen classes.

    Steps of  Single Incremental Task (a.k.a. task-free) scenarios are usually
    called "batches" while in Multi Task scenarios a Step is usually associated
    to a "task". Finally, in a Multi Incremental Task scenario the Step may be
    composed by patterns from different tasks.
    """

    # The current step. This is usually an incremental, 0-indexed, value used to
    # keep track of the current batch/task.
    current_step: int

    def current_training_set(self, bucket_classes=False, sort_classes=False,
                             sort_indexes=False) -> MTSingleSet:
        """
        Gets the training set for the current step (batch/task).

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The current step training set, as a tuple containing the
            Dataset and the task label. For SIT scenarios, the task label
            will always be 0.
        """
        ...

    def cumulative_training_sets(self, include_current_step: bool = True,
                                 bucket_classes=False, sort_classes=False,
                                 sort_indexes=False) -> MTMultipleSet:
        """
        Gets the list of cumulative training sets.

        :param include_current_step: If True, include the current step
            training set. Defaults to True.
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The cumulative training sets, as a list. Each element of the
            list is a tuple containing the Dataset and the task label. For SIT
            scenarios, the task label will always be 0.
        """
        ...

    def complete_training_sets(self, bucket_classes=False, sort_classes=False,
                               sort_indexes=False) -> MTMultipleSet:
        """
        Gets the complete list of training sets.

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: All the training sets, as a list. Each element of the list is
            a tuple containing the Dataset and the task label. For SIT
            scenarios, the task label will always be 0.
        """
        ...

    def future_training_sets(self, bucket_classes=False, sort_classes=False,
                             sort_indexes=False) -> MTMultipleSet:
        """
        Gets the "future" training sets. That is, datasets of future steps.

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The future training sets, as a list. Each element of the list
            is a tuple containing the Dataset and the task label. For SIT
            scenarios, the task label will always be 0.
        """
        ...

    def step_specific_training_set(self, step_id: int, bucket_classes=False,
                                   sort_classes=False, sort_indexes=False) \
            -> MTSingleSet:
        """
        Gets the training set of a specific step (batch/task), given its ID.

        :param step_id: The ID of the step.
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The required training set, as a tuple containing the Dataset
            and the task label. For SIT scenarios, the task label will always
            be 0.
        """
        ...

    def training_set_part(self, dataset_part: DatasetPart, bucket_classes=False,
                          sort_classes=False, sort_indexes=False) \
            -> MTMultipleSet:
        """
        Gets the training subset of a specific part of the scenario.

        :param dataset_part: The part of the scenario.
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The training set of the desired part, as a list. Each element
            of the list is a tuple containing the Dataset and the task label.
            For SIT scenarios, the task label will always be 0.
        """
        ...

    def current_test_set(self, bucket_classes=False, sort_classes=False,
                         sort_indexes=False) -> MTSingleSet:
        """
        Gets the test set for the current step (batch/task).

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The current test set, as a tuple containing the Dataset and
            the task label. For SIT scenarios, the task label will always be 0.
        """
        ...

    def cumulative_test_sets(self, include_current_step: bool = True,
                             bucket_classes=False, sort_classes=False,
                             sort_indexes=False) -> MTMultipleSet:
        """
        Gets the list of cumulative test sets (batch/task).

        :param include_current_step: If True, include the current step
            training set. Defaults to True.
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The cumulative test sets, as a list. Each element of the
            list is a tuple containing the Dataset and the task label. For SIT
            scenarios, the task label will always be 0.
        """
        ...

    def complete_test_sets(self, bucket_classes=False, sort_classes=False,
                           sort_indexes=False) -> MTMultipleSet:
        """
        Gets the complete list of test sets.

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: All the test sets, as a list. Each element of the list is a
            tuple containing the Dataset and the task label. For SIT scenarios,
            the task label will always be 0.
        """
        ...

    def future_test_sets(self, bucket_classes=False, sort_classes=False,
                         sort_indexes=False) -> MTMultipleSet:
        """
        Gets the "future" test sets. That is, datasets of future steps.

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The future test sets, as a list. Each element of the list is a
            tuple containing the Dataset and the task label. For SIT scenarios,
            the task label will always be 0.
        """
        ...

    def step_specific_test_set(self, step_id: int, bucket_classes=False,
                               sort_classes=False, sort_indexes=False) \
            -> MTSingleSet:
        """
        Gets the test set of a specific step (batch/task), given its ID.

        :param step_id: The ID of the step (batch/task).
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The required test set, as a tuple containing the Dataset
            and the task label. For SIT scenarios, the task label will always
            be 0.
        """
        ...

    def test_set_part(self, dataset_part: DatasetPart, bucket_classes=False,
                      sort_classes=False, sort_indexes=False) -> MTMultipleSet:
        """
        Gets the test subset of a specific part of the scenario.

        :param dataset_part: The part of the scenario
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The test sets of the desired part, as a list. Each element
            of the list is a tuple containing the Dataset and the task label.
            For SIT scenarios, the task label will always be 0.
        """
        ...

    def disable_transformations(self: TStepInfo) -> TStepInfo:
        """
        Returns a new step info instance in which transformations are disabled.
        The current instance is not affected. This is useful when there is a
        need to access raw data. Can be used when picking and storing
        rehearsal/replay patterns.

        :returns: A new ``IStepInfo`` in which transformations are disabled.
        """
        ...

    def enable_transformations(self: TStepInfo) -> TStepInfo:
        """
        Returns a new step info instance in which transformations are enabled.
        The current instance is not affected. When created the ``IStepInfo``
        instance already has transformations enabled. This method can be used to
        re-enable transformations after a previous call to
        ``disable_transformations()``.

        :returns: A new ``IStepInfo`` in which transformations are enabled.
        """
        ...

    def with_train_transformations(self: TStepInfo) -> TStepInfo:
        """
        Returns a new step info instance in which train transformations are
        applied to both training and test sets. The current instance is not
        affected.

        :returns: A new ``IStepInfo`` in which train transformations are applied
            to both training and test sets.
        """
        ...

    def with_test_transformations(self: TStepInfo) -> TStepInfo:
        """
        Returns a new step info instance in which test transformations are
        applied to both training and test sets. The current instance is
        not affected. This is useful to get the accuracy on the training set
        without considering the usual training data augmentations.

        :returns: A new ``IStepInfo`` in which test transformations are applied
            to both training and test sets.
        """
        ...


__all__ = ['DatasetPart', 'DatasetType', 'TrainSetWithTargets',
           'TestSetWithTargets', 'MTSingleSet', 'MTMultipleSet',
           'IStepInfo', 'TStepInfo']
