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

from __future__ import annotations

import torch
from typing import Sequence, Any, List, Optional, Dict, Generic

from .nc_definitions import DatasetPart, T_train_set_w_targets, \
    T_test_set_w_targets
from .nc_utils import make_transformation_subset
from training.utils.transform_dataset import TransformationSubset, \
    DatasetWithTargets


class NCGenericScenario(Generic[T_train_set_w_targets,
                                T_test_set_w_targets]):
    """
    This class defines a "New Classes" scenario. It is used when creating both
    task-oriented and single-incremental-batches (a.k.a. task-free) as
    it doesn't make any difference between them. Once created, an instance
    of this class can be iterated in order to obtain the batches/batch sequence
    under the form of instances of :class:`NCGenericBatchInfo`.

    This class can be used directly. However, we recommend using facilities like
    :func:`benchmarks.scenarios.create_nc_single_dataset_sit_scenario`,
    :func:`benchmarks.scenarios.create_nc_single_dataset_multi_task_scenario`,
    :func:`benchmarks.scenarios.create_nc_multi_dataset_sit_scenario` and
    :func:`benchmarks.scenarios.create_nc_multi_dataset_multi_task_scenario`.
    """
    def __init__(self, train_dataset: T_train_set_w_targets,
                 test_dataset: T_test_set_w_targets,
                 n_batches: int, shuffle: bool = True,
                 seed: Optional[int] = None,
                 fixed_class_order: Optional[Sequence[int]] = None,
                 per_batch_classes: Optional[Dict[int, int]] = None,
                 remap_class_indexes: bool = False):
        """
        Creates a NCGenericScenario instance given the training and test
        Datasets and the number of batches/batches.

        By default, the number of classes will be automatically detected by
        looking at the training Dataset targets field. Classes will be
        uniformly distributed across the "n_batches" unless a per_task_classes
        argument is specified.

        This scenario manager can be used easily manage transformations.
        For an example of transformations, have a look at:
        https://pytorch.org/docs/stable/torchvision/transforms.html

        The number of classes must be divisible without remainder by the number
        of batches. This also applies when the per_task_classes argument is not
        None.

        :param train_dataset: The training dataset. The dataset must contain a
            "targets" field. For instance one can safely use Datasets from
            the torchvision package.
        :param test_dataset: The test dataset. The dataset must contain a
            "targets" field. For instance one can safely use Datasets from
            the torchvision package.
        :param n_batches: The number of batches/batches.
        :param shuffle: If True, the class order will be shuffled. Defaults to
            True.
        :param seed: If shuffle is True and seed is not None, the class order
            will be shuffled according to the seed. When None, the current
            random number generator will be used. Defaults to None.
        :param fixed_class_order: If not None, the class order to use (overrides
            the shuffle argument). Very useful for enhancing
            reproducibility. Defaults to None.
        :param per_batch_classes: Is not None, a dictionary whose keys are
            (0-indexed) batch IDs and their values are the number of classes
            to include in the respective batches. The dictionary doesn't
            have to contain a key for each batch! All the remaining batches
            will contain an equal amount of the remaining classes. The
            remaining number of classes must be divisible without remainder
            by the remaining number of batches. For instance,
            if you want to include 50 classes in the first batch
            while equally distributing remaining classes across remaining
            batches, just pass the "{0: 50}" dictionary as the
            per_batch_classes parameter. Defaults to None.
        :param remap_class_indexes: If True, original class IDs will be
            remapped so that they will appear as having an ascending order.
            For instance, if the resulting class order after shuffling
            (or defined by fixed_class_order) is [23, 34, 11, 7, 6, ...] and
            remap_class_indexes is True, then all the patterns belonging to
            class 23 will appear as belonging to class "0", class "34" will
            be mapped to "1", class "11" to "2" and so on. This is very
            useful when drawing confusion matrices and when dealing with
            algorithms with dynamic head expansion. Defaults to False.
        """

        # A reference to the full training set
        self.original_train_dataset: T_train_set_w_targets = train_dataset
        # A reference to the full test set
        self.original_test_dataset: T_test_set_w_targets = test_dataset
        # The number of batches
        self.n_batches: int = n_batches
        # The class order
        self.classes_order: List[int] = []
        # class_mapping stores the class_list_per_batch such that
        # mapped_class_id = class_mapping[original_class_id]
        self.class_mapping: List[int] = []
        # The classes order (original class IDs)
        self.classes_order_original_ids: List[int] = torch.unique(
            torch.as_tensor(train_dataset.targets),
            sorted=True).tolist()
        # Training patterns transformation (can be None)
        self.train_transform: Any = None
        # Training targets transformation (can be None)
        self.train_target_transform: Any = None
        # Test patterns transformation (can be None)
        self.test_transform: Any = None
        # Test targets transformation (can be None)
        self.test_target_transform: Any = None
        # A list that, for each batch (identified by its index/ID),
        # stores the number of classes assigned to that batch
        self.n_classes_per_batch: List[int] = []
        # A list that, for each batch (identified by its index/ID),
        # stores a list of the IDs of classes assigned to that batch
        self.classes_in_batch: List[List[int]] = []
        # If True, class IDs sequence will be re-mapped to appear
        # as being encountered with an ascending ID
        # (see related __init__ argument).
        self.remap_class_indexes: bool = remap_class_indexes

        if n_batches < 1:
            raise ValueError('Invalid number of batches (n_batches parameter): '
                             'must be greater than 0')

        # Note: if fixed_class_order is None and shuffle is False,
        # the class order will be the one encountered
        # By looking at the train_dataset targets field
        if fixed_class_order is not None:
            # User defined class order -> just use it
            self.classes_order_original_ids = list(fixed_class_order)
        elif shuffle:
            # No user defined class order.
            # If a seed is defined, set the random number generator seed.
            # If no seed has been defined, use the actual
            # random number generator state.
            # Finally, shuffle the class list to obtain a random classes order
            if seed is not None:
                torch.random.manual_seed(seed)
            self.classes_order_original_ids = \
                torch.as_tensor(self.classes_order_original_ids)[
                    torch.randperm(len(self.classes_order_original_ids))
                ].tolist()

        # The number of classes
        self.n_classes: int = len(self.classes_order_original_ids)

        if per_batch_classes is not None:
            # per_task_classes is a user-defined dictionary that defines
            # the number of classes to include in some (or all) batches.
            # Remaining classes are equally distributed across the other batches
            #
            # Format of per_task_classes dictionary:
            #   - key = batch id
            #   - value = number of classes for this batch

            if max(per_batch_classes.keys()) >= n_batches or min(
                    per_batch_classes.keys()) < 0:
                # The dictionary contains a key (that is, a batch id) >=
                # the number of requested batches... or < 0
                raise ValueError(
                    'Invalid batch id in per_task_classes parameter: '
                    'batch ids must be in range [0, n_batches)')
            if min(per_batch_classes.values()) < 0:
                # One or more values (number of classes for each batch) < 0
                raise ValueError('Wrong number of classes defined for one or '
                                 'more batches: must be a non-negative value')

            if sum(per_batch_classes.values()) > self.n_classes:
                # The sum of dictionary values (n. of classes for each batch)
                # >= the number of classes
                raise ValueError('Insufficient number of classes: '
                                 'per_task_classes parameter can\'t '
                                 'be satisfied')

            # Remaining classes are equally distributed across remaining batches
            # This amount of classes must be be divisible without remainder by
            # the number of remaining batches
            remaining_batches = n_batches - len(per_batch_classes)
            if remaining_batches > 0 and (self.n_classes - sum(
                    per_batch_classes.values())) % remaining_batches > 0:
                raise ValueError('Invalid number of batches: remaining classes '
                                 'cannot be divided by n_batches')

            # default_per_batch_classes is the default amount of classes
            # for the remaining batches
            default_per_batches_classes = (self.n_classes - sum(
                per_batch_classes.values())) // remaining_batches

            # Initialize the self.n_classes_per_batch list using
            # "default_per_batches_classes" as the default
            # amount of classes per batch. Then, loop through the
            # per_task_classes dictionary to set the customized,
            # user defined, classes for the required batches.
            self.n_classes_per_batch = [default_per_batches_classes] * n_batches
            for batch_id in per_batch_classes:
                self.n_classes_per_batch[batch_id] = per_batch_classes[batch_id]
        else:
            # Classes will be equally distributed across the batches
            # The amount of classes must be be divisible without remainder
            # by the number of batches
            if self.n_classes % n_batches > 0:
                raise ValueError(
                    'Invalid number of batches: classes contained in dataset '
                    'cannot be divided by n_batches')
            self.n_classes_per_batch = [self.n_classes // n_batches] * n_batches

        # Before populating the classes_in_batch list,
        # define the correct class IDs class_list_per_batch.
        if not self.remap_class_indexes:
            self.classes_order = self.classes_order_original_ids
            self.class_mapping = list(range(0, self.n_classes))
        else:
            self.classes_order = list(range(0, self.n_classes))
            self.class_mapping = [
                self.classes_order_original_ids.index(class_id)
                for class_id in range(self.n_classes)]
        self.train_dataset = TransformationSubset(
            train_dataset, None, class_mapping=self.class_mapping)
        self.test_dataset = TransformationSubset(
            test_dataset, None, class_mapping=self.class_mapping)

        # Populate the classes_in_batch list
        # "classes_in_batch[batch_id]": list of class IDs assigned
        # to batch "batch_id"
        for batch_id in range(self.n_batches):
            classes_start_idx = sum(self.n_classes_per_batch[:batch_id])
            classes_end_idx = classes_start_idx + self.n_classes_per_batch[
                batch_id]

            self.classes_in_batch.append(
                self.classes_order[classes_start_idx:classes_end_idx])

        # Stea transforms from the datasets, that is, copy the reference to the
        # transformation functions, and set to None the fields in the
        # respective Dataset instances. This will allow us to disable
        # transformations (useful while managing rehearsal) or even apply test
        # transforms to train patterns (useful when if testing on the training
        # sets, as test transforms usually don't contain data augmentation
        # transforms)
        if hasattr(train_dataset, 'transform') and \
                train_dataset.transform is not None:
            self.train_transform = train_dataset.transform
            train_dataset.transform = None
        if hasattr(train_dataset, 'target_transform') and \
                train_dataset.target_transform is not None:
            self.train_target_transform = train_dataset.target_transform
            train_dataset.target_transform = None

        if hasattr(test_dataset, 'transform') and \
                test_dataset.transform is not None:
            self.test_transform = test_dataset.transform
            test_dataset.transform = None
        if hasattr(test_dataset, 'target_transform') and \
                test_dataset.target_transform is not None:
            self.test_target_transform = test_dataset.target_transform
            test_dataset.target_transform = None

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, batch_idx) -> \
            NCGenericBatchInfo[T_train_set_w_targets, T_test_set_w_targets]:
        return NCGenericBatchInfo(self, current_batch=batch_idx)


class NCGenericBatchInfo(Generic[T_train_set_w_targets, T_test_set_w_targets]):
    """
    Defines a "New Classes" batch. It contains methods to obtain the current,
    previous, cumulative and future training and test sets. It also defines
    fields that can be used to check which classes are in this batch,
    the previously encountered ones and the future ones.

    It keeps a reference to the original :class:`NCGenericScenario` which can
    be used to retrieve data about the NC scenario.
    """

    def __init__(self, scenario: NCGenericScenario[T_train_set_w_targets,
                                                   T_test_set_w_targets],
                 force_train_transformations: bool = False,
                 force_test_transformations: bool = False,
                 are_transformations_disabled: bool = False,
                 current_batch: int = -1):
        """
        Creates an instance given the root scenario. Instances of this class are
        usually created automatically while iterating over an instance of
        :class:`NCGenericScenario`.

        :param scenario: A reference to the global NC scenario
        :param force_train_transformations: If True, train transformations will
            be applied to the test set too. The ``force_test_transformations``
            parameter can't be True at the same time. Defaults to False.
        :param force_test_transformations: If True, test transformations will be
            applied to the training set too. The ``force_train_transformations``
            parameter can't be True at the same time. Defaults to False.
        :param are_transformations_disabled: If True, transformations are
            disabled. That is, patterns and targets will be returned as
            outputted by  the original training and test Datasets. Overrides
            ``force_train_transformations`` and ``force_test_transformations``.
            Defaults to False.
        :param current_batch: Defines the current batch ID. Defaults to -1.
        """

        # The current batch ID
        self.current_batch: int = current_batch
        # The reference to the NCGenericScenario
        self.scenario: NCGenericScenario[T_train_set_w_targets,
                                         T_test_set_w_targets] = scenario

        self.force_train_transformations = force_train_transformations
        self.force_test_transformations = force_test_transformations
        self.are_transformations_disabled = are_transformations_disabled

        # are_transformations_disabled can be True without constraints
        if self.force_test_transformations and self.force_train_transformations:
            raise ValueError(
                'Error in force_train/test_transformations arguments.'
                'Can\'t be both True.')

        # List of classes of current and previous batches,
        # in their encounter order
        self.classes_seen_so_far: List[int] = []

        # The list of classes in this batch
        self.classes_in_this_batch: List[int] = []

        # The list of classes in previous batches, in their encounter order
        self.previous_classes: List[int] = []

        # The list of classes of next batches, in their encounter order
        self.future_classes: List[int] = []

        # _go_to_batch initializes the above lists
        self._go_to_batch()

    # Training set utils
    def current_training_set(self, bucket_classes=False, sort_classes=False,
                             sort_indexes=False) -> DatasetWithTargets:
        """
        Gets the training set for the current batch

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The current batch training set, as a Dataset.
        """
        return self.__make_train_subset(self.classes_in_this_batch,
                                        bucket_classes, sort_classes,
                                        sort_indexes)

    def cumulative_training_set(self, include_current_batch: bool = True,
                                bucket_classes=False, sort_classes=False,
                                sort_indexes=False) -> DatasetWithTargets:
        """
        Gets the cumulative training set

        :param include_current_batch: If True, include the current batch
            training set. Defaults to True.
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The cumulative training set, as a Dataset.
        """
        if include_current_batch:
            return self.__make_train_subset(self.classes_seen_so_far,
                                            bucket_classes, sort_classes,
                                            sort_indexes)
        else:
            return self.__make_train_subset(self.previous_classes,
                                            bucket_classes, sort_classes,
                                            sort_indexes)

    def complete_training_set(self, bucket_classes=False, sort_classes=False,
                              sort_indexes=False) -> DatasetWithTargets:
        """
        Gets the complete training set

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The complete training set, as a Dataset.
        """
        return self.__make_train_subset(self.scenario.classes_order,
                                        bucket_classes, sort_classes,
                                        sort_indexes)

    def future_training_set(self, bucket_classes=False, sort_classes=False,
                            sort_indexes=False) \
            -> DatasetWithTargets:
        """
        Gets the "future" training set. That is, a dataset made of training
        patterns belonging to not-already-encountered classes.

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The "future" training set, as a Dataset.
        """
        return self.__make_train_subset(self.future_classes, bucket_classes,
                                        sort_classes, sort_indexes)

    def batch_specific_training_set(self, batch_id: int, bucket_classes=False,
                                    sort_classes=False, sort_indexes=False) \
            -> DatasetWithTargets:
        """
        Gets the training set of a specific batch, given its ID.

        :param batch_id: The ID of the batch
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The batch specific training set, as a Dataset.
        """
        classes_in_required_batch = self.scenario.classes_in_batch[batch_id]
        return self.__make_train_subset(classes_in_required_batch,
                                        bucket_classes, sort_classes,
                                        sort_indexes)

    def training_set_part(self, dataset_part: DatasetPart, bucket_classes=False,
                          sort_classes=False, sort_indexes=False) \
            -> DatasetWithTargets:
        """
        Gets the training subset of a specific part of the scenario.

        :param dataset_part: The part of the scenario
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The training set of the desired specific part, as a Dataset.
        """
        if dataset_part == DatasetPart.CURRENT:
            return self.current_training_set(bucket_classes=bucket_classes,
                                             sort_classes=sort_classes,
                                             sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.CUMULATIVE:
            return self.cumulative_training_set(include_current_batch=True,
                                                bucket_classes=bucket_classes,
                                                sort_classes=sort_classes,
                                                sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.OLD:
            return self.cumulative_training_set(include_current_batch=False,
                                                bucket_classes=bucket_classes,
                                                sort_classes=sort_classes,
                                                sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.FUTURE:
            return self.future_training_set(bucket_classes=bucket_classes,
                                            sort_classes=sort_classes,
                                            sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.COMPLETE:
            return self.complete_training_set(bucket_classes=bucket_classes,
                                              sort_classes=sort_classes,
                                              sort_indexes=sort_indexes)
        else:
            raise ValueError('Unsupported dataset part')

    # Test set utils
    def current_test_set(self, bucket_classes=False, sort_classes=False,
                         sort_indexes=False) \
            -> DatasetWithTargets:
        """
        Gets the test set for the current batch

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The current batch test set, as a Dataset.
        """
        return self.__make_test_subset(self.classes_in_this_batch,
                                       bucket_classes, sort_classes,
                                       sort_indexes)

    def cumulative_test_set(self, include_current_batch: bool = True,
                            bucket_classes=False, sort_classes=False,
                            sort_indexes=False) -> DatasetWithTargets:
        """
        Gets the cumulative test set

        :param include_current_batch: If True, include the current batch
            training set. Defaults to True.
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The cumulative test set, as a Dataset.
        """
        selected_classes = self.classes_seen_so_far if include_current_batch \
            else self.previous_classes
        return self.__make_test_subset(selected_classes, bucket_classes,
                                       sort_classes, sort_indexes)

    def complete_test_set(self, bucket_classes=False, sort_classes=False,
                          sort_indexes=False) -> DatasetWithTargets:
        """
        Gets the complete test set

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The complete test set, as a Dataset.
        """
        return self.__make_test_subset(self.scenario.classes_order,
                                       bucket_classes, sort_classes,
                                       sort_indexes)

    def future_test_set(self, bucket_classes=False, sort_classes=False,
                        sort_indexes=False) -> DatasetWithTargets:
        """
        Gets the "future" test set. That is, a dataset made of training patterns
        belonging to not-already-encountered classes.

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The "future" test set, as a Dataset.
        """
        return self.__make_test_subset(self.future_classes, bucket_classes,
                                       sort_classes, sort_indexes)

    def batch_specific_test_set(self, batch_id: int, bucket_classes=False,
                                sort_classes=False, sort_indexes=False) \
            -> DatasetWithTargets:
        """
        Gets the test set of a specific batch, given its ID.

        :param batch_id: The ID of the batch
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The batch specific test set, as a Dataset.
        """
        return self.__make_test_subset(self.scenario.classes_in_batch[batch_id],
                                       bucket_classes, sort_classes,
                                       sort_indexes)

    def test_set_part(self, dataset_part: DatasetPart, bucket_classes=False,
                      sort_classes=False, sort_indexes=False) \
            -> DatasetWithTargets:
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

        :returns: The test set of the desired specific part, as a Dataset.
        """
        if dataset_part == DatasetPart.CURRENT:
            return self.current_test_set(bucket_classes=bucket_classes,
                                         sort_classes=sort_classes,
                                         sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.CUMULATIVE:
            return self.cumulative_test_set(include_current_batch=True,
                                            bucket_classes=bucket_classes,
                                            sort_classes=sort_classes,
                                            sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.OLD:
            return self.cumulative_test_set(include_current_batch=False,
                                            bucket_classes=bucket_classes,
                                            sort_classes=sort_classes,
                                            sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.FUTURE:
            return self.future_test_set(bucket_classes=bucket_classes,
                                        sort_classes=sort_classes,
                                        sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.COMPLETE:
            return self.complete_test_set(bucket_classes=bucket_classes,
                                          sort_classes=sort_classes,
                                          sort_indexes=sort_indexes)
        else:
            raise ValueError('Unsupported dataset part')

    def disable_transformations(self) -> NCGenericBatchInfo[
        T_train_set_w_targets, T_test_set_w_targets
    ]:
        """
        Returns a new batch info instance in which transformations are disabled.
        The current instance is not affected. This is useful when there is a
        need to access raw data. Can be used when picking and storing
        rehearsal/replay patterns.

        :returns: A new NCGenericBatchInfo in which transformations are
        disabled.
        """
        return NCGenericBatchInfo(
            self.scenario,
            force_test_transformations=self.force_test_transformations,
            force_train_transformations=self.force_train_transformations,
            are_transformations_disabled=True,
            current_batch=self.current_batch)

    def enable_transformations(self) -> NCGenericBatchInfo[
        T_train_set_w_targets, T_test_set_w_targets
    ]:
        """
        Returns a new batch info instance in which transformations are enabled.
        The current instance is not affected. When created, the
        NCGenericBatchInfo instance already has transformations enabled.
        This method can be used to re-enable transformations after a previous
        call to ``disable_transformations``.

        :returns: A new NCGenericBatchInfo in which transformations are
        enabled.
        """
        return NCGenericBatchInfo(
            self.scenario,
            force_test_transformations=self.force_test_transformations,
            force_train_transformations=self.force_train_transformations,
            are_transformations_disabled=False,
            current_batch=self.current_batch)

    def with_train_transformations(self) -> NCGenericBatchInfo[
        T_train_set_w_targets, T_test_set_w_targets
    ]:
        """
        Returns a new batch info instance in which train transformations are
        applied to both training and test sets. The current instance is not
        affected.

        :returns: A new NCGenericBatchInfo in which train transformations
        are applied to both training and test sets.
        """
        return NCGenericBatchInfo(
            self.scenario,
            force_test_transformations=False,
            force_train_transformations=True,
            are_transformations_disabled=self.are_transformations_disabled,
            current_batch=self.current_batch)

    def with_test_transformations(self) -> NCGenericBatchInfo[
        T_train_set_w_targets, T_test_set_w_targets
    ]:
        """
        Returns a new batch info instance in which test transformations are
        applied to both training and test sets. The current instance is
        not affected. This is useful to get the accuracy on the training set
        without considering the usual training data augmentations.

        :returns: A new NCGenericBatchInfo in which test transformations
        are applied to both training and test sets.
        """
        return NCGenericBatchInfo(
            self.scenario,
            force_test_transformations=True,
            force_train_transformations=False,
            are_transformations_disabled=self.are_transformations_disabled,
            current_batch=self.current_batch)

    def __get_batches_classes(self, batch_start: int,
                              batch_end: Optional[int] = None) -> List[int]:
        # Ref: https://stackoverflow.com/a/952952
        if batch_end is None:
            return [item for sublist in
                    self.scenario.classes_in_batch[batch_start:] for item in
                    sublist]
        else:
            return [item for sublist
                    in self.scenario.classes_in_batch[batch_start:batch_end]
                    for item in sublist]

    def _go_to_batch(self):
        if self.current_batch >= 0:
            self.classes_in_this_batch = self.scenario.classes_in_batch[
                self.current_batch]
            self.previous_classes = self.__get_batches_classes(
                0, self.current_batch)
            self.classes_seen_so_far = \
                self.previous_classes + self.classes_in_this_batch
            self.future_classes = self.__get_batches_classes(self.current_batch)
        else:
            self.classes_in_this_batch = []
            self.previous_classes = []
            self.classes_seen_so_far = []
            self.future_classes = self.__get_batches_classes(0)

    def __make_subset(self, is_train, batch_classes, bucket_classes: bool,
                      sort_classes: bool, sort_indexes: bool):

        if self.are_transformations_disabled:
            patterns_transformation = None
            targets_transformation = None
        elif self.force_test_transformations:
            patterns_transformation = \
                self.scenario.test_transform
            targets_transformation = \
                self.scenario.test_target_transform
        else:
            patterns_transformation = \
                self.scenario.train_transform if is_train \
                else self.scenario.test_transform
            targets_transformation = \
                self.scenario.train_target_transform if is_train \
                else self.scenario.test_target_transform

        dataset = self.scenario.train_dataset if is_train \
            else self.scenario.test_dataset

        return make_transformation_subset(dataset,
                                          patterns_transformation,
                                          targets_transformation,
                                          batch_classes,
                                          bucket_classes=bucket_classes,
                                          sort_classes=sort_classes,
                                          sort_indexes=sort_indexes)

    def __make_train_subset(self, batch_classes, bucket_classes: bool,
                            sort_classes: bool, sort_indexes: bool):
        return self.__make_subset(
            True, batch_classes, bucket_classes=bucket_classes,
            sort_classes=sort_classes, sort_indexes=sort_indexes)

    def __make_test_subset(self, batch_classes, bucket_classes: bool,
                           sort_classes: bool, sort_indexes: bool):
        return self.__make_subset(
            False, batch_classes, bucket_classes=bucket_classes,
            sort_classes=sort_classes, sort_indexes=sort_indexes)


__all__ = ['NCGenericScenario', 'NCGenericBatchInfo']
