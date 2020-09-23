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

import torch
from typing import Sequence, List, Optional, Dict, Generic, Any

from avalanche.benchmarks.scenarios.generic_definitions import \
    TrainSetWithTargets, TestSetWithTargets
from avalanche.training.utils import TransformationSubset


class NCGenericScenario(Generic[TrainSetWithTargets, TestSetWithTargets]):
    """
    This class defines a "New Classes" scenario. It is used when creating both
    task-oriented and single-incremental-batches (a.k.a. task-free) as
    it doesn't make any difference between them. Once created, an instance
    of this class can be iterated in order to obtain the batches/batch sequence
    under the form of instances of :class:`NCGenericBatchInfo`.

    This class can be used directly. However, we recommend using facilities like
    :func:`.scenario_creation.create_nc_single_dataset_sit_scenario`,
    :func:`.scenario_creation.create_nc_single_dataset_multi_task_scenario`,
    :func:`.scenario_creation.create_nc_multi_dataset_sit_scenario` and
    :func:`.scenario_creation.create_nc_multi_dataset_multi_task_scenario`.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, train_dataset: TrainSetWithTargets,
                 test_dataset: TestSetWithTargets,
                 n_batches: int, shuffle: bool = True,
                 seed: Optional[int] = None,
                 fixed_class_order: Optional[Sequence[int]] = None,
                 per_batch_classes: Optional[Dict[int, int]] = None,
                 remap_class_indexes: bool = False,
                 reproducibility_data: Optional[Dict[str, Any]] = None):
        """
        Creates a NCGenericScenario instance given the training and test
        Datasets and the number of batches.

        By default, the number of classes will be automatically detected by
        looking at the training Dataset targets field. Classes will be
        uniformly distributed across the "n_batches" unless a per_task_classes
        argument is specified.

        The number of classes must be divisible without remainder by the number
        of batches. This also applies when the per_task_classes argument is not
        None.

        :param train_dataset: The training dataset. The dataset must contain a
            "targets" field. For instance, one can safely use the datasets from
            the torchvision package.
        :param test_dataset: The test dataset. The dataset must contain a
            "targets" field. For instance, one can safely use the datasets from
            the torchvision package.
        :param n_batches: The number of batches.
        :param shuffle: If True, the class order will be shuffled. Defaults to
            True.
        :param seed: If shuffle is True and seed is not None, the class order
            will be shuffled according to the seed. When None, the current
            PyTorch random number generator state will be used.
            Defaults to None.
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
        :param reproducibility_data: If not None, overrides all the other
            scenario definition options. This is usually a dictionary containing
            data used to reproduce a specific experiment. One can use the
            ``get_reproducibility_data`` method to get (and even distribute)
            the experiment setup so that it can be loaded by passing it as this
            parameter. In this way one can be sure that the same specific
            experimental setup is being used (for reproducibility purposes).
            Beware that, in order to reproduce an experiment, the same train and
            test datasets must be used. Defaults to None.
        """

        # A reference to the full training set
        self.original_train_dataset: TrainSetWithTargets = train_dataset
        # A reference to the full test set
        self.original_test_dataset: TestSetWithTargets = test_dataset

        # The number of batches
        self.n_batches: int
        if reproducibility_data:
            self.n_batches = reproducibility_data['n_batches']
        else:
            self.n_batches = n_batches
        # The class order
        self.classes_order: List[int] = []
        # class_mapping stores the class_list_per_batch such that
        # mapped_class_id = class_mapping[original_class_id]
        self.class_mapping: List[int] = []

        # The classes order (original class IDs)
        self.classes_order_original_ids: List[int] = torch.unique(
            torch.as_tensor(train_dataset.targets),
            sorted=True).tolist()
        all_dataset_classes = len(self.classes_order_original_ids)

        # A list that, for each batch (identified by its index/ID),
        # stores the number of classes assigned to that batch
        self.n_classes_per_batch: List[int] = []
        # A list that, for each batch (identified by its index/ID),
        # stores a list of the IDs of classes assigned to that batch
        self.classes_in_batch: List[List[int]] = []
        # If True, class IDs sequence will be re-mapped to appear
        # as being encountered with an ascending ID
        # (see related __init__ argument).
        self.remap_class_indexes: bool
        if reproducibility_data:
            self.remap_class_indexes = \
                reproducibility_data['remap_class_indexes']
        else:
            self.remap_class_indexes = remap_class_indexes

        if self.n_batches < 1:
            raise ValueError('Invalid number of batches (n_batches parameter): '
                             'must be greater than 0')

        # Note: if fixed_class_order is None and shuffle is False,
        # the class order will be the one encountered
        # By looking at the train_dataset targets field
        if reproducibility_data:
            self.classes_order_original_ids = \
                reproducibility_data['classes_order_original_ids']
        elif fixed_class_order is not None:
            # User defined class order -> just use it
            if len(set(self.classes_order_original_ids).union(
                    set(fixed_class_order))) != \
                    len(self.classes_order_original_ids):
                raise ValueError('Invalid classes defined in fixed_class_order')

            self.classes_order_original_ids = list(fixed_class_order)
        elif shuffle:
            # No user defined class order.
            # If a seed is defined, set the random number generator seed.
            # If no seed has been defined, use the actual
            # random number generator state.
            # Finally, shuffle the class list to obtain a random classes
            # order
            if seed is not None:
                torch.random.manual_seed(seed)
            self.classes_order_original_ids = \
                torch.as_tensor(self.classes_order_original_ids)[
                    torch.randperm(len(self.classes_order_original_ids))
                ].tolist()

        # The number of classes
        self.n_classes: int = len(self.classes_order_original_ids)

        if reproducibility_data:
            self.n_classes_per_batch = \
                reproducibility_data['n_classes_per_batch']
        elif per_batch_classes is not None:
            # per_task_classes is a user-defined dictionary that defines
            # the number of classes to include in some (or all) batches.
            # Remaining classes are equally distributed across the other batches
            #
            # Format of per_task_classes dictionary:
            #   - key = batch id
            #   - value = number of classes for this batch

            if max(per_batch_classes.keys()) >= self.n_batches or min(
                    per_batch_classes.keys()) < 0:
                # The dictionary contains a key (that is, a batch id) >=
                # the number of requested batches... or < 0
                raise ValueError(
                    'Invalid batch id in per_batch_patterns parameter: '
                    'batch ids must be in range [0, n_batches)')
            if min(per_batch_classes.values()) < 0:
                # One or more values (number of classes for each batch) < 0
                raise ValueError('Wrong number of classes defined for one or '
                                 'more batches: must be a non-negative value')

            if sum(per_batch_classes.values()) > self.n_classes:
                # The sum of dictionary values (n. of classes for each batch)
                # >= the number of classes
                raise ValueError('Insufficient number of classes: '
                                 'per_batch_classes parameter can\'t '
                                 'be satisfied')

            # Remaining classes are equally distributed across remaining batches
            # This amount of classes must be be divisible without remainder by
            # the number of remaining batches
            remaining_batches = self.n_batches - len(per_batch_classes)
            if remaining_batches > 0 and (self.n_classes - sum(
                    per_batch_classes.values())) % remaining_batches > 0:
                raise ValueError('Invalid number of batches: remaining classes '
                                 'cannot be divided by n_batches')

            # default_per_batch_classes is the default amount of classes
            # for the remaining batches
            if remaining_batches > 0:
                default_per_batches_classes = (self.n_classes - sum(
                    per_batch_classes.values())) // remaining_batches
            else:
                default_per_batches_classes = 0

            # Initialize the self.n_classes_per_batch list using
            # "default_per_batches_classes" as the default
            # amount of classes per batch. Then, loop through the
            # per_task_classes dictionary to set the customized,
            # user defined, classes for the required batches.
            self.n_classes_per_batch = \
                [default_per_batches_classes] * self.n_batches
            for batch_id in per_batch_classes:
                self.n_classes_per_batch[batch_id] = per_batch_classes[batch_id]
        else:
            # Classes will be equally distributed across the batches
            # The amount of classes must be be divisible without remainder
            # by the number of batches
            if self.n_classes % self.n_batches > 0:
                raise ValueError(
                    'Invalid number of batches: classes contained in dataset '
                    'cannot be divided by n_batches')
            self.n_classes_per_batch = \
                [self.n_classes // self.n_batches] * self.n_batches

        # Before populating the classes_in_batch list,
        # define the correct class IDs class_list_per_batch.
        if reproducibility_data:
            self.classes_order = reproducibility_data['classes_order']
            self.class_mapping = reproducibility_data['class_mapping']
        elif not self.remap_class_indexes:
            self.classes_order = self.classes_order_original_ids
            self.class_mapping = list(range(0, self.n_classes))
        else:
            self.classes_order = list(range(0, self.n_classes))
            self.class_mapping = [-1] * all_dataset_classes
            for class_id in range(self.n_classes):
                if class_id in self.classes_order_original_ids:
                    self.class_mapping[class_id] = \
                        self.classes_order_original_ids.index(class_id)

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

        self.train_steps_patterns_assignment = []
        self.test_steps_patterns_assignment = []
        for batch_id in range(self.n_batches):
            selected_classes = set(self.classes_in_batch[batch_id])
            selected_indexes_train = []
            for idx, element in enumerate(self.train_dataset.targets):
                if element in selected_classes:
                    selected_indexes_train.append(idx)

            selected_indexes_test = []
            for idx, element in enumerate(self.test_dataset.targets):
                if element in selected_classes:
                    selected_indexes_test.append(idx)

            self.train_steps_patterns_assignment.append(selected_indexes_train)
            self.test_steps_patterns_assignment.append(selected_indexes_test)

    def get_reproducibility_data(self):
        reproducibility_data = {
            'class_mapping': self.class_mapping,
            'n_classes_per_batch': self.n_classes_per_batch,
            'remap_class_indexes': bool(self.remap_class_indexes),
            'classes_order': self.classes_order,
            'classes_order_original_ids': self.classes_order_original_ids,
            'n_batches': int(self.n_batches)}
        return reproducibility_data

    def classes_in_batch_range(self, batch_start: int,
                               batch_end: Optional[int] = None) -> List[int]:
        """
        Gets a list of classes contained int the given batches. The batches are
        defined by range. This means that only the classes in range
        [batch_start, batch_end) will be included.

        :param batch_start: The starting batch ID
        :param batch_end: The final batch ID. Can be None, which means that all
            the remaining batches will be taken.

        :returns: The classes contained in the required batch range.
        """
        # Ref: https://stackoverflow.com/a/952952
        if batch_end is None:
            return [
                item for sublist in
                self.classes_in_batch[batch_start:]
                for item in sublist]

        return [
            item for sublist in
            self.classes_in_batch[batch_start:batch_end]
            for item in sublist]

    def get_class_split(self, batch_id: int):
        if batch_id >= 0:
            classes_in_this_batch = \
                self.classes_in_batch[batch_id]
            previous_classes = self.classes_in_batch_range(0, batch_id)
            classes_seen_so_far = \
                previous_classes + classes_in_this_batch
            future_classes = self.classes_in_batch_range(batch_id + 1)
        else:
            classes_in_this_batch = []
            previous_classes = []
            classes_seen_so_far = []
            future_classes = self.classes_in_batch_range(0)

        # Without explicit tuple parenthesis, PEP8 E127 occurs
        return (classes_in_this_batch, previous_classes, classes_seen_so_far,
                future_classes)


__all__ = ['NCGenericScenario']
