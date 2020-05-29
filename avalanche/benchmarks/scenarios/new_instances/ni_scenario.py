################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 28-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

from typing import Optional, List, Generic, Any, Union, Tuple, Iterable, \
    Sequence

import torch

from avalanche.benchmarks.scenarios.generic_definitions import \
    T_train_set_w_targets, T_test_set_w_targets, DatasetPart, MTSingleSet, \
    MTMultipleSet
from avalanche.training.utils import TransformationSubset
from .ni_utils import make_ni_transformation_subset


class NIScenario(Generic[T_train_set_w_targets, T_test_set_w_targets]):
    """
    This class defines a "New Instance" Single Incremental Task scenario.
    Once created, an instance of this class can be iterated in order to obtain
    the batch sequence under the form of instances of :class:`NIBatchInfo`.

    Instances of this class can be created using the constructor directly.
    However, we recommend using facilities like:
    :func:`.scenario_creation.create_ni_single_dataset_sit_scenario` and
    :func:`.scenario_creation.create_ni_multi_dataset_sit_scenario`.

    Being a Single Incremental Task scenario, the task label will always be "0".
    Also, consider that every method from :class:`NIBatchInfo` used to retrieve
    parts of the test set (past, current, furure, cumulative) always return the
    complete test set. That is, they behave as the getter for the complete test
    set. These methods are left for compatibility with the ones found in the
    :class:`avalanche.benchmarks.scenarios.new_classes.NCBatchInfo` scenario.
    """

    def __init__(
            self, train_dataset: T_train_set_w_targets,
            test_dataset: T_test_set_w_targets, n_batches: int,
            shuffle: bool = True, seed: Optional[int] = None,
            balance_batches: bool = False,
            min_class_patterns_in_batch: int = 0,
            fixed_batch_assignment: Optional[Sequence[Sequence[int]]] = None):
        """
        Creates a NIScenario instance given the training and test Datasets and
        the number of batches.

        :param train_dataset: The training dataset. The dataset must contain a
            "targets" field. For instance, one can safely use the datasets from
            the torchvision package.
        :param test_dataset: The test dataset. The dataset must contain a
            "targets" field. For instance, one can safely use the datasets from
            the torchvision package.
        :param n_batches: The number of batches.
        :param shuffle: If True, the patterns order will be shuffled. Defaults
            to True.
        :param seed: If shuffle is True and seed is not None, the class order
            will be shuffled according to the seed. When None, the current
            PyTorch random number generator state will be used.
            Defaults to None.
        :param balance_batches: If True, pattern of each class will be equally
            spread across all batches. If False, patterns will be assigned to
            batches in a complete random way. Defaults to False.
        :param min_class_patterns_in_batch: The minimum amount of patterns of
            every class that must be assigned to every batch. Compatible with
            the ``balance_batches`` parameter. An exception will be raised if
            this constraint can't be satisfied. Defaults to 0.
        :param fixed_batch_assignment: If not None, the pattern assignment
            to use. It must be a list with an entry for each batch. Each entry
            is a list that contains the indexes of patterns belonging to that
            batch. Overrides the ``shuffle``, ``balance_batches`` and
            ``min_class_patterns_in_batch`` parameters.
        """

        # A reference to the full training set
        self.train_dataset: T_train_set_w_targets = train_dataset
        # A reference to the full test set
        self.test_dataset: T_test_set_w_targets = test_dataset
        # The number of batches
        self.n_batches: int = n_batches
        # Training patterns transformation (can be None)
        self.train_transform: Any = None
        # Training targets transformation (can be None)
        self.train_target_transform: Any = None
        # Test patterns transformation (can be None)
        self.test_transform: Any = None
        # Test targets transformation (can be None)
        self.test_target_transform: Any = None

        if n_batches < 1:
            raise ValueError('Invalid number of batches (n_batches parameter): '
                             'must be greater than 0')

        if min_class_patterns_in_batch < 0:
            raise ValueError('Invalid min_class_patterns_in_batch parameter: '
                             'must be greater than or equal to 0')

        unique_targets, unique_count = torch.unique(
            torch.as_tensor(train_dataset.targets), return_counts=True)

        # The number of classes
        self.n_classes: int = len(unique_targets)

        # n_patterns_per_class contains the number of patterns for each class
        self.n_patterns_per_class: List[int] = \
            [0 for _ in range(self.n_classes)]

        if fixed_batch_assignment:
            included_patterns = list()
            for batch_def in fixed_batch_assignment:
                included_patterns.extend(batch_def)
            subset = TransformationSubset(train_dataset, included_patterns)
            unique_targets, unique_count = torch.unique(
                torch.as_tensor(subset.targets), return_counts=True)

        for unique_idx in range(len(unique_targets)):
            class_id = int(unique_targets[unique_idx])
            class_count = (unique_count[unique_idx])
            self.n_patterns_per_class[class_id] = class_count

        # The number of patterns in each batch
        self.n_patterns_per_batch: List[int] = []
        # self.batch_structure[batch_id][class_id] is the amount of patterns
        # of class "class_id" in batch "batch_id
        self.batch_structure: List[List[int]] = []
        # batch_patterns contains, for each batch, the list of patterns
        # assigned to that batch (as indexes of elements from the training set)
        self.batch_patterns: List[List[int]] = []

        if fixed_batch_assignment:
            # fixed_patterns_assignment is the user provided self.batch_patterns
            # all we have to do is populate remaining fields of the class!
            # n_patterns_per_batch is filled later based on batch_structure
            # so we only need to fill batch_structure.
            self.batch_patterns = fixed_batch_assignment
            self.batch_structure = [[0 for _ in range(self.n_classes)]
                                    for _ in range(self.n_batches)]

            for batch_id in range(self.n_batches):
                batch_targets = [train_dataset.targets[pattern_idx]
                                 for pattern_idx in
                                 self.batch_patterns[batch_id]]
                cls_ids, cls_counts = torch.unique(torch.as_tensor(
                    batch_targets), return_counts=True)

                for unique_idx in range(len(cls_ids)):
                    self.batch_structure[batch_id][int(cls_ids[unique_idx])] +=\
                        int(cls_counts[unique_idx])
        else:
            # All batches will all contain the same amount of patterns
            # The amount of patterns doesn't need to be divisible without
            # remainder by the number of batches, so we distribute remaining
            # patterns across randomly selected batches (when shuffling) or
            # the first N batches (when not shuffling). However, we first have
            # to check if the min_class_patterns_in_batch constraint is
            # satisfiable.
            min_class_patterns = min(self.n_patterns_per_class)
            if min_class_patterns < n_batches * min_class_patterns_in_batch:
                raise ValueError('min_class_patterns_in_batch constraint '
                                 'can\'t be satisfied')

            if seed is not None:
                torch.random.manual_seed(seed)

            # First, get the patterns indexes for each class
            targets_as_tensor = torch.as_tensor(train_dataset.targets)
            classes_to_patterns_idx = [
                (targets_as_tensor == class_id).nonzero().view(-1).tolist()
                for class_id in range(self.n_classes)
            ]

            if shuffle:
                classes_to_patterns_idx = [
                    torch.as_tensor(cls_patterns)[
                        torch.randperm(len(cls_patterns))
                    ].tolist() for cls_patterns in classes_to_patterns_idx
                ]

            # Here we assign patterns to each batch. Two different strategies
            # are required in order to manage the balance_batches parameter.
            if balance_batches:
                # If balance_batches is True we have to make sure that patterns
                # of each class are equally distributed across batches.
                #
                # To do this, populate self.batch_structure, which will
                # describe how many patterns of each class are assigned to each
                # batch. Then, for each batch, assign the required amount of
                # patterns of each class.
                #
                # We already checked that there are enough patterns for each
                # class to satisfy the min_class_patterns_in_batch param so here
                # we don't need to explicitly enforce that constraint.

                # First, count how many patterns of each class we have to assign
                # to all the batches (avg). We also get the number of remaining
                # patterns which we'll have to assign in a second step.
                class_patterns_per_batch = [
                    ((n_class_patterns // self.n_batches),
                     (n_class_patterns % self.n_batches))
                    for n_class_patterns in self.n_patterns_per_class
                ]

                # Remember: batch_structure[batch_id][class_id] is the amount of
                # patterns of class "class_id" in batch "batch_id"
                #
                # This is the easier step: just assign the average amount of
                # class patterns to each batch.
                self.batch_structure = [
                    [class_patterns_this_batch[0]
                     for class_patterns_this_batch
                     in class_patterns_per_batch] for _ in range(self.n_batches)
                ]

                # Now we have to distribute the remaining patterns of each class
                #
                # This means that, for each class, we can (randomly) select
                # "n_class_patterns % self.n_batches" batches to assign a single
                # additional pattern of that class.
                for class_id in range(self.n_classes):
                    n_remaining = class_patterns_per_batch[class_id][1]
                    if n_remaining == 0:
                        continue
                    if shuffle:
                        assignment_of_remaining_patterns = torch.randperm(
                            self.n_batches).tolist()[:n_remaining]
                    else:
                        assignment_of_remaining_patterns = range(n_remaining)
                    for batch_id in assignment_of_remaining_patterns:
                        self.batch_structure[batch_id][class_id] += 1

                # Following the self.batch_structure definition, assign
                # the actual patterns to each batch.
                #
                # For each batch we assign exactly
                # self.batch_structure[batch_id][class_id] patterns of
                # class "class_id"
                self.batch_patterns = [[] for _ in range(self.n_batches)]
                next_idx_per_class = [0 for _ in range(self.n_classes)]
                for batch_id in range(self.n_batches):
                    for class_id in range(self.n_classes):
                        end_idx = next_idx_per_class[class_id] + \
                            self.batch_structure[batch_id][class_id]
                        self.batch_patterns[batch_id].extend(
                            classes_to_patterns_idx[class_id][
                                next_idx_per_class:end_idx]
                        )
                        next_idx_per_class[class_id] = end_idx
            else:
                # If balance_batches if False, we just randomly shuffle the
                # patterns indexes and pick N patterns for each batch.
                #
                # However, we have to enforce the min_class_patterns_in_batch
                # constraint, which makes things difficult.
                # In the balance_batches scenario, that constraint is implicitly
                # enforced by equally distributing class patterns in each batch
                # (we already checked that there are enough overall patterns
                # for each class to satisfy min_class_patterns_in_batch)

                # Here we have to assign the minimum required amount of class
                # patterns to each batch first, then we can move to randomly
                # assign the remaining patterns to each batch.

                # First, initialize batch_patterns and batch_structure
                self.batch_patterns = [[] for _ in range(self.n_batches)]
                self.batch_structure = [[0 for _ in range(self.n_classes)]
                                        for _ in range(self.n_batches)]

                # For each batch we assign exactly
                # min_class_patterns_in_batch patterns from each class
                #
                # Very similar to the loop found in the balance_batches branch!
                # Remember that classes_to_patterns_idx is already shuffled
                # (if required)
                next_idx_per_class = [0 for _ in range(self.n_classes)]
                remaining_patterns = set(range(len(train_dataset)))

                for batch_id in range(self.n_batches):
                    for class_id in range(self.n_classes):
                        next_idx = next_idx_per_class[class_id]
                        end_idx = next_idx + min_class_patterns_in_batch
                        selected_patterns = \
                            classes_to_patterns_idx[next_idx:end_idx]
                        self.batch_patterns[batch_id].extend(selected_patterns)
                        self.batch_structure[batch_id][class_id] += \
                            min_class_patterns_in_batch
                        remaining_patterns.difference_update(selected_patterns)
                        next_idx_per_class[class_id] = end_idx

                # We have assigned the required min_class_patterns_in_batch,
                # now we assign the remaining patterns
                #
                # We'll work on remaining_patterns, which contains indexes of
                # patterns not assigned in the previous step.
                if shuffle:
                    patterns_order = torch.as_tensor(remaining_patterns)[
                        torch.randperm(len(remaining_patterns))
                    ].tolist()
                else:
                    patterns_order = remaining_patterns
                targets_order = [train_dataset.targets[pattern_idx]
                                 for pattern_idx in patterns_order]

                avg_batch_size = len(patterns_order) // self.n_batches
                n_remaining = len(patterns_order) % self.n_batches
                prev_idx = 0
                for batch_id in range(self.n_batches):
                    next_idx = prev_idx + avg_batch_size
                    self.batch_patterns[batch_id].extend(
                        patterns_order[prev_idx:next_idx])
                    cls_ids, cls_counts = torch.unique(torch.as_tensor(
                        targets_order[prev_idx:next_idx]), return_counts=True)

                    cls_ids = cls_ids.tolist()
                    cls_counts = cls_counts.tolist()

                    for unique_idx in range(len(cls_ids)):
                        self.batch_structure[batch_id][cls_ids[unique_idx]] += \
                            cls_counts[unique_idx]
                    prev_idx = next_idx

                # Distribute remaining patterns
                if n_remaining > 0:
                    if shuffle:
                        assignment_of_remaining_patterns = torch.randperm(
                            self.n_batches).tolist()[:n_remaining]
                    else:
                        assignment_of_remaining_patterns = range(n_remaining)
                    for batch_id in assignment_of_remaining_patterns:
                        pattern_idx = patterns_order[prev_idx]
                        pattern_target = targets_order[prev_idx]
                        self.batch_patterns[batch_id].append(pattern_idx)

                        self.batch_structure[batch_id][pattern_target] += 1
                        prev_idx += 1

        self.n_patterns_per_batch = [len(self.batch_patterns[batch_id])
                                     for batch_id in range(self.n_batches)]

        # Steal transforms from the datasets, that is, copy the reference to the
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
            'NIBatchInfo[T_train_set_w_targets, T_test_set_w_targets]':
        return NIBatchInfo(self, current_batch=batch_idx)


class NIBatchInfo(Generic[T_train_set_w_targets, T_test_set_w_targets]):
    """
    Defines a "New Instances" batch. It defines methods to obtain the current,
    previous, cumulative and future training sets. The returned test
    set is always the complete one (methods used to get previous, cumulative and
    future sets simply return the complete one). It also defines fields that can
    be used to check which classes are in this, previous and
    future batches. Instances of this class are usually created when iterating
    over a :class:`NIScenario` instance.


    It keeps a reference to that :class:`NIScenario` instance, which can be
    used to retrieve additional info about the scenario.
    """
    def __init__(self, scenario: NIScenario[T_train_set_w_targets,
                                            T_test_set_w_targets],
                 force_train_transformations: bool = False,
                 force_test_transformations: bool = False,
                 are_transformations_disabled: bool = False,
                 current_batch: int = -1):
        """
        Creates a NCBatchInfo instance given the root scenario.
        Instances of this class are usually created automatically while
        iterating over an instance of :class:`NCSingleTaskScenario`.

        :param scenario: A reference to the NI scenario
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
        # The reference to the NIScenario
        self.scenario: NIScenario[T_train_set_w_targets,
                                  T_test_set_w_targets] = scenario

        self.force_train_transformations = force_train_transformations
        self.force_test_transformations = force_test_transformations
        self.are_transformations_disabled = are_transformations_disabled

        # are_transformations_disabled can be True without constraints
        if self.force_test_transformations and self.force_train_transformations:
            raise ValueError(
                'Error in force_train/test_transformations arguments.'
                'Can\'t be both True.')

        # List of classes of current and previous batches
        # Being a NI Scenario, will probably equal to the full set of classes
        self.classes_seen_so_far: List[int] = []

        # The list of classes in this batch
        # Being a NI Scenario, will probably equal to the full set of classes
        self.classes_in_this_batch: List[int] = []

        # The list of classes in previous batches, in their encounter order
        # Being a NI Scenario, will probably equal to the full set of classes
        self.previous_classes: List[int] = []

        # The list of classes of next batches, in their encounter order
        # Being a NI Scenario, will probably equal to the full set of classes
        self.future_classes: List[int] = []

        # _go_to_batch initializes the above lists
        self._go_to_batch()

    def current_training_set(self, bucket_classes=False, sort_classes=False,
                             sort_indexes=False) -> MTSingleSet:
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

        :returns: The current batch training set, as a tuple containing the
            Dataset and the task label "0".
        """
        return self.batch_specific_training_set(self.current_batch,
                                                bucket_classes=bucket_classes,
                                                sort_classes=sort_classes,
                                                sort_indexes=sort_indexes)

    def cumulative_training_sets(self, include_current_batch: bool = True,
                                 bucket_classes=False, sort_classes=False,
                                 sort_indexes=False) -> MTMultipleSet:
        """
        Gets the list of cumulative training sets

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

        :returns: The cumulative training sets, as a list. Each element of the
            list is a tuple containing the Dataset and the task label "0".
        """
        if include_current_batch:
            batches = range(0, self.current_batch + 1)
        else:
            batches = range(0, self.current_batch)
        return self.__make_train_subset(batches, bucket_classes=bucket_classes,
                                        sort_classes=sort_classes,
                                        sort_indexes=sort_indexes)

    def complete_training_sets(self, bucket_classes=False, sort_classes=False,
                               sort_indexes=False) -> MTMultipleSet:
        """
        Gets the complete list of training sets

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: All the training sets, as a list. Each element of
            the list is a tuple containing the Dataset and the task label "0".
        """
        return self.__make_train_subset(list(range(0, self.scenario.n_batches)),
                                        bucket_classes=bucket_classes,
                                        sort_classes=sort_classes,
                                        sort_indexes=sort_indexes)

    def future_training_sets(self, bucket_classes=False, sort_classes=False,
                             sort_indexes=False) \
            -> MTMultipleSet:
        """
        Gets the "future" training sets. That is, datasets made of training
        patterns belonging to not-already-encountered batches.

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The future training sets, as a list. Each element of the
            list is a tuple containing the Dataset and the task label "0".
        """
        return self.__make_train_subset(range(self.current_batch + 1,
                                              self.scenario.n_batches),
                                        bucket_classes=bucket_classes,
                                        sort_classes=sort_classes,
                                        sort_indexes=sort_indexes)

    def batch_specific_training_set(self, batch_id: int,
                                    bucket_classes=False,
                                    sort_classes=False, sort_indexes=False) \
            -> MTSingleSet:
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

        :returns: The required training set, as a tuple containing the Dataset
            and the task label "0".
        """
        return self.__make_train_subset(batch_id,
                                        bucket_classes=bucket_classes,
                                        sort_classes=sort_classes,
                                        sort_indexes=sort_indexes)[0]

    def training_set_part(self, dataset_part: DatasetPart,
                          bucket_classes=False, sort_classes=False,
                          sort_indexes=False) \
            -> Union[MTSingleSet, MTMultipleSet]:
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

        :returns: The training sets of the desired part, as a list. Each element
            of the list is a tuple containing the Dataset and the task label
            "0".
        """
        if dataset_part == DatasetPart.CURRENT:
            return self.current_training_set(bucket_classes=bucket_classes,
                                             sort_classes=sort_classes,
                                             sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.CUMULATIVE:
            return self.cumulative_training_sets(include_current_batch=True,
                                                 bucket_classes=bucket_classes,
                                                 sort_classes=sort_classes,
                                                 sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.OLD:
            return self.cumulative_training_sets(include_current_batch=False,
                                                 bucket_classes=bucket_classes,
                                                 sort_classes=sort_classes,
                                                 sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.FUTURE:
            return self.future_training_sets(bucket_classes=bucket_classes,
                                             sort_classes=sort_classes,
                                             sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.COMPLETE:
            return self.complete_training_sets(bucket_classes=bucket_classes,
                                               sort_classes=sort_classes,
                                               sort_indexes=sort_indexes)
        else:
            raise ValueError('Unsupported dataset part')

    def current_test_set(self, bucket_classes=False, sort_classes=False,
                         sort_indexes=False) \
            -> MTSingleSet:
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

        :returns: The complete test set, as a tuple containing the Dataset and
            the task label "0".
        """
        return self.batch_specific_test_set(self.current_batch,
                                            bucket_classes=bucket_classes,
                                            sort_classes=sort_classes,
                                            sort_indexes=sort_indexes)

    def cumulative_test_sets(self, include_current_batch: bool = True,
                             bucket_classes=False, sort_classes=False,
                             sort_indexes=False) -> MTMultipleSet:
        """
        Gets the complete test set

        :param include_current_batch: ignored, kept for compatibility with
            methods from the NC scenario.
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The complete test set, as a list containing one element:
            a tuple containing the Dataset and the task label "0".
        """
        return self.__make_test_subset(bucket_classes=bucket_classes,
                                       sort_classes=sort_classes,
                                       sort_indexes=sort_indexes)

    def complete_test_sets(self, bucket_classes=False, sort_classes=False,
                           sort_indexes=False) -> MTMultipleSet:
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

        :returns: The complete test sets, as a list containing one element:
            a tuple containing the Dataset and the task label "0".
        """
        return self.__make_test_subset(bucket_classes=bucket_classes,
                                       sort_classes=sort_classes,
                                       sort_indexes=sort_indexes)

    def future_test_sets(self, bucket_classes=False, sort_classes=False,
                         sort_indexes=False) -> MTMultipleSet:
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

        :returns: The complete test set, as a list containing one element:
            a tuple containing the Dataset and the task label "0".
        """
        return self.__make_test_subset(bucket_classes=bucket_classes,
                                       sort_classes=sort_classes,
                                       sort_indexes=sort_indexes)

    def batch_specific_test_set(self, batch_id: int, bucket_classes=False,
                                sort_classes=False, sort_indexes=False) \
            -> MTSingleSet:
        """
        Gets the complete test set

        :param batch_id: ignored, kept for compatibility with
            methods from the NC scenario.
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The complete test set, as a list containing one element:
            a tuple containing the Dataset and the task label "0".
        """
        return self.__make_test_subset(bucket_classes=bucket_classes,
                                       sort_classes=sort_classes,
                                       sort_indexes=sort_indexes)[0]

    def test_set_part(self, dataset_part: DatasetPart, bucket_classes=False,
                      sort_classes=False, sort_indexes=False) \
            -> Union[MTSingleSet, MTMultipleSet]:
        """
        GGets the complete test set

        :param dataset_part: ignored, kept for compatibility with
            methods from the NC scenario.
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The complete test set, as a list containing one element:
            a tuple containing the Dataset and the task label "0".
        """
        if dataset_part == DatasetPart.CURRENT:
            return self.current_test_set(bucket_classes=bucket_classes,
                                         sort_classes=sort_classes,
                                         sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.CUMULATIVE:
            return self.cumulative_test_sets(include_current_batch=True,
                                             bucket_classes=bucket_classes,
                                             sort_classes=sort_classes,
                                             sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.OLD:
            return self.cumulative_test_sets(include_current_batch=False,
                                             bucket_classes=bucket_classes,
                                             sort_classes=sort_classes,
                                             sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.FUTURE:
            return self.future_test_sets(bucket_classes=bucket_classes,
                                         sort_classes=sort_classes,
                                         sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.COMPLETE:
            return self.complete_test_sets(bucket_classes=bucket_classes,
                                           sort_classes=sort_classes,
                                           sort_indexes=sort_indexes)
        else:
            raise ValueError('Unsupported dataset part')

    def disable_transformations(self) -> 'NIBatchInfo[' \
                                         'T_train_set_w_targets, ' \
                                         'T_test_set_w_targets]':
        """
        Returns a new batch info instance in which transformations are disabled.
        The current instance is not affected. This is useful when there is a
        need to access raw data. Can be used when picking and storing
        rehearsal/replay patterns.

        :returns: A new NIBatchInfo in which transformations are disabled.
        """

        return NIBatchInfo(
            self.scenario,
            force_train_transformations=self.force_train_transformations,
            force_test_transformations=self.force_test_transformations,
            are_transformations_disabled=True,
            current_batch=self.current_batch)

    def enable_transformations(self) -> 'NIBatchInfo[' \
                                        'T_train_set_w_targets, ' \
                                        'T_test_set_w_targets]':
        """
        Returns a new batch info instance in which transformations are enabled.
        The current instance is not affected. When created the
        NIBatchInfo instance already has transformations enabled.
        This method can be used to re-enable transformations after a previous
        call to disable_transformations().

        :returns: A new NIBatchInfo in which transformations are enabled.
        """
        return NIBatchInfo(
            self.scenario,
            force_train_transformations=self.force_train_transformations,
            force_test_transformations=self.force_test_transformations,
            are_transformations_disabled=False,
            current_batch=self.current_batch)

    def with_train_transformations(self) -> 'NIBatchInfo[' \
                                            'T_train_set_w_targets, ' \
                                            'T_test_set_w_targets]':
        """
        Returns a new batch info instance in which train transformations are
        applied to both training and test sets. The current instance is not
        affected.

        :returns: A new NIBatchInfo in which train transformations
            are applied to both training and test sets.
        """
        return NIBatchInfo(
            self.scenario,
            force_train_transformations=True,
            force_test_transformations=False,
            are_transformations_disabled=self.are_transformations_disabled,
            current_batch=self.current_batch)

    def with_test_transformations(self) -> 'NIBatchInfo[' \
                                           'T_train_set_w_targets, ' \
                                           'T_test_set_w_targets]':
        """
        Returns a new batch info instance in which test transformations are
        applied to both training and test sets. The current instance is
        not affected. This is useful to get the accuracy on the training set
        without considering the usual training data augmentations.

        :returns: A new NIBatchInfo in which test transformations
            are applied to both training and test sets.
        """
        return NIBatchInfo(
            self.scenario,
            force_train_transformations=False,
            force_test_transformations=True,
            are_transformations_disabled=self.are_transformations_disabled,
            current_batch=self.current_batch)

    def _go_to_batch(self):
        if self.current_batch < 0:
            self.classes_in_this_batch = []
            self.previous_classes = []
            self.classes_seen_so_far = []
            self.future_classes = list(range(self.scenario.n_classes))
            return

        class_set_this_batch = set()
        for class_id, class_count in enumerate(
                self.scenario.batch_structure[self.current_batch]):
            if class_count > 0:
                class_set_this_batch.add(class_id)

        self.classes_in_this_batch = list(class_set_this_batch)

        class_set_prev_batches = set()
        for batch_id in range(0, self.current_batch):
            for class_id, class_count in enumerate(
                    self.scenario.batch_structure[batch_id]):
                if class_count > 0:
                    class_set_prev_batches.add(class_id)
        self.previous_classes = list(class_set_prev_batches)

        self.classes_seen_so_far = \
            list(class_set_this_batch.union(class_set_prev_batches))

        class_set_future_batches = set()
        for batch_id in range(self.current_batch, self.scenario.n_batches):
            for class_id, class_count in enumerate(
                    self.scenario.batch_structure[batch_id]):
                if class_count > 0:
                    class_set_future_batches.add(class_id)
        self.future_classes = list(class_set_future_batches)

    def __make_single_subset(self, batch: int, is_train: bool,
                             bucket_classes: bool, sort_classes: bool,
                             sort_indexes: bool):
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

        if is_train:
            patterns_indexes = self.scenario.batch_patterns[batch]

            return make_ni_transformation_subset(
                self.scenario.train_dataset, patterns_transformation,
                targets_transformation, patterns_indexes,
                bucket_classes=bucket_classes, sort_classes=sort_classes,
                sort_indexes=sort_indexes)
        else:
            return make_ni_transformation_subset(
                self.scenario.test_dataset, patterns_transformation,
                targets_transformation, None, bucket_classes=bucket_classes,
                sort_classes=sort_classes, sort_indexes=sort_indexes)

    def __make_subset(self, batches: Union[int, Iterable[int]],
                      is_train, bucket_classes: bool, sort_classes: bool,
                      sort_indexes: bool) -> MTMultipleSet:
        """
        Given the batches IDs list and the dataset type (train or test),
        returns a list of tuples (dataset, task_label "0") in the order
        specified by the batches parameter.

        :param batches: The list of batches IDs. Can be a single int.
        :param is_train: when True, training sets will be returned. When False,
            test sets will be returned.

        :returns: A list of tuples each containing 2 elements: the dataset and
            the corresponding task label (always "0", as this is a SIT
            scenario).
        """

        if isinstance(batches, int):  # Required single batch
            return [(self.__make_single_subset(
                batches, is_train, bucket_classes=bucket_classes,
                sort_classes=sort_classes, sort_indexes=sort_indexes), 0)]

        result: List[Tuple[TransformationSubset, int]] = []

        for batch_id in batches:
            dataset = self.__make_single_subset(
                batch_id, is_train, bucket_classes=bucket_classes,
                sort_classes=sort_classes, sort_indexes=sort_indexes)

            result.append((dataset, 0))

        return result

    def __make_train_subset(self, batches, bucket_classes: bool,
                            sort_classes: bool, sort_indexes: bool):
        return self.__make_subset(
            batches, True, bucket_classes=bucket_classes,
            sort_classes=sort_classes, sort_indexes=sort_indexes)

    def __make_test_subset(self, bucket_classes: bool,
                           sort_classes: bool, sort_indexes: bool):
        return [(self.__make_single_subset(
            0, False, bucket_classes=bucket_classes,
            sort_classes=sort_classes, sort_indexes=sort_indexes), 0)]


__all__ = ['NIScenario', 'NIBatchInfo']
