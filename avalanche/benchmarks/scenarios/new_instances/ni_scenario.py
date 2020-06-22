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

from typing import Optional, List, Generic, Sequence, Dict, Any

import torch

from avalanche.benchmarks.scenarios.generic_definitions import \
    TrainSetWithTargets, TestSetWithTargets
from avalanche.training.utils import TransformationSubset, IDatasetWithTargets
from avalanche.benchmarks.scenarios.generic_cl_scenario import \
    GenericCLScenario, GenericStepInfo


def _batch_structure_from_assignment(dataset: IDatasetWithTargets,
                                     assignment: Sequence[Sequence[int]],
                                     n_classes: int):
    n_batches = len(assignment)
    batch_structure = [[0 for _ in range(n_classes)]
                       for _ in range(n_batches)]

    for batch_id in range(n_batches):
        batch_targets = [dataset.targets[pattern_idx]
                         for pattern_idx in
                         assignment[batch_id]]
        cls_ids, cls_counts = torch.unique(torch.as_tensor(
            batch_targets), return_counts=True)

        for unique_idx in range(len(cls_ids)):
            batch_structure[batch_id][int(cls_ids[unique_idx])] += \
                int(cls_counts[unique_idx])

    return batch_structure


class NIScenario(GenericCLScenario[TrainSetWithTargets, TestSetWithTargets],
                 Generic[TrainSetWithTargets, TestSetWithTargets]):

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
    parts of the test set (past, current, future, cumulative) always return the
    complete test set. That is, they behave as the getter for the complete test
    set. These methods are left for compatibility with the ones found in the
    :class:`avalanche.benchmarks.scenarios.new_classes.NCBatchInfo` scenario.
    """

    def __init__(
            self, train_dataset: TrainSetWithTargets,
            test_dataset: TestSetWithTargets, n_batches: int,
            shuffle: bool = True, seed: Optional[int] = None,
            balance_batches: bool = False,
            min_class_patterns_in_batch: int = 0,
            fixed_batch_assignment: Optional[Sequence[Sequence[int]]] = None,
            reproducibility_data: Optional[Dict[str, Any]] = None):
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
        :param reproducibility_data: If not None, overrides all the other
            scenario definition options, including ``fixed_batch_assignment``.
            This is usually a dictionary containing data used to
            reproduce a specific experiment. One can use the
            ``get_reproducibility_data`` method to get (and even distribute)
            the experiment setup so that it can be loaded by passing it as this
            parameter. In this way one can be sure that the same specific
            experimental setup is being used (for reproducibility purposes).
            Beware that, in order to reproduce an experiment, the same train and
            test datasets must be used. Defaults to None.
        """

        if reproducibility_data is not None:
            super(NIScenario, self).__init__(
                train_dataset, test_dataset, [], [], [],
                reproducibility_data=reproducibility_data)
            n_batches = self.n_steps

        # The number of batches
        self.n_batches: int = n_batches

        if n_batches < 1:
            raise ValueError('Invalid number of batches (n_batches parameter): '
                             'must be greater than 0')

        if min_class_patterns_in_batch < 0 and reproducibility_data is None:
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
            class_count = int(unique_count[unique_idx])
            self.n_patterns_per_class[class_id] = class_count

        # The number of patterns in each batch
        self.n_patterns_per_batch: List[int] = []
        # self.batch_structure[batch_id][class_id] is the amount of patterns
        # of class "class_id" in batch "batch_id
        self.batch_structure: List[List[int]] = []
        # batch_patterns contains, for each batch, the list of patterns
        # assigned to that batch (as indexes of elements from the training set)

        if reproducibility_data or fixed_batch_assignment:
            # fixed_patterns_assignment/reproducibility_data is the user
            # provided pattern assignment. All we have to do is populate
            # remaining fields of the class!
            # n_patterns_per_batch is filled later based on batch_structure
            # so we only need to fill batch_structure.

            if reproducibility_data:
                batch_patterns = self.train_steps_patterns_assignment
            else:
                batch_patterns = fixed_batch_assignment
            self.batch_structure = _batch_structure_from_assignment(
                train_dataset, batch_patterns, self.n_classes
            )
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
                batch_patterns = [[] for _ in range(self.n_batches)]
                next_idx_per_class = [0 for _ in range(self.n_classes)]
                for batch_id in range(self.n_batches):
                    for class_id in range(self.n_classes):
                        start_idx = next_idx_per_class[class_id]
                        n_patterns = self.batch_structure[batch_id][class_id]
                        end_idx = start_idx + n_patterns
                        batch_patterns[batch_id].extend(
                            classes_to_patterns_idx[class_id][start_idx:end_idx]
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
                batch_patterns = [[] for _ in range(self.n_batches)]
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
                        batch_patterns[batch_id].extend(selected_patterns)
                        self.batch_structure[batch_id][class_id] += \
                            min_class_patterns_in_batch
                        remaining_patterns.difference_update(selected_patterns)
                        next_idx_per_class[class_id] = end_idx

                remaining_patterns = list(remaining_patterns)

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
                    remaining_patterns.sort()
                    patterns_order = remaining_patterns
                targets_order = [train_dataset.targets[pattern_idx]
                                 for pattern_idx in patterns_order]

                avg_batch_size = len(patterns_order) // self.n_batches
                n_remaining = len(patterns_order) % self.n_batches
                prev_idx = 0
                for batch_id in range(self.n_batches):
                    next_idx = prev_idx + avg_batch_size
                    batch_patterns[batch_id].extend(
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
                        batch_patterns[batch_id].append(pattern_idx)

                        self.batch_structure[batch_id][pattern_target] += 1
                        prev_idx += 1

        self.n_patterns_per_batch = [len(batch_patterns[batch_id])
                                     for batch_id in range(self.n_batches)]

        self.classes_in_batch = []
        for batch_id in range(self.n_batches):
            self.classes_in_batch.append([])
            batch_s = self.batch_structure[batch_id]
            for class_id, n_patterns_of_class in enumerate(batch_s):
                if n_patterns_of_class > 0:
                    self.classes_in_batch[batch_id].append(class_id)
        super(NIScenario, self).__init__(
            train_dataset, test_dataset, batch_patterns, [],
            task_labels=[0] * self.n_batches,
            return_complete_test_set_only=True)

    def __getitem__(self, batch_id):
        return NIBatchInfo(self, batch_id)

    def get_reproducibility_data(self) -> Dict[str, Any]:
        # In fact, the only data required for reproducibility of a NI Scenario
        # is the one already included in the GenericCLScenario!
        return super().get_reproducibility_data()


class NIBatchInfo(GenericStepInfo[NIScenario[TrainSetWithTargets,
                                             TestSetWithTargets]],
                  Generic[TrainSetWithTargets, TestSetWithTargets]):
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
    def __init__(self, scenario: NIScenario[TrainSetWithTargets,
                                            TestSetWithTargets],
                 current_batch: int,
                 force_train_transformations: bool = False,
                 force_test_transformations: bool = False,
                 are_transformations_disabled: bool = False):
        """
        Creates a NCBatchInfo instance given the root scenario.
        Instances of this class are usually created automatically while
        iterating over an instance of :class:`NCSingleTaskScenario`.

        :param scenario: A reference to the NI scenario
        :param current_batch: Defines the current batch ID.
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
        """
        super(NIBatchInfo, self).__init__(
            scenario, current_batch,
            force_train_transformations=force_train_transformations,
            force_test_transformations=force_test_transformations,
            are_transformations_disabled=are_transformations_disabled,
            transformation_step_factory=NIBatchInfo)

        # The current batch ID
        self.current_batch: int = current_batch

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


__all__ = ['NIScenario', 'NIBatchInfo']
