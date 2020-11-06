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

from typing import Optional, List, Generic, Sequence, Dict, Any, Set

import torch

from avalanche.benchmarks.scenarios.generic_definitions import \
    TrainSetWithTargets, TestSetWithTargets
from avalanche.training.utils import TransformationSubset, IDatasetWithTargets
from avalanche.benchmarks.scenarios.generic_cl_scenario import \
    GenericCLScenario, GenericStepInfo


def _step_structure_from_assignment(dataset: IDatasetWithTargets,
                                    assignment: Sequence[Sequence[int]],
                                    n_classes: int):
    n_steps = len(assignment)
    step_structure = [[0 for _ in range(n_classes)] for _ in range(n_steps)]

    for step_id in range(n_steps):
        step_targets = [dataset.targets[pattern_idx]
                        for pattern_idx in assignment[step_id]]
        cls_ids, cls_counts = torch.unique(torch.as_tensor(
            step_targets), return_counts=True)

        for unique_idx in range(len(cls_ids)):
            step_structure[step_id][int(cls_ids[unique_idx])] += \
                int(cls_counts[unique_idx])

    return step_structure


class NIScenario(GenericCLScenario[TrainSetWithTargets,
                                   TestSetWithTargets, 'NIStepInfo'],
                 Generic[TrainSetWithTargets, TestSetWithTargets]):
    """
    This class defines a "New Instance" scenario.
    Once created, an instance of this class can be iterated in order to obtain
    the step sequence under the form of instances of :class:`NIStepInfo`.

    Instances of this class can be created using the constructor directly.
    However, we recommend using facilities like
    :func:`avalanche.benchmarks.generators.ni_scenario`.

    Consider that every method from :class:`NIStepInfo` used to retrieve
    parts of the test set (past, current, future, cumulative) always return the
    complete test set. That is, they behave as the getter for the complete test
    set.
    """

    def __init__(
            self, train_dataset: TrainSetWithTargets,
            test_dataset: TestSetWithTargets,
            n_steps: int,
            task_labels: bool = False,
            shuffle: bool = True,
            seed: Optional[int] = None,
            balance_steps: bool = False,
            min_class_patterns_in_step: int = 0,
            fixed_step_assignment: Optional[Sequence[Sequence[int]]] = None,
            reproducibility_data: Optional[Dict[str, Any]] = None):
        """
        Creates a NIScenario instance given the training and test Datasets and
        the number of steps.

        :param train_dataset: The training dataset. The dataset must contain a
            "targets" field. For instance, one can safely use the datasets from
            the torchvision package.
        :param test_dataset: The test dataset. The dataset must contain a
            "targets" field. For instance, one can safely use the datasets from
            the torchvision package.
        :param n_steps: The number of steps.
        :param task_labels: If True, each step will have an ascending task
            label. If False, the task label will be 0 for all the steps.
            Defaults to False.
        :param shuffle: If True, the patterns order will be shuffled. Defaults
            to True.
        :param seed: If shuffle is True and seed is not None, the class order
            will be shuffled according to the seed. When None, the current
            PyTorch random number generator state will be used.
            Defaults to None.
        :param balance_steps: If True, pattern of each class will be equally
            spread across all steps. If False, patterns will be assigned to
            steps in a complete random way. Defaults to False.
        :param min_class_patterns_in_step: The minimum amount of patterns of
            every class that must be assigned to every step. Compatible with
            the ``balance_steps`` parameter. An exception will be raised if
            this constraint can't be satisfied. Defaults to 0.
        :param fixed_step_assignment: If not None, the pattern assignment
            to use. It must be a list with an entry for each step. Each entry
            is a list that contains the indexes of patterns belonging to that
            step. Overrides the ``shuffle``, ``balance_steps`` and
            ``min_class_patterns_in_step`` parameters.
        :param reproducibility_data: If not None, overrides all the other
            scenario definition options, including ``fixed_step_assignment``.
            This is usually a dictionary containing data used to
            reproduce a specific experiment. One can use the
            ``get_reproducibility_data`` method to get (and even distribute)
            the experiment setup so that it can be loaded by passing it as this
            parameter. In this way one can be sure that the same specific
            experimental setup is being used (for reproducibility purposes).
            Beware that, in order to reproduce an experiment, the same train and
            test datasets must be used. Defaults to None.
        """

        task_ids: List[int]
        if task_labels:
            task_ids = list(range(n_steps))
        else:
            task_ids = [0] * n_steps

        if reproducibility_data is not None:
            super(NIScenario, self).__init__(
                train_dataset, test_dataset,
                train_dataset, test_dataset,
                [], [], task_ids,
                complete_test_set_only=True,
                reproducibility_data=reproducibility_data,
                step_factory=NIStepInfo)
            n_steps = self.n_steps
            if task_labels:
                task_ids = list(range(n_steps))
            else:
                task_ids = [0] * n_steps

        if n_steps < 1:
            raise ValueError('Invalid number of steps (n_steps parameter): '
                             'must be greater than 0')

        if min_class_patterns_in_step < 0 and reproducibility_data is None:
            raise ValueError('Invalid min_class_patterns_in_step parameter: '
                             'must be greater than or equal to 0')

        unique_targets, unique_count = torch.unique(
            torch.as_tensor(train_dataset.targets), return_counts=True)

        # The number of classes
        self.n_classes: int = len(unique_targets)

        # n_patterns_per_class contains the number of patterns for each class
        self.n_patterns_per_class: List[int] = \
            [0 for _ in range(self.n_classes)]

        if fixed_step_assignment:
            included_patterns = list()
            for step_def in fixed_step_assignment:
                included_patterns.extend(step_def)
            subset = TransformationSubset(train_dataset, included_patterns)
            unique_targets, unique_count = torch.unique(
                torch.as_tensor(subset.targets), return_counts=True)

        for unique_idx in range(len(unique_targets)):
            class_id = int(unique_targets[unique_idx])
            class_count = int(unique_count[unique_idx])
            self.n_patterns_per_class[class_id] = class_count

        # The number of patterns in each step
        self.n_patterns_per_step: List[int] = []
        # self.step_structure[step_id][class_id] is the amount of patterns
        # of class "class_id" in step "step_id
        self.step_structure: List[List[int]] = []
        # step_patterns contains, for each step, the list of patterns
        # assigned to that step (as indexes of elements from the training set)

        if reproducibility_data or fixed_step_assignment:
            # fixed_patterns_assignment/reproducibility_data is the user
            # provided pattern assignment. All we have to do is populate
            # remaining fields of the class!
            # n_patterns_per_step is filled later based on step_structure
            # so we only need to fill step_structure.

            if reproducibility_data:
                step_patterns = self.train_steps_patterns_assignment
            else:
                step_patterns = fixed_step_assignment
            self.step_structure = _step_structure_from_assignment(
                train_dataset, step_patterns, self.n_classes
            )
        else:
            # All steps will all contain the same amount of patterns
            # The amount of patterns doesn't need to be divisible without
            # remainder by the number of steps, so we distribute remaining
            # patterns across randomly selected steps (when shuffling) or
            # the first N steps (when not shuffling). However, we first have
            # to check if the min_class_patterns_in_step constraint is
            # satisfiable.
            min_class_patterns = min(self.n_patterns_per_class)
            if min_class_patterns < n_steps * min_class_patterns_in_step:
                raise ValueError('min_class_patterns_in_step constraint '
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

            # Here we assign patterns to each steo. Two different strategies
            # are required in order to manage the balance_steps parameter.
            if balance_steps:
                # If balance_steps is True we have to make sure that patterns
                # of each class are equally distributed across steps.
                #
                # To do this, populate self.step_structure, which will
                # describe how many patterns of each class are assigned to each
                # step. Then, for each step, assign the required amount of
                # patterns of each class.
                #
                # We already checked that there are enough patterns for each
                # class to satisfy the min_class_patterns_in_step param so here
                # we don't need to explicitly enforce that constraint.

                # First, count how many patterns of each class we have to assign
                # to all the steps (avg). We also get the number of remaining
                # patterns which we'll have to assign in a second step.
                class_patterns_per_step = [
                    ((n_class_patterns // n_steps),
                     (n_class_patterns % n_steps))
                    for n_class_patterns in self.n_patterns_per_class
                ]

                # Remember: step_structure[step_id][class_id] is the amount of
                # patterns of class "class_id" in step "step_id"
                #
                # This is the easier step: just assign the average amount of
                # class patterns to each step.
                self.step_structure = [
                    [class_patterns_this_step[0]
                     for class_patterns_this_step
                     in class_patterns_per_step] for _ in range(n_steps)
                ]

                # Now we have to distribute the remaining patterns of each class
                #
                # This means that, for each class, we can (randomly) select
                # "n_class_patterns % n_steps" steps to assign a single
                # additional pattern of that class.
                for class_id in range(self.n_classes):
                    n_remaining = class_patterns_per_step[class_id][1]
                    if n_remaining == 0:
                        continue
                    if shuffle:
                        assignment_of_remaining_patterns = torch.randperm(
                            n_steps).tolist()[:n_remaining]
                    else:
                        assignment_of_remaining_patterns = range(n_remaining)
                    for step_id in assignment_of_remaining_patterns:
                        self.step_structure[step_id][class_id] += 1

                # Following the self.step_structure definition, assign
                # the actual patterns to each step.
                #
                # For each step we assign exactly
                # self.step_structure[step_id][class_id] patterns of
                # class "class_id"
                step_patterns = [[] for _ in range(n_steps)]
                next_idx_per_class = [0 for _ in range(self.n_classes)]
                for step_id in range(n_steps):
                    for class_id in range(self.n_classes):
                        start_idx = next_idx_per_class[class_id]
                        n_patterns = self.step_structure[step_id][class_id]
                        end_idx = start_idx + n_patterns
                        step_patterns[step_id].extend(
                            classes_to_patterns_idx[class_id][start_idx:end_idx]
                        )
                        next_idx_per_class[class_id] = end_idx
            else:
                # If balance_steps if False, we just randomly shuffle the
                # patterns indexes and pick N patterns for each step.
                #
                # However, we have to enforce the min_class_patterns_in_step
                # constraint, which makes things difficult.
                # In the balance_steps scenario, that constraint is implicitly
                # enforced by equally distributing class patterns in each step
                # (we already checked that there are enough overall patterns
                # for each class to satisfy min_class_patterns_in_step)

                # Here we have to assign the minimum required amount of class
                # patterns to each step first, then we can move to randomly
                # assign the remaining patterns to each step.

                # First, initialize step_patterns and step_structure
                step_patterns = [[] for _ in range(n_steps)]
                self.step_structure = [[0 for _ in range(self.n_classes)]
                                       for _ in range(n_steps)]

                # For each step we assign exactly
                # min_class_patterns_in_step patterns from each class
                #
                # Very similar to the loop found in the balance_steps branch!
                # Remember that classes_to_patterns_idx is already shuffled
                # (if required)
                next_idx_per_class = [0 for _ in range(self.n_classes)]
                remaining_patterns = set(range(len(train_dataset)))

                for step_id in range(n_steps):
                    for class_id in range(self.n_classes):
                        next_idx = next_idx_per_class[class_id]
                        end_idx = next_idx + min_class_patterns_in_step
                        selected_patterns = \
                            classes_to_patterns_idx[next_idx:end_idx]
                        step_patterns[step_id].extend(selected_patterns)
                        self.step_structure[step_id][class_id] += \
                            min_class_patterns_in_step
                        remaining_patterns.difference_update(selected_patterns)
                        next_idx_per_class[class_id] = end_idx

                remaining_patterns = list(remaining_patterns)

                # We have assigned the required min_class_patterns_in_step,
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

                avg_step_size = len(patterns_order) // n_steps
                n_remaining = len(patterns_order) % n_steps
                prev_idx = 0
                for step_id in range(n_steps):
                    next_idx = prev_idx + avg_step_size
                    step_patterns[step_id].extend(
                        patterns_order[prev_idx:next_idx])
                    cls_ids, cls_counts = torch.unique(torch.as_tensor(
                        targets_order[prev_idx:next_idx]), return_counts=True)

                    cls_ids = cls_ids.tolist()
                    cls_counts = cls_counts.tolist()

                    for unique_idx in range(len(cls_ids)):
                        self.step_structure[step_id][cls_ids[unique_idx]] += \
                            cls_counts[unique_idx]
                    prev_idx = next_idx

                # Distribute remaining patterns
                if n_remaining > 0:
                    if shuffle:
                        assignment_of_remaining_patterns = torch.randperm(
                            n_steps).tolist()[:n_remaining]
                    else:
                        assignment_of_remaining_patterns = range(n_remaining)
                    for step_id in assignment_of_remaining_patterns:
                        pattern_idx = patterns_order[prev_idx]
                        pattern_target = targets_order[prev_idx]
                        step_patterns[step_id].append(pattern_idx)

                        self.step_structure[step_id][pattern_target] += 1
                        prev_idx += 1

        self.n_patterns_per_step = [len(step_patterns[step_id])
                                    for step_id in range(n_steps)]

        self._classes_in_step = None  # Will be lazy initialized later

        super(NIScenario, self).__init__(
            train_dataset, test_dataset,
            train_dataset, test_dataset,
            step_patterns, [], task_ids,
            complete_test_set_only=True,
            step_factory=NIStepInfo)

    @property
    def classes_in_step(self) -> Sequence[Set[int]]:
        if self._classes_in_step is None:
            self._classes_in_step = []
            for step_id in range(self.n_steps):
                self._classes_in_step.append(set())
                step_s = self.step_structure[step_id]
                for class_id, n_patterns_of_class in enumerate(step_s):
                    if n_patterns_of_class > 0:
                        self._classes_in_step[step_id].add(class_id)
        return self._classes_in_step

    def get_reproducibility_data(self) -> Dict[str, Any]:
        # In fact, the only data required for reproducibility of a NI Scenario
        # is the one already included in the GenericCLScenario!
        return super().get_reproducibility_data()


class NIStepInfo(GenericStepInfo[NIScenario[TrainSetWithTargets,
                                            TestSetWithTargets]],
                 Generic[TrainSetWithTargets, TestSetWithTargets]):
    """
    Defines a "New Instances" step. It defines methods to obtain the current,
    previous, cumulative and future training sets. The returned test
    set is always the complete one (methods used to get previous, cumulative and
    future sets simply return the complete one). It also defines fields that can
    be used to check which classes are in this, previous and
    future steps. Instances of this class are usually created when iterating
    over a :class:`NIScenario` instance.

    It keeps a reference to that :class:`NIScenario` instance, which can be
    used to retrieve additional info about the scenario.
    """
    def __init__(self, scenario: NIScenario[TrainSetWithTargets,
                                            TestSetWithTargets],
                 current_step: int,
                 force_train_transformations: bool = False,
                 force_test_transformations: bool = False,
                 are_transformations_disabled: bool = False):
        """
        Creates a NCStepInfo instance given the root scenario.
        Instances of this class are usually created automatically while
        iterating over an instance of :class:`NIScenario`.

        :param scenario: A reference to the NI scenario
        :param current_step: Defines the current step ID.
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

        super(NIStepInfo, self).__init__(
            scenario, current_step,
            force_train_transformations=force_train_transformations,
            force_test_transformations=force_test_transformations,
            are_transformations_disabled=are_transformations_disabled,
            transformation_step_factory=NIStepInfo)


__all__ = ['NIScenario', 'NIStepInfo']
