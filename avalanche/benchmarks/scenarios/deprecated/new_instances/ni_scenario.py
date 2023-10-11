################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 28-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

from typing import Optional, List, Sequence, Dict, Any

import torch

from avalanche.benchmarks.scenarios.deprecated.classification_scenario import (
    ClassificationScenario,
    ClassificationStream,
    ClassificationExperience,
)
from avalanche.benchmarks.scenarios.deprecated.new_instances.ni_utils import (
    _exp_structure_from_assignment,
)
from avalanche.benchmarks.utils.classification_dataset import (
    _taskaware_classification_subset,
    TaskAwareClassificationDataset,
    TaskAwareSupervisedClassificationDataset,
)
from avalanche.benchmarks.utils.flat_data import ConstantSequence


class NIScenario(
    ClassificationScenario[
        "NIStream", "NIExperience", TaskAwareSupervisedClassificationDataset
    ]
):
    """
    This class defines a "New Instance" scenario.
    Once created, an instance of this class can be iterated in order to obtain
    the experience sequence under the form of instances of
    :class:`NIExperience`.

    Instances of this class can be created using the constructor directly.
    However, we recommend using facilities like
    :func:`avalanche.benchmarks.generators.ni_scenario`.

    Consider that every method from :class:`NIExperience` used to retrieve
    parts of the test set (past, current, future, cumulative) always return the
    complete test set. That is, they behave as the getter for the complete test
    set.
    """

    def __init__(
        self,
        train_dataset: TaskAwareClassificationDataset,
        test_dataset: TaskAwareClassificationDataset,
        n_experiences: int,
        task_labels: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
        balance_experiences: bool = False,
        min_class_patterns_in_exp: int = 0,
        fixed_exp_assignment: Optional[Sequence[Sequence[int]]] = None,
        reproducibility_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates a NIScenario instance given the training and test Datasets and
        the number of experiences.

        :param train_dataset: The training dataset. The dataset must be an
            instance of :class:`AvalancheDataset`. For instance, one can
            use the datasets from the torchvision package like that:
            ``train_dataset=AvalancheDataset(torchvision_dataset)``.
        :param test_dataset: The test dataset. The dataset must be a
            subclass of :class:`AvalancheDataset`. For instance, one can
            use the datasets from the torchvision package like that:
            ``test_dataset=AvalancheDataset(torchvision_dataset)``.
        :param n_experiences: The number of experiences.
        :param task_labels: If True, each experience will have an ascending task
            label. If False, the task label will be 0 for all the experiences.
            Defaults to False.
        :param shuffle: If True, the patterns order will be shuffled. Defaults
            to True.
        :param seed: If shuffle is True and seed is not None, the class order
            will be shuffled according to the seed. When None, the current
            PyTorch random number generator state will be used.
            Defaults to None.
        :param balance_experiences: If True, pattern of each class will be
            equally spread across all experiences. If False, patterns will be
            assigned to experiences in a complete random way. Defaults to False.
        :param min_class_patterns_in_exp: The minimum amount of patterns of
            every class that must be assigned to every experience. Compatible
            with the ``balance_experiences`` parameter. An exception will be
            raised if this constraint can't be satisfied. Defaults to 0.
        :param fixed_exp_assignment: If not None, the pattern assignment
            to use. It must be a list with an entry for each experience. Each
            entry is a list that contains the indexes of patterns belonging to
            that experience. Overrides the ``shuffle``, ``balance_experiences``
            and ``min_class_patterns_in_exp`` parameters.
        :param reproducibility_data: If not None, overrides all the other
            scenario definition options, including ``fixed_exp_assignment``.
            This is usually a dictionary containing data used to
            reproduce a specific experiment. One can use the
            ``get_reproducibility_data`` method to get (and even distribute)
            the experiment setup so that it can be loaded by passing it as this
            parameter. In this way one can be sure that the same specific
            experimental setup is being used (for reproducibility purposes).
            Beware that, in order to reproduce an experiment, the same train and
            test datasets must be used. Defaults to None.
        """

        train_dataset = TaskAwareSupervisedClassificationDataset(train_dataset)
        test_dataset = TaskAwareSupervisedClassificationDataset(test_dataset)

        self._has_task_labels = task_labels

        self.train_exps_patterns_assignment: List[List[int]] = []

        if reproducibility_data is not None:
            self.train_exps_patterns_assignment = reproducibility_data[
                "exps_patterns_assignment"
            ]
            self._has_task_labels = reproducibility_data["has_task_labels"]
            n_experiences = len(self.train_exps_patterns_assignment)

        if n_experiences < 1:
            raise ValueError(
                "Invalid number of experiences (n_experiences "
                "parameter): must be greater than 0"
            )

        if min_class_patterns_in_exp < 0 and reproducibility_data is None:
            raise ValueError(
                "Invalid min_class_patterns_in_exp parameter: "
                "must be greater than or equal to 0"
            )

        # # Good idea, but doesn't work
        # transform_groups = train_eval_transforms(train_dataset, test_dataset)
        #
        # train_dataset = train_dataset \
        #     .replace_transforms(*transform_groups['train'], group='train') \
        #     .replace_transforms(*transform_groups['eval'], group='eval')
        #
        # test_dataset = test_dataset \
        #     .replace_transforms(*transform_groups['train'], group='train') \
        #     .replace_transforms(*transform_groups['eval'], group='eval')

        unique_targets, unique_count = torch.unique(
            torch.as_tensor(train_dataset.targets), return_counts=True
        )

        self.n_classes: int = len(unique_targets)
        """
        The amount of classes in the original training set.
        """

        self.n_patterns_per_class: List[int] = [0 for _ in range(self.n_classes)]
        """
        The amount of patterns for each class in the original training set.
        """

        lst_fixed_exp_assignment: Optional[List[List[int]]] = None
        if fixed_exp_assignment is not None:
            lst_fixed_exp_assignment = list()
            for lst in fixed_exp_assignment:
                lst_fixed_exp_assignment.append(list(lst))

        if lst_fixed_exp_assignment is not None:
            included_patterns: List[int] = list()
            for exp_def in lst_fixed_exp_assignment:
                included_patterns.extend(exp_def)
            subset = _taskaware_classification_subset(
                train_dataset, indices=included_patterns
            )
            unique_targets, unique_count = torch.unique(
                torch.as_tensor(subset.targets), return_counts=True
            )

        for unique_idx in range(len(unique_targets)):
            class_id = int(unique_targets[unique_idx])
            class_count = int(unique_count[unique_idx])
            self.n_patterns_per_class[class_id] = class_count

        self.n_patterns_per_experience: List[int] = []
        """
        The number of patterns in each experience.
        """

        self.exp_structure: List[List[int]] = []
        """ This field contains, for each training experience, the number of
        instances of each class assigned to that experience. """

        if reproducibility_data or lst_fixed_exp_assignment:
            # fixed_patterns_assignment/reproducibility_data is the user
            # provided pattern assignment. All we have to do is populate
            # remaining fields of the class!
            # n_patterns_per_experience is filled later based on exp_structure
            # so we only need to fill exp_structure.

            if reproducibility_data:
                exp_patterns = self.train_exps_patterns_assignment
            else:
                assert lst_fixed_exp_assignment is not None
                exp_patterns = lst_fixed_exp_assignment
            self.exp_structure = _exp_structure_from_assignment(
                train_dataset, exp_patterns, self.n_classes
            )
        else:
            # All experiences will all contain the same amount of patterns
            # The amount of patterns doesn't need to be divisible without
            # remainder by the number of experience, so we distribute remaining
            # patterns across randomly selected experience (when shuffling) or
            # the first N experiences (when not shuffling). However, we first
            # have to check if the min_class_patterns_in_exp constraint is
            # satisfiable.
            min_class_patterns = min(self.n_patterns_per_class)
            if min_class_patterns < n_experiences * min_class_patterns_in_exp:
                raise ValueError(
                    "min_class_patterns_in_exp constraint " "can't be satisfied"
                )

            if seed is not None:
                torch.random.manual_seed(seed)

            # First, get the patterns indexes for each class
            targets_as_tensor = torch.as_tensor(train_dataset.targets)
            classes_to_patterns_idx = [
                torch.nonzero(torch.eq(targets_as_tensor, class_id)).view(-1).tolist()
                for class_id in range(self.n_classes)
            ]

            if shuffle:
                classes_to_patterns_idx = [
                    torch.as_tensor(cls_patterns)[
                        torch.randperm(len(cls_patterns))
                    ].tolist()
                    for cls_patterns in classes_to_patterns_idx
                ]

            # Here we assign patterns to each experience. Two different
            # strategies are required in order to manage the
            # balance_experiences parameter.
            if balance_experiences:
                # If balance_experiences is True we have to make sure that
                # patterns of each class are equally distributed across
                # experiences.
                #
                # To do this, populate self.exp_structure, which will
                # describe how many patterns of each class are assigned to each
                # experience. Then, for each experience, assign the required
                # amount of patterns of each class.
                #
                # We already checked that there are enough patterns for each
                # class to satisfy the min_class_patterns_in_exp param so here
                # we don't need to explicitly enforce that constraint.

                # First, count how many patterns of each class we have to assign
                # to all the experiences (avg). We also get the number of
                # remaining patterns which we'll have to assign in a second
                # experience.
                class_patterns_per_exp = [
                    (
                        (n_class_patterns // n_experiences),
                        (n_class_patterns % n_experiences),
                    )
                    for n_class_patterns in self.n_patterns_per_class
                ]

                # Remember: exp_structure[exp_id][class_id] is the amount of
                # patterns of class "class_id" in experience "exp_id"
                #
                # This is the easier experience: just assign the average amount
                # of class patterns to each experience.
                self.exp_structure = [
                    [
                        class_patterns_this_exp[0]
                        for class_patterns_this_exp in class_patterns_per_exp
                    ]
                    for _ in range(n_experiences)
                ]

                # Now we have to distribute the remaining patterns of each class
                #
                # This means that, for each class, we can (randomly) select
                # "n_class_patterns % n_experiences" experiences to assign a
                # single additional pattern of that class.
                for class_id in range(self.n_classes):
                    n_remaining = class_patterns_per_exp[class_id][1]
                    if n_remaining == 0:
                        continue
                    if shuffle:
                        assignment_of_remaining_patterns = torch.randperm(
                            n_experiences
                        ).tolist()[:n_remaining]
                    else:
                        assignment_of_remaining_patterns = range(n_remaining)
                    for exp_id in assignment_of_remaining_patterns:
                        self.exp_structure[exp_id][class_id] += 1

                # Following the self.exp_structure definition, assign
                # the actual patterns to each experience.
                #
                # For each experience we assign exactly
                # self.exp_structure[exp_id][class_id] patterns of
                # class "class_id"
                exp_patterns = [[] for _ in range(n_experiences)]
                next_idx_per_class = [0 for _ in range(self.n_classes)]
                for exp_id in range(n_experiences):
                    for class_id in range(self.n_classes):
                        start_idx = next_idx_per_class[class_id]
                        n_patterns = self.exp_structure[exp_id][class_id]
                        end_idx = start_idx + n_patterns
                        exp_patterns[exp_id].extend(
                            classes_to_patterns_idx[class_id][start_idx:end_idx]
                        )
                        next_idx_per_class[class_id] = end_idx
            else:
                # If balance_experiences if False, we just randomly shuffle the
                # patterns indexes and pick N patterns for each experience.
                #
                # However, we have to enforce the min_class_patterns_in_exp
                # constraint, which makes things difficult.
                # In the balance_experiences scenario, that constraint is
                # implicitly enforced by equally distributing class patterns in
                # each experience (we already checked that there are enough
                # overall patterns for each class to satisfy
                # min_class_patterns_in_exp)

                # Here we have to assign the minimum required amount of class
                # patterns to each experience first, then we can move to
                # randomly assign the remaining patterns to each experience.

                # First, initialize exp_patterns and exp_structure
                exp_patterns = [[] for _ in range(n_experiences)]
                self.exp_structure = [
                    [0 for _ in range(self.n_classes)] for _ in range(n_experiences)
                ]

                # For each experience we assign exactly
                # min_class_patterns_in_exp patterns from each class
                #
                # Very similar to the loop found in the balance_experiences
                # branch! Remember that classes_to_patterns_idx is already
                # shuffled (if required)
                next_idx_per_class = [0 for _ in range(self.n_classes)]
                remaining_patterns = set(range(len(train_dataset)))

                for exp_id in range(n_experiences):
                    for class_id in range(self.n_classes):
                        next_idx = next_idx_per_class[class_id]
                        end_idx = next_idx + min_class_patterns_in_exp
                        selected_patterns = classes_to_patterns_idx[next_idx:end_idx]
                        exp_patterns[exp_id].extend(selected_patterns)
                        self.exp_structure[exp_id][
                            class_id
                        ] += min_class_patterns_in_exp
                        remaining_patterns.difference_update(selected_patterns)
                        next_idx_per_class[class_id] = end_idx

                lst_remaining_patterns = list(remaining_patterns)

                # We have assigned the required min_class_patterns_in_exp,
                # now we assign the remaining patterns
                #
                # We'll work on lst_remaining_patterns, which contains
                # indexes of patterns not assigned in the previous
                # experience.
                if shuffle:
                    patterns_order = torch.as_tensor(lst_remaining_patterns)[
                        torch.randperm(len(lst_remaining_patterns))
                    ].tolist()
                else:
                    lst_remaining_patterns.sort()
                    patterns_order = lst_remaining_patterns
                targets_order = [
                    train_dataset.targets[pattern_idx] for pattern_idx in patterns_order
                ]

                avg_exp_size = len(patterns_order) // n_experiences
                n_remaining = len(patterns_order) % n_experiences
                prev_idx = 0
                for exp_id in range(n_experiences):
                    next_idx = prev_idx + avg_exp_size
                    exp_patterns[exp_id].extend(patterns_order[prev_idx:next_idx])
                    cls_ids, cls_counts = torch.unique(
                        torch.as_tensor(targets_order[prev_idx:next_idx]),
                        return_counts=True,
                    )

                    cls_ids = cls_ids.tolist()
                    cls_counts = cls_counts.tolist()

                    for unique_idx in range(len(cls_ids)):
                        self.exp_structure[exp_id][cls_ids[unique_idx]] += cls_counts[
                            unique_idx
                        ]
                    prev_idx = next_idx

                # Distribute remaining patterns
                if n_remaining > 0:
                    if shuffle:
                        assignment_of_remaining_patterns = torch.randperm(
                            n_experiences
                        ).tolist()[:n_remaining]
                    else:
                        assignment_of_remaining_patterns = range(n_remaining)
                    for exp_id in assignment_of_remaining_patterns:
                        pattern_idx = patterns_order[prev_idx]
                        pattern_target = targets_order[prev_idx]
                        exp_patterns[exp_id].append(pattern_idx)

                        self.exp_structure[exp_id][pattern_target] += 1
                        prev_idx += 1

        self.n_patterns_per_experience = [
            len(exp_patterns[exp_id]) for exp_id in range(n_experiences)
        ]

        self._classes_in_exp = None  # Will be lazy initialized later

        train_experiences = []
        train_task_labels = []
        for t_id, exp_def in enumerate(exp_patterns):
            if self._has_task_labels:
                train_task_labels.append(t_id)
            else:
                train_task_labels.append(0)

            exp_task_labels = ConstantSequence(
                train_task_labels[-1], len(train_dataset)
            )
            train_experiences.append(
                _taskaware_classification_subset(
                    train_dataset, indices=exp_def, task_labels=exp_task_labels
                )
            )

        self.train_exps_patterns_assignment = exp_patterns
        """ A list containing which training instances are assigned to each
        experience in the train stream. Instances are identified by their id 
        w.r.t. the dataset found in the original_train_dataset field. """

        super().__init__(
            stream_definitions={
                "train": (train_experiences, train_task_labels, train_dataset),
                "test": (test_dataset, [0], test_dataset),
            },
            complete_test_set_only=True,
            stream_factory=NIStream,
            experience_factory=NIExperience,
        )

    def get_reproducibility_data(self) -> Dict[str, Any]:
        reproducibility_data = {
            "exps_patterns_assignment": self.train_exps_patterns_assignment,
            "has_task_labels": bool(self._has_task_labels),
        }
        return reproducibility_data


class NIStream(ClassificationStream["NIExperience"]):
    def __init__(
        self,
        name: str,
        benchmark: NIScenario,
        *,
        slice_ids: Optional[List[int]] = None,
        set_stream_info: bool = True
    ):
        self.benchmark: NIScenario = benchmark
        super().__init__(
            name=name,
            benchmark=benchmark,
            slice_ids=slice_ids,
            set_stream_info=set_stream_info,
        )


class NIExperience(ClassificationExperience[TaskAwareSupervisedClassificationDataset]):
    """
    Defines a "New Instances" experience. It defines fields to obtain the
    current dataset and the associated task label. It also keeps a reference
    to the stream from which this experience was taken.
    """

    def __init__(
        self,
        origin_stream: NIStream,
        current_experience: int,
    ):
        """
        Creates a ``NIExperience`` instance given the stream from this
        experience was taken and and the current experience ID.

        :param origin_stream: The stream from which this experience was
            obtained.
        :param current_experience: The current experience ID, as an integer.
        """

        self._benchmark: NIScenario = origin_stream.benchmark

        super().__init__(origin_stream, current_experience)

    @property  # type: ignore[override]
    def benchmark(self) -> NIScenario:
        bench = self._benchmark
        NIExperience._check_unset_attribute("benchmark", bench)
        return bench

    @benchmark.setter
    def benchmark(self, bench: NIScenario):
        self._benchmark = bench


__all__ = ["NIScenario", "NIStream", "NIExperience"]
