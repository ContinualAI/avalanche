################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

from typing import Sequence, List, Optional, Dict, Any, Set

import torch

from avalanche.benchmarks.scenarios.deprecated.classification_scenario import (
    ClassificationScenario,
    ClassificationStream,
    ClassificationExperience,
)
from avalanche.benchmarks.utils.classification_dataset import (
    _taskaware_classification_subset,
)
from avalanche.benchmarks.utils.classification_dataset import (
    TaskAwareClassificationDataset,
    TaskAwareSupervisedClassificationDataset,
)

from avalanche.benchmarks.utils.flat_data import ConstantSequence


class NCScenario(
    ClassificationScenario[
        "NCStream", "NCExperience", TaskAwareSupervisedClassificationDataset
    ]
):

    """
    This class defines a "New Classes" scenario. Once created, an instance
    of this class can be iterated in order to obtain the experience sequence
    under the form of instances of :class:`NCExperience`.

    This class can be used directly. However, we recommend using facilities like
    :func:`avalanche.benchmarks.generators.nc_benchmark`.
    """

    def __init__(
        self,
        train_dataset: TaskAwareClassificationDataset,
        test_dataset: TaskAwareClassificationDataset,
        n_experiences: int,
        task_labels: bool,
        shuffle: bool = True,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        per_experience_classes: Optional[Dict[int, int]] = None,
        class_ids_from_zero_from_first_exp: bool = False,
        class_ids_from_zero_in_each_exp: bool = False,
        reproducibility_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates a ``NCGenericScenario`` instance given the training and test
        Datasets and the number of experiences.

        By default, the number of classes will be automatically detected by
        looking at the training Dataset ``targets`` field. Classes will be
        uniformly distributed across ``n_experiences`` unless a
        ``per_experience_classes`` argument is specified.

        The number of classes must be divisible without remainder by the number
        of experiences. This also applies when the ``per_experience_classes``
        argument is not None.

        :param train_dataset: The training dataset. The dataset must be a
            subclass of :class:`AvalancheDataset`. For instance, one can
            use the datasets from the torchvision package like that:
            ``train_dataset=AvalancheDataset(torchvision_dataset)``.
        :param test_dataset: The test dataset. The dataset must be a
            subclass of :class:`AvalancheDataset`. For instance, one can
            use the datasets from the torchvision package like that:
            ``test_dataset=AvalancheDataset(torchvision_dataset)``.
        :param n_experiences: The number of experiences.
        :param task_labels: If True, each experience will have an ascending task
            label. If False, the task label will be 0 for all the experiences.
        :param shuffle: If True, the class order will be shuffled. Defaults to
            True.
        :param seed: If shuffle is True and seed is not None, the class order
            will be shuffled according to the seed. When None, the current
            PyTorch random number generator state will be used.
            Defaults to None.
        :param fixed_class_order: If not None, the class order to use (overrides
            the shuffle argument). Very useful for enhancing
            reproducibility. Defaults to None.
        :param per_experience_classes: Is not None, a dictionary whose keys are
            (0-indexed) experience IDs and their values are the number of
            classes to include in the respective experiences. The dictionary
            doesn't have to contain a key for each experience! All the remaining
            experiences will contain an equal amount of the remaining classes.
            The remaining number of classes must be divisible without remainder
            by the remaining number of experiences. For instance,
            if you want to include 50 classes in the first experience
            while equally distributing remaining classes across remaining
            experiences, just pass the "{0: 50}" dictionary as the
            per_experience_classes parameter. Defaults to None.
        :param class_ids_from_zero_from_first_exp: If True, original class IDs
            will be remapped so that they will appear as having an ascending
            order. For instance, if the resulting class order after shuffling
            (or defined by fixed_class_order) is [23, 34, 11, 7, 6, ...] and
            class_ids_from_zero_from_first_exp is True, then all the patterns
            belonging to class 23 will appear as belonging to class "0",
            class "34" will be mapped to "1", class "11" to "2" and so on.
            This is very useful when drawing confusion matrices and when dealing
            with algorithms with dynamic head expansion. Defaults to False.
            Mutually exclusive with the ``class_ids_from_zero_in_each_exp``
            parameter.
        :param class_ids_from_zero_in_each_exp: If True, original class IDs
            will be mapped to range [0, n_classes_in_exp) for each experience.
            Defaults to False. Mutually exclusive with the
            ``class_ids_from_zero_from_first_exp parameter``.
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

        if not isinstance(train_dataset, TaskAwareSupervisedClassificationDataset):
            train_dataset = TaskAwareSupervisedClassificationDataset(train_dataset)
        if not isinstance(test_dataset, TaskAwareSupervisedClassificationDataset):
            test_dataset = TaskAwareSupervisedClassificationDataset(test_dataset)

        if class_ids_from_zero_from_first_exp and class_ids_from_zero_in_each_exp:
            raise ValueError(
                "Invalid mutually exclusive options "
                "class_ids_from_zero_from_first_exp and "
                "class_ids_from_zero_in_each_exp set at the "
                "same time"
            )
        if reproducibility_data:
            n_experiences = reproducibility_data["n_experiences"]

        if n_experiences < 1:
            raise ValueError(
                "Invalid number of experiences (n_experiences "
                "parameter): must be greater than 0"
            )

        self.classes_order: List[int] = []
        """ Stores the class order (remapped class IDs). """

        self.classes_order_original_ids: List[int] = torch.unique(
            torch.as_tensor(train_dataset.targets), sorted=True
        ).tolist()
        """ Stores the class order (original class IDs) """

        n_original_classes = max(self.classes_order_original_ids) + 1

        self.class_mapping: List[int] = []
        """
        class_mapping stores the class mapping so that 
        `mapped_class_id = class_mapping[original_class_id]`. 
        
        If the benchmark is created with an amount of classes which is less than
        the amount of all classes in the dataset, then class_mapping will 
        contain some -1 values corresponding to ignored classes. This can
        happen when passing a fixed class order to the constructor.
        """

        self.n_classes_per_exp: List[int] = []
        """ A list that, for each experience (identified by its index/ID),
            stores the number of classes assigned to that experience. """

        self._classes_in_exp: List[Set[int]] = []

        self.original_classes_in_exp: List[Set[int]] = []
        """
        A list that, for each experience (identified by its index/ID), stores a 
        set of the original IDs of classes assigned to that experience. 
        This field applies to both train and test streams.
        """

        self.class_ids_from_zero_from_first_exp: bool = (
            class_ids_from_zero_from_first_exp
        )
        """ If True the class IDs have been remapped to start from zero. """

        self.class_ids_from_zero_in_each_exp: bool = class_ids_from_zero_in_each_exp
        """ If True the class IDs have been remapped to start from zero in 
        each experience """

        # Note: if fixed_class_order is None and shuffle is False,
        # the class order will be the one encountered
        # By looking at the train_dataset targets field
        if reproducibility_data:
            self.classes_order_original_ids = reproducibility_data[
                "classes_order_original_ids"
            ]
            self.class_ids_from_zero_from_first_exp = reproducibility_data[
                "class_ids_from_zero_from_first_exp"
            ]
            self.class_ids_from_zero_in_each_exp = reproducibility_data[
                "class_ids_from_zero_in_each_exp"
            ]
        elif fixed_class_order is not None:
            # User defined class order -> just use it
            if len(
                set(self.classes_order_original_ids).union(set(fixed_class_order))
            ) != len(self.classes_order_original_ids):
                raise ValueError("Invalid classes defined in fixed_class_order")

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
            self.classes_order_original_ids = torch.as_tensor(
                self.classes_order_original_ids
            )[torch.randperm(len(self.classes_order_original_ids))].tolist()

        self.n_classes: int = len(self.classes_order_original_ids)
        """ The number of classes """

        if reproducibility_data:
            self.n_classes_per_exp = reproducibility_data["n_classes_per_exp"]
        elif per_experience_classes is not None:
            # per_experience_classes is a user-defined dictionary that defines
            # the number of classes to include in some (or all) experiences.
            # Remaining classes are equally distributed across the other
            # experiences.
            #
            # Format of per_experience_classes dictionary:
            #   - key = experience id
            #   - value = number of classes for this experience

            if (
                max(per_experience_classes.keys()) >= n_experiences
                or min(per_experience_classes.keys()) < 0
            ):
                # The dictionary contains a key (that is, a experience id) >=
                # the number of requested experiences... or < 0
                raise ValueError(
                    "Invalid experience id in per_experience_classes parameter:"
                    " experience ids must be in range [0, n_experiences)"
                )
            if min(per_experience_classes.values()) < 0:
                # One or more values (number of classes for each experience) < 0
                raise ValueError(
                    "Wrong number of classes defined for one or "
                    "more experiences: must be a non-negative "
                    "value"
                )

            if sum(per_experience_classes.values()) > self.n_classes:
                # The sum of dictionary values (n. of classes for each
                # experience) >= the number of classes
                raise ValueError(
                    "Insufficient number of classes: "
                    "per_experience_classes parameter can't "
                    "be satisfied"
                )

            # Remaining classes are equally distributed across remaining
            # experiences. This amount of classes must be be divisible without
            # remainder by the number of remaining experiences
            remaining_exps = n_experiences - len(per_experience_classes)
            if (
                remaining_exps > 0
                and (self.n_classes - sum(per_experience_classes.values()))
                % remaining_exps
                > 0
            ):
                raise ValueError(
                    "Invalid number of experiences: remaining "
                    "classes cannot be divided by n_experiences"
                )

            # default_per_exp_classes is the default amount of classes
            # for the remaining experiences
            if remaining_exps > 0:
                default_per_exp_classes = (
                    self.n_classes - sum(per_experience_classes.values())
                ) // remaining_exps
            else:
                default_per_exp_classes = 0

            # Initialize the self.n_classes_per_exp list using
            # "default_per_exp_classes" as the default
            # amount of classes per experience. Then, loop through the
            # per_experience_classes dictionary to set the customized,
            # user defined, classes for the required experiences.
            self.n_classes_per_exp = [default_per_exp_classes] * n_experiences
            for exp_id in per_experience_classes:
                self.n_classes_per_exp[exp_id] = per_experience_classes[exp_id]
        else:
            # Classes will be equally distributed across the experiences
            # The amount of classes must be be divisible without remainder
            # by the number of experiences
            if self.n_classes % n_experiences > 0:
                raise ValueError(
                    f"Invalid number of experiences: classes contained in "
                    f"dataset ({self.n_classes}) cannot be divided by "
                    f"n_experiences ({n_experiences})"
                )
            self.n_classes_per_exp = [self.n_classes // n_experiences] * n_experiences

        # Before populating the classes_in_experience list,
        # define the remapped class IDs.
        if reproducibility_data:
            # Method 0: use reproducibility data
            self.classes_order = reproducibility_data["classes_order"]
            self.class_mapping = reproducibility_data["class_mapping"]
        elif self.class_ids_from_zero_from_first_exp:
            # Method 1: remap class IDs so that they appear in ascending order
            # over all experiences
            self.classes_order = list(range(0, self.n_classes))
            self.class_mapping = [-1] * n_original_classes
            for class_id in range(n_original_classes):
                # This check is needed because, when a fixed class order is
                # used, the user may have defined an amount of classes less than
                # the overall amount of classes in the dataset.
                if class_id in self.classes_order_original_ids:
                    self.class_mapping[
                        class_id
                    ] = self.classes_order_original_ids.index(class_id)
        elif self.class_ids_from_zero_in_each_exp:
            # Method 2: remap class IDs so that they appear in range [0, N] in
            # each experience
            self.classes_order = []
            self.class_mapping = [-1] * n_original_classes
            next_class_idx = 0
            for exp_id, exp_n_classes in enumerate(self.n_classes_per_exp):
                self.classes_order += list(range(exp_n_classes))
                for exp_class_idx in range(exp_n_classes):
                    original_class_position = next_class_idx + exp_class_idx
                    original_class_id = self.classes_order_original_ids[
                        original_class_position
                    ]
                    self.class_mapping[original_class_id] = exp_class_idx
                next_class_idx += exp_n_classes
        else:
            # Method 3: no remapping of any kind
            # remapped_id = class_mapping[class_id] -> class_id == remapped_id
            self.classes_order = self.classes_order_original_ids
            self.class_mapping = list(range(0, n_original_classes))

        original_training_dataset = train_dataset
        original_test_dataset = test_dataset

        # Populate the _classes_in_exp and original_classes_in_exp lists
        # "_classes_in_exp[exp_id]": list of (remapped) class IDs assigned
        # to experience "exp_id"
        # "original_classes_in_exp[exp_id]": list of original class IDs
        # assigned to experience "exp_id"
        for exp_id in range(n_experiences):
            classes_start_idx = sum(self.n_classes_per_exp[:exp_id])
            classes_end_idx = classes_start_idx + self.n_classes_per_exp[exp_id]

            self._classes_in_exp.append(
                set(self.classes_order[classes_start_idx:classes_end_idx])
            )
            self.original_classes_in_exp.append(
                set(self.classes_order_original_ids[classes_start_idx:classes_end_idx])
            )

        # Finally, create the experience -> patterns assignment.
        # In order to do this, we don't load all the patterns
        # instead we use the targets field.
        train_exps_patterns_assignment = []
        test_exps_patterns_assignment = []

        self._has_task_labels = task_labels
        if reproducibility_data is not None:
            self._has_task_labels = bool(reproducibility_data["has_task_labels"])

        pattern_train_task_labels: Sequence[int]
        pattern_test_task_labels: Sequence[int]
        if self._has_task_labels:
            pattern_train_task_labels = [-1] * len(train_dataset)
            pattern_test_task_labels = [-1] * len(test_dataset)
        else:
            pattern_train_task_labels = ConstantSequence(0, len(train_dataset))
            pattern_test_task_labels = ConstantSequence(0, len(test_dataset))

        for exp_id in range(n_experiences):
            selected_classes = self.original_classes_in_exp[exp_id]
            selected_indexes_train = []
            for sc in selected_classes:
                train_tgt_idx = original_training_dataset.targets.val_to_idx[sc]
                selected_indexes_train.extend(train_tgt_idx)
                if self._has_task_labels:
                    for idx in train_tgt_idx:
                        pattern_train_task_labels[idx] = exp_id

            selected_indexes_test = []
            for sc in selected_classes:
                test_tgt_idx = original_test_dataset.targets.val_to_idx[sc]
                selected_indexes_test.extend(test_tgt_idx)
                if self._has_task_labels:
                    for idx in test_tgt_idx:
                        pattern_test_task_labels[idx] = exp_id

            train_exps_patterns_assignment.append(selected_indexes_train)
            test_exps_patterns_assignment.append(selected_indexes_test)

        # Good idea, but doesn't work
        # transform_groups = train_eval_transforms(train_dataset, test_dataset)
        #
        # train_dataset = train_dataset\
        #     .replace_transforms(*transform_groups['train'], group='train') \
        #     .replace_transforms(*transform_groups['eval'], group='eval')
        #
        # test_dataset = test_dataset \
        #     .replace_transforms(*transform_groups['train'], group='train') \
        #     .replace_transforms(*transform_groups['eval'], group='eval')

        train_dataset = _taskaware_classification_subset(
            train_dataset,
            class_mapping=self.class_mapping,
            initial_transform_group="train",
        )
        test_dataset = _taskaware_classification_subset(
            test_dataset,
            class_mapping=self.class_mapping,
            initial_transform_group="eval",
        )

        self.train_exps_patterns_assignment = train_exps_patterns_assignment
        """ A list containing which training instances are assigned to each
        experience in the train stream. Instances are identified by their id 
        w.r.t. the dataset found in the original_train_dataset field. """

        self.test_exps_patterns_assignment = test_exps_patterns_assignment
        """ A list containing which test instances are assigned to each
        experience in the test stream. Instances are identified by their id 
        w.r.t. the dataset found in the original_test_dataset field. """

        train_experiences: List[TaskAwareClassificationDataset] = []
        train_task_labels: List[int] = []
        for t_id, exp_def in enumerate(train_exps_patterns_assignment):
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

        test_experiences: List[TaskAwareClassificationDataset] = []
        test_task_labels: List[int] = []
        for t_id, exp_def in enumerate(test_exps_patterns_assignment):
            if self._has_task_labels:
                test_task_labels.append(t_id)
            else:
                test_task_labels.append(0)

            exp_task_labels = ConstantSequence(test_task_labels[-1], len(test_dataset))
            test_experiences.append(
                _taskaware_classification_subset(
                    test_dataset, indices=exp_def, task_labels=exp_task_labels
                )
            )

        super().__init__(
            stream_definitions={
                "train": (train_experiences, train_task_labels, train_dataset),
                "test": (test_experiences, test_task_labels, test_dataset),
            },
            stream_factory=NCStream,
            experience_factory=NCExperience,
        )

    def get_reproducibility_data(self):
        reproducibility_data = {
            "class_ids_from_zero_from_first_exp": bool(
                self.class_ids_from_zero_from_first_exp
            ),
            "class_ids_from_zero_in_each_exp": bool(
                self.class_ids_from_zero_in_each_exp
            ),
            "class_mapping": self.class_mapping,
            "classes_order": self.classes_order,
            "classes_order_original_ids": self.classes_order_original_ids,
            "n_classes_per_exp": self.n_classes_per_exp,
            "n_experiences": int(self.n_experiences),
            "has_task_labels": self._has_task_labels,
        }
        return reproducibility_data

    def classes_in_exp_range(
        self, exp_start: int, exp_end: Optional[int] = None
    ) -> List[int]:
        """
        Gets a list of classes contained in the given experiences. The
        experiences are defined by range. This means that only the classes in
        range [exp_start, exp_end) will be included.

        :param exp_start: The starting experience ID.
        :param exp_end: The final experience ID. Can be None, which means that
            all the remaining experiences will be taken.

        :returns: The classes contained in the required experience range.
        """
        # Ref: https://stackoverflow.com/a/952952
        if exp_end is None:
            return [
                item
                for sublist in self.classes_in_experience["train"][exp_start:]
                for item in sublist
            ]

        return [
            item
            for sublist in self.classes_in_experience["train"][exp_start:exp_end]
            for item in sublist
        ]


class NCStream(ClassificationStream["NCExperience"]):
    def __init__(
        self,
        name: str,
        benchmark: NCScenario,
        *,
        slice_ids: Optional[List[int]] = None,
        set_stream_info: bool = True,
    ):
        self.benchmark: NCScenario = benchmark
        super().__init__(
            name=name,
            benchmark=benchmark,
            slice_ids=slice_ids,
            set_stream_info=set_stream_info,
        )


class NCExperience(ClassificationExperience[TaskAwareSupervisedClassificationDataset]):
    """
    Defines a "New Classes" experience. It defines fields to obtain the current
    dataset and the associated task label. It also keeps a reference to the
    stream from which this experience was taken.
    """

    def __init__(self, origin_stream: NCStream, current_experience: int):
        """
        Creates a ``NCExperience`` instance given the stream from this
        experience was taken and and the current experience ID.

        :param origin_stream: The stream from which this experience was
            obtained.
        :param current_experience: The current experience ID, as an integer.
        """

        self._benchmark: NCScenario = origin_stream.benchmark

        super().__init__(origin_stream, current_experience)

    @property  # type: ignore[override]
    def benchmark(self) -> NCScenario:
        bench = self._benchmark
        NCExperience._check_unset_attribute("benchmark", bench)
        return bench

    @benchmark.setter
    def benchmark(self, bench: NCScenario):
        self._benchmark = bench


__all__ = ["NCScenario", "NCStream", "NCExperience"]
