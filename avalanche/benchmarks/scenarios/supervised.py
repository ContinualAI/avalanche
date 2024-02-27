################################################################################
# Copyright (c) 2023 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 15-09-2023                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""High-level benchmark generators for supervised scenarios such as class-incremental."""
import warnings
from copy import copy
from typing import (
    Iterable,
    Sequence,
    Optional,
    Dict,
    List,
    Protocol,
)

import torch

from avalanche.benchmarks.utils.classification_dataset import (
    ClassificationDataset,
    _as_taskaware_supervised_classification_dataset,
)
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from .dataset_scenario import _split_dataset_by_attribute, DatasetExperience
from .generic_scenario import CLScenario, CLStream, EagerCLStream


def class_incremental_benchmark(
    datasets_dict: Dict[str, ClassificationDataset],
    *,
    class_order: Optional[Sequence[int]] = None,
    num_experiences: Optional[int] = None,
    num_classes_per_exp: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
) -> CLScenario:
    """Splits datasets according to a class-incremental scenario.

    Each dataset will create a stream with the same class order.

    :param datasets_dict: A dictionary with stream names as keys (str) and
        AvalancheDataset as values. Usually, you want to provide at least train
        and test stream.
    :param class_order: List of classes that determine the order of appearance
        in the stream. If `None`, random classes will be used.
        Defaults to None (random classes).
    :param num_experiences: desired number of experiences in the stream.
    :param num_classes_per_exp: If not None, a list with the number of classes
        to pick for each experience.
    :param seed: The seed to use for random shuffling if `class_order is None`.
        If None, the current PyTorch random number generator state will be used.
        Defaults to None.

    :return: A class-incremental :class:`CLScenario`.
    """
    if (class_order is not None) and (seed is not None):
        raise ValueError("Can't set `seed` if a fixed `class_order` is given.")
    if (num_classes_per_exp is not None) and (num_experiences is not None):
        raise ValueError(
            "Only one of `num_classes_per_exp` or `num_experiences` can be used."
        )
    if (num_classes_per_exp is None) and (num_experiences is None):
        raise ValueError(
            "One of `num_classes_per_exp` or `num_experiences` must be set."
        )
    if num_experiences is not None and num_experiences < 1:
        raise ValueError(
            "Invalid number of experiences (n_experiences "
            "parameter): must be greater than 0"
        )

    # convert to avalanche datasets
    for name, dd in datasets_dict.items():
        if not isinstance(dd, AvalancheDataset):
            datasets_dict[name] = _as_taskaware_supervised_classification_dataset(dd)

    # validate classes
    dd_classes = list(datasets_dict.values())[0].targets.uniques
    num_classes: int = 1 + max(list(datasets_dict.values())[0].targets.uniques)
    if (num_classes_per_exp is not None) and (num_classes != sum(num_classes_per_exp)):
        raise ValueError(
            "`sum(num_classes_per_exp)` must be equal to the total number of classes."
        )
    for dd in datasets_dict.values():  # all datasets have the same classes
        clss = dd.targets.uniques
        if dd_classes != clss:
            raise ValueError("`datasets` must all have the same classes")

    # pick random class order if needed
    if class_order is None:  # sample random class order
        if seed is not None:
            torch.random.manual_seed(seed)
        class_order = torch.randperm(num_classes).tolist()

    # split classes by experience
    classes_exp_assignment = []
    if num_experiences is not None:
        assert num_classes_per_exp is None, "BUG: num_classes_per_exp must be None"
        curr_classess_per_exp: int = num_classes // num_experiences
        for eid in range(num_experiences):
            if eid == 0:
                classes_exp_assignment.append(class_order[:curr_classess_per_exp])
            else:
                # final exp will take reminder of classes if they don't divide equally
                start_idx = curr_classess_per_exp * eid
                end_idx = start_idx + curr_classess_per_exp
                classes_exp_assignment.append(class_order[start_idx:end_idx])
    elif num_classes_per_exp is not None:
        num_curr = 0
        for eid, num_classes in enumerate(num_classes_per_exp):
            curr_classes = class_order[num_curr : num_curr + num_classes]
            classes_exp_assignment.append(curr_classes)
            num_curr += num_classes

    # create the streams using class_order to split the data
    streams = []
    for name, dd in datasets_dict.items():
        curr_stream = []
        data_by_class = _split_dataset_by_attribute(dd, "targets")
        for eid, clss in enumerate(classes_exp_assignment):
            curr_data: ClassificationDataset = ClassificationDataset(
                [], data_attributes=[DataAttribute([], "targets")]
            )
            for cls in clss:
                # TODO: curr_data.concat(data_by_class[cls]) is bugged and removes targets
                curr_data = data_by_class[cls].concat(curr_data)
            curr_stream.append(DatasetExperience(dataset=curr_data))
        streams.append(EagerCLStream(name, curr_stream))
    return with_classes_timeline(CLScenario(streams))


def _class_balanced_indices(
    data: ClassificationDataset,
    num_experiences: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """class-balanced indices.

    Internal helper for `new_instances_benchmark`.

    :param data: the `AvalancheDataset` to split
    :param num_experiences: length of the stream
    :param shuffle: -
    :param seed: -
    """
    if seed is not None:
        torch.random.manual_seed(seed)

    # Validate function arguments
    if num_experiences < 1:
        raise ValueError(
            "Invalid number of experiences (n_experiences "
            "parameter): must be greater than 0"
        )

    # experience -> idxs assignment
    exps_idxs: List[List[int]] = [[] for _ in range(num_experiences)]
    # TODO: fix pycharm type hints
    for class_id, class_idxs in data.targets.val_to_idx.items():
        # INVARIANT: class_idxs keeps only indices that are not assigned yet.
        # Whenever we add idxs to an experience, we remove them from class_idxs

        if shuffle:  # shuffle each class
            perm = torch.randperm(len(class_idxs))
            class_idxs = torch.as_tensor(class_idxs)[perm].tolist()

        # distribute equally each class to experiences
        npats = len(class_idxs) // num_experiences
        for eid in range(num_experiences):
            exps_idxs[eid].extend(class_idxs[:npats])
            class_idxs = class_idxs[npats:]

        # distribute remainder if not divisible by num_experiences
        if len(class_idxs) > 0:
            if shuffle:
                exps_remaining: Iterable[int] = torch.randperm(
                    num_experiences
                ).tolist()[: len(class_idxs)]
            else:
                exps_remaining = range(len(class_idxs))
            for eid in exps_remaining:
                exps_idxs[eid].append(class_idxs[0])
                class_idxs = class_idxs[1:]

    # sort to keep original order instead of class-order
    for eid in range(len(exps_idxs)):
        exps_idxs[eid].sort()
    return exps_idxs


def _random_indices(
    data,
    num_experiences: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
    min_class_patterns_in_exp: int = 0,
) -> List[List[int]]:
    """Random indices splitter.

    Internal helper for `new_instances_benchmark.

    :param min_class_patterns_in_exp: the random split must respect the
        constraint of having at least `min_min_class_patterns_in_exp`
        samples per class.

    :return: a list of indices for each experience.
    """
    if seed is not None:
        torch.random.manual_seed(seed)

    ##############################
    # Validate function arguments
    ##############################
    if num_experiences < 1:
        raise ValueError(
            "Invalid number of experiences (n_experiences "
            "parameter): must be greater than 0"
        )
    if min_class_patterns_in_exp < 0:
        raise ValueError(
            "Invalid min_class_patterns_in_exp parameter: "
            "must be greater than or equal to 0"
        )

    ##############################
    # patterns -> experience assignment for train stream
    ##############################
    idxs_per_class = data.targets.val_to_idx

    # experience->idxs assignment
    exps_idxs: List[List[int]] = [[] for _ in range(num_experiences)]

    # validate `min_class_patterns_in_exp` argument
    min_class_patterns = min([len(el) for el in idxs_per_class.values()])
    if min_class_patterns < num_experiences * min_class_patterns_in_exp:
        raise ValueError("min_class_patterns_in_exp constraint " "can't be satisfied")

    for class_id, class_idxs in idxs_per_class.items():
        # INVARIANT: class_idxs keeps only indices that are not assigned yet. Whenever we add idxs to an experience, we remove them from class_idxs

        # first assign exactly min_class_patterns_in_exp.
        for eid in range(num_experiences):
            exps_idxs[eid].extend(class_idxs[:min_class_patterns_in_exp])
            class_idxs = class_idxs[min_class_patterns_in_exp:]

        # distribute equally among experiences
        samples_per_exp = len(class_idxs) // num_experiences
        for eid in range(num_experiences):
            exps_idxs[eid].extend(class_idxs[:samples_per_exp])
            class_idxs = class_idxs[samples_per_exp:]

        # distribute remaining patterns
        if len(class_idxs) > 0:
            if shuffle:
                exps_remaining: Iterable[int] = torch.randperm(
                    num_experiences
                ).tolist()[: len(class_idxs)]
            else:
                exps_remaining = range(len(class_idxs))

            for eid in exps_remaining:
                exps_idxs[eid].append(class_idxs[0])
                class_idxs = class_idxs[1:]

    # sort to keep original order instead of class-order
    for ii in range(len(exps_idxs)):
        exps_idxs[ii].sort()
    return exps_idxs


def new_instances_benchmark(
    train_dataset: ClassificationDataset,
    test_dataset: AvalancheDataset,
    num_experiences: int,
    *,
    shuffle: bool = True,
    seed: Optional[int] = None,
    balance_experiences: bool = False,
    min_class_patterns_in_exp: int = 0,
) -> CLScenario:
    """Benchmark generator for "New Instances" (NI) scenarios.

    Given a `train_dataset` and a `test_dataset, the generator creates a
    benchmark where the training stream is split according to the
    New Instances setting.

    Notice that we don't split the test dataset in this generator because we
    have random splits, so it is more natural to test on the full test set
    at each step instead of an i.i.d. random test split.

    :param train_dataset: An AvalancheDataset used to define the training stream.
    :param test_dataset: A test AvalancheDataset. This will not be split.
    :param num_experiences: The desired stream length.
    :param shuffle: If True, patterns order will be shuffled.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param balance_experiences: If True, pattern of each class will be equally
        spread across all experiences. If False, patterns will be assigned to
        experiences in a complete random way. Defaults to False.
    :param min_class_patterns_in_exp: The minimum amount of patterns of
        every class that must be assigned to every experience. Compatible with
        the ``balance_experiences`` parameter. An exception will be raised if
        this constraint can't be satisfied. Defaults to 0.

    :return: A properly initialized :class:`NIScenario` instance.
    """

    if balance_experiences:  # class-balanced split
        exps_idxs = _class_balanced_indices(
            data=train_dataset,
            num_experiences=num_experiences,
            shuffle=shuffle,
            seed=seed,
        )
    else:
        exps_idxs = _random_indices(
            data=train_dataset,
            num_experiences=num_experiences,
            shuffle=shuffle,
            seed=seed,
            min_class_patterns_in_exp=min_class_patterns_in_exp,
        )

    train_experiences = []
    for idxs in exps_idxs:
        curr_data = train_dataset.subset(indices=idxs)
        train_experiences.append(DatasetExperience(dataset=curr_data))

    train_stream = CLStream("train", train_experiences)
    test_stream = CLStream("test", [DatasetExperience(dataset=test_dataset)])
    return CLScenario(streams=[train_stream, test_stream])


__all__ = [
    "class_incremental_benchmark",
    "new_instances_benchmark",
]


class ClassesTimeline(Protocol):
    """Experience decorator that provides info about classes occurrence over time."""

    @property
    def classes_in_this_experience(self) -> List[int]:
        """The list of classes in this experience."""
        ...

    @property
    def previous_classes(self) -> List[int]:
        """The list of classes in previous experiences."""
        ...

    @property
    def classes_seen_so_far(self) -> List[int]:
        """List of classes of current and previous experiences."""
        ...

    @property
    def future_classes(self) -> List[int]:
        """The list of classes of next experiences."""
        ...


def with_classes_timeline(obj):
    """Add `ClassesTimeline` attributes.

    `obj` must be a scenario or a stream.
    """

    def _decorate_benchmark(obj: CLScenario):
        new_streams = []
        for s in obj.streams.values():
            new_streams.append(_decorate_stream(s))
        return CLScenario(new_streams)

    def _decorate_stream(obj: CLStream):
        # TODO: support stream generators. Should return a new generators which applies
        #  foo_decorate_exp every time a new experience is generated.
        new_stream = []
        if not isinstance(obj, EagerCLStream):
            warnings.warn("stream generator will be converted to a list.")

        # compute set of all classes in the stream
        all_cls: set[int] = set()
        for exp in obj:
            all_cls = all_cls.union(exp.dataset.targets.uniques)

        prev_cls: set[int] = set()
        for exp in obj:
            new_exp = copy(exp)
            curr_cls = exp.dataset.targets.uniques

            new_exp.classes_in_this_experience = curr_cls
            new_exp.previous_classes = set(prev_cls)
            new_exp.classes_seen_so_far = curr_cls.union(prev_cls)
            # TODO: future_classes ignores repetitions right now...
            #  implement and test scenario with repetitions
            new_exp.future_classes = all_cls.difference(new_exp.classes_seen_so_far)
            new_stream.append(new_exp)

            prev_cls = prev_cls.union(curr_cls)
        return EagerCLStream(obj.name, new_stream)

    if isinstance(obj, CLScenario):
        return _decorate_benchmark(obj)
    elif isinstance(obj, CLStream):
        return _decorate_stream(obj)
    else:
        raise ValueError(
            "Unsupported object type: must be one of {CLScenario, CLStream}"
        )


__all__ = [
    "class_incremental_benchmark",
    "new_instances_benchmark",
    "with_classes_timeline",
]
