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

"""Generic definitions for CL benchmarks defined via list of datasets."""

import random
from avalanche.benchmarks.utils.data import AvalancheDataset
import torch
from itertools import tee
from typing import (
    Callable,
    Generator,
    Generic,
    List,
    Sequence,
    TypeVar,
    Union,
    Tuple,
    Optional,
    Iterable,
    Dict,
)

from .generic_scenario import EagerCLStream, CLScenario, CLExperience, make_stream
from ..utils import TaskAwareSupervisedClassificationDataset


TCLDataset = TypeVar("TCLDataset", bound="AvalancheDataset")


def benchmark_from_datasets(**dataset_streams: Sequence[TCLDataset]) -> CLScenario:
    """Creates a benchmark given a list of datasets for each stream.

    Each dataset will be considered as a separate experience.
    Contents of the datasets must already be set, including task labels.
    Transformations will be applied if defined.

    Avalanche benchmarks usually provide at least a train and test stream,
    but this generator is fully generic.

    To use this generator, you must convert your data into an Avalanche Dataset.

    :param dataset_streams: A dictionary with stream-name as key and
        list-of-datasets as values, where stream-name is the name of the stream,
        while list-of-datasets is a list of Avalanche datasets, where
        list-of-datasets[i] contains the data for experience i.
    """
    exps_streams = []
    for stream_name, data_s in dataset_streams.items():
        for dd in data_s:
            if not isinstance(dd, AvalancheDataset):
                raise ValueError("datasets must be AvalancheDatasets")
        des = [
            DatasetExperience(dataset=dd, current_experience=eid)
            for eid, dd in enumerate(data_s)
        ]
        s = EagerCLStream(stream_name, des)
        exps_streams.append(s)
    return CLScenario(exps_streams)


class DatasetExperience(CLExperience, Generic[TCLDataset]):
    """An Experience that provides a dataset."""

    def __init__(
        self, *, dataset: TCLDataset, current_experience: Optional[int] = None
    ):
        super().__init__(current_experience=current_experience, origin_stream=None)
        self._dataset: AvalancheDataset = dataset

    @property
    def dataset(self) -> AvalancheDataset:
        # dataset is a read-only property
        data = self._dataset
        return data


def _split_dataset_by_attribute(
    data: TCLDataset, attr_name: str
) -> Dict[int, TCLDataset]:
    """Helper to split a dataset by attribute.

    :param data: an Avalanche dataset.
    :param attr_name: the name of the attribute of `data` to use for splitting `data`.
    """
    da = getattr(data, attr_name)
    dds = {}
    for el in da.uniques:
        idxs = da.val_to_idx[el]
        dds[el] = data.subset(idxs)
    return dds


def split_validation_random(
    validation_size: Union[int, float],
    shuffle: bool,
    seed: Optional[int] = None,
    dataset: Optional[AvalancheDataset] = None,
) -> Tuple[AvalancheDataset, AvalancheDataset]:
    """Splits an `AvalancheDataset` in two splits.

    The default splitting strategy used by
    :func:`benchmark_with_validation_stream`.

    This splitting strategy simply splits the datasets in two (e.g. a
    train and validation split) of size `validation_size`.

    When taking inspiration for your custom splitting strategy, please consider
    that all parameters preceding `experience` are filled by
    :func:`benchmark_with_validation_stream` by using `partial` from the
    `functools` standard library. A custom splitting strategy must have only
    a single parameter: the experience. Consider wrapping your custom
    splitting strategy with `partial` if more parameters are needed.

    You can use this split strategy with methdos that require a custom
    split strategy such as :func:`benchmark_with_validation_stream`to split
    a benchmark with::

        validation_size = 0.2
        foo = lambda exp: split_validation_class_balanced(validation_size, exp)
        bm = benchmark_with_validation_stream(bm, split_strategy=foo)

    :param validation_size: The number of instances to allocate to the
    validation experience. Can be an int value or a float between 0 and 1.
    :param shuffle: If True, instances will be shuffled before splitting.
        Otherwise, the first instances will be allocated to the training
        dataset by leaving the last ones to the validation dataset.
    :param dataset: The dataset to split.
    :return: A tuple containing 2 elements: the new training and validation
        datasets.
    """
    if dataset is None:
        raise ValueError("dataset must be provided")
    exp_indices = list(range(len(dataset)))

    if seed is None:
        seed = random.randint(0, 1000000)
    g = torch.Generator()
    g.manual_seed(seed)

    if shuffle:
        exp_indices = torch.as_tensor(exp_indices)[
            torch.randperm(len(exp_indices), generator=g)
        ].tolist()

    if 0.0 <= validation_size <= 1.0:
        valid_n_instances = int(validation_size * len(dataset))
    else:
        valid_n_instances = int(validation_size)
        if valid_n_instances > len(dataset):
            raise ValueError(
                f"Can't split the dataset: not enough "
                f"instances. Required {valid_n_instances}, got only"
                f"{len(dataset)}"
            )

    train_n_instances = len(dataset) - valid_n_instances
    d1 = dataset.subset(exp_indices[:train_n_instances])
    d2 = dataset.subset(exp_indices[train_n_instances:])
    return d1, d2


def split_validation_class_balanced(
    validation_size: Union[int, float],
    dataset: TaskAwareSupervisedClassificationDataset,
) -> Tuple[
    TaskAwareSupervisedClassificationDataset, TaskAwareSupervisedClassificationDataset
]:
    """Class-balanced dataset split.

    This splitting strategy splits `dataset` into train and validation data of
    size `validation_size` using a class-balanced split.
    Samples of each class are chosen randomly.

    You can use this split strategy to split a benchmark with::

        validation_size = 0.2
        foo = lambda data: split_validation_class_balanced(validation_size, data)
        bm = benchmark_with_validation_stream(bm, split_strategy=foo)

    :param validation_size: The percentage of samples to allocate to the
        validation experience as a float between 0 and 1.
    :param dataset: The dataset to split.
    :return: A tuple containing 2 elements: the new training and validation
        datasets.
    """
    if not isinstance(validation_size, float):
        raise ValueError("validation_size must be an integer")
    if not 0.0 <= validation_size <= 1.0:
        raise ValueError("validation_size must be a float in [0, 1].")

    if validation_size > len(dataset):
        raise ValueError(
            f"Can't create the validation experience: not enough "
            f"instances. Required {validation_size}, got only"
            f"{len(dataset)}"
        )
    exp_indices = list(range(len(dataset)))
    targets_as_tensor = torch.as_tensor(dataset.targets)
    exp_classes: List[int] = targets_as_tensor.unique().tolist()

    # shuffle exp_indices
    exp_indices_t = torch.as_tensor(exp_indices)[torch.randperm(len(exp_indices))]
    # shuffle the targets as well
    exp_targets = targets_as_tensor[exp_indices_t]

    train_exp_indices: list[int] = []
    valid_exp_indices: list[int] = []
    for cid in exp_classes:  # split indices for each class separately.
        c_indices = exp_indices_t[exp_targets == cid]
        valid_n_instances = int(validation_size * len(c_indices))
        valid_exp_indices.extend(c_indices[:valid_n_instances])
        train_exp_indices.extend(c_indices[valid_n_instances:])

    result_train_dataset = dataset.subset(train_exp_indices)
    result_valid_dataset = dataset.subset(valid_exp_indices)
    return result_train_dataset, result_valid_dataset


class LazyTrainValSplitter:
    def __init__(
        self,
        split_strategy: Callable[
            [AvalancheDataset],
            Tuple[AvalancheDataset, AvalancheDataset],
        ],
        experiences: Iterable[DatasetExperience],
    ) -> None:
        """
        Creates a generator operating around the split strategy and the
        experiences stream.

        :param split_strategy: The strategy used to split each experience in train
            and validation datasets.
        :return: A generator returning a 2 elements tuple (the train and validation
            datasets).
        """
        self.split_strategy = split_strategy
        self.experiences = experiences

    def __iter__(
        self,
    ) -> Generator[Tuple[AvalancheDataset, AvalancheDataset], None, None]:
        for new_experience in self.experiences:
            yield self.split_strategy(new_experience.dataset)


def benchmark_with_validation_stream(
    benchmark: CLScenario,
    validation_size: Union[int, float] = 0.5,
    shuffle: bool = False,
    seed: Optional[int] = None,
    split_strategy: Optional[
        Callable[[AvalancheDataset], Tuple[AvalancheDataset, AvalancheDataset]]
    ] = None,
) -> CLScenario:
    """Helper to obtain a benchmark with a validation stream.

    This generator accepts an existing benchmark instance and returns a version
    of it in which the train stream has been split into training and validation
    streams.

    Each train/validation experience will be by splitting the original training
    experiences. Patterns selected for the validation experience will be removed
    from the training experiences.

    The default splitting strategy is a random split as implemented by `split_validation_random`.
    If you want to use class balancing you can use `split_validation_class_balanced`, or
    use a custom `split_strategy`, as shown in the following example::

        validation_size = 0.2
        foo = lambda exp: split_dataset_class_balanced(validation_size, exp)
        bm = benchmark_with_validation_stream(bm, custom_split_strategy=foo)

    :param benchmark: The benchmark to split.
    :param validation_size: The size of the validation experience, as an int
        or a float between 0 and 1. Ignored if `custom_split_strategy` is used.
    :param shuffle: If True, patterns will be allocated to the validation
        stream randomly. This will use the default PyTorch random number
        generator at its current state. Defaults to False. Ignored if
        `custom_split_strategy` is used. If False, the first instances will be
        allocated to the training  dataset by leaving the last ones to the
        validation dataset.
    :param split_strategy: A function that implements a custom splitting
        strategy. The function must accept an AvalancheDataset and return a tuple
        containing the new train and validation dataset. By default, the splitting
        strategy will split the data according to `validation_size` and `shuffle`).
        A good starting to understand the mechanism is to look at the
        implementation of the standard splitting function
        :func:`random_validation_split_strategy`.

    :return: A benchmark instance in which the validation stream has been added.
    """

    if split_strategy is None:
        if seed is None:
            seed = random.randint(0, 1000000)

        # functools.partial is a more compact option
        # However, MyPy does not understand what a partial is -_-
        def random_validation_split_strategy_wrapper(data):
            return split_validation_random(validation_size, shuffle, seed, data)

        split_strategy = random_validation_split_strategy_wrapper
    else:
        split_strategy = split_strategy

    stream = benchmark.streams["train"]
    if isinstance(stream, EagerCLStream):  # eager split
        train_exps, valid_exps = [], []

        exp: DatasetExperience
        for exp in stream:
            train_data, valid_data = split_strategy(exp.dataset)
            train_exps.append(DatasetExperience(dataset=train_data))
            valid_exps.append(DatasetExperience(dataset=valid_data))
    else:  # Lazy splitting (based on a generator)
        split_generator = LazyTrainValSplitter(split_strategy, stream)
        train_exps = (DatasetExperience(dataset=a) for a, _ in split_generator)
        valid_exps = (DatasetExperience(dataset=b) for _, b in split_generator)

    train_stream = make_stream(name="train", exps=train_exps)
    valid_stream = make_stream(name="valid", exps=valid_exps)
    other_streams = benchmark.streams

    del other_streams["train"]
    return CLScenario(
        streams=[train_stream, valid_stream] + list(other_streams.values())
    )


__all__ = [
    "_split_dataset_by_attribute",
    "benchmark_from_datasets",
    "DatasetExperience",
    "split_validation_random",
    "benchmark_with_validation_stream",
]
