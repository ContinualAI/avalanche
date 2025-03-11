import random

from pathlib import Path
from typing import Optional, Union, Any, List, TypeVar

from torchvision import transforms

from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.classification_dataset import (
    _as_taskaware_supervised_classification_dataset,
)
from avalanche.benchmarks import benchmark_from_datasets, CLScenario

from avalanche.benchmarks.datasets.concon import ConConDataset


TCLDataset = TypeVar("TCLDataset", bound="AvalancheDataset")


_default_train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

_default_eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def build_concon_scenario(
    list_train_dataset: List[TCLDataset],
    list_test_dataset: List[TCLDataset],
    seed: Optional[int] = None,
    n_experiences: int = 3,
    shuffle_order: bool = False,
):
    if shuffle_order and not n_experiences == 1:
        random.seed(seed)
        random.shuffle(list_train_dataset)
        random.seed(seed)
        random.shuffle(list_test_dataset)

    if n_experiences == 1:
        new_list_train_dataset = []
        new_list_train_dataset.append(list_train_dataset[0])

        for i in range(1, len(list_train_dataset)):
            new_list_train_dataset[0] = new_list_train_dataset[0].concat(
                list_train_dataset[i]
            )

        list_train_dataset = new_list_train_dataset

        new_list_test_dataset = []
        new_list_test_dataset.append(list_test_dataset[0])

        for i in range(1, len(list_test_dataset)):
            new_list_test_dataset[0] = new_list_test_dataset[0].concat(
                list_test_dataset[i]
            )

        list_test_dataset = new_list_test_dataset

    return benchmark_from_datasets(train=list_train_dataset, test=list_test_dataset)


def ConConDisjoint(
    n_experiences: int,
    *,
    seed: Optional[int] = None,
    shuffle_order: bool = False,
    train_transform: Optional[Any] = _default_train_transform,
    eval_transform: Optional[Any] = _default_eval_transform,
    dataset_root: Optional[Union[str, Path]] = None,
) -> CLScenario:
    """
    Creates a ConCon Disjoint benchmark.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned benchmark will be a domain-incremental one, where each task
    is a different domain with different confounders. In this setting,
    task-specific confounders never appear in other tasks.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    :param dataset_root: The root directory of the dataset.
    :param n_experiences: The number of experiences to use.
    :param seed: The seed to use.
    :param shuffle_order: Whether to shuffle the order of the experiences.
    :param train_transform: The training transform to use.
    :param eval_transform: The evaluation transform to use.

    :returns: The ConCon Disjoint benchmark.
    """
    assert (
        n_experiences == 3 or n_experiences == 1
    ), "n_experiences must be 1 or 3 for ConCon Disjoint"
    list_train_dataset = []
    list_test_dataset = []

    for i in range(3):
        train_dataset = ConConDataset("disjoint", i, root=dataset_root, train=True)
        test_dataset = ConConDataset("disjoint", i, root=dataset_root, train=False)
        train_dataset = _as_taskaware_supervised_classification_dataset(
            train_dataset, transform=train_transform
        )
        test_dataset = _as_taskaware_supervised_classification_dataset(
            test_dataset, transform=eval_transform
        )
        list_train_dataset.append(train_dataset)
        list_test_dataset.append(test_dataset)

    return build_concon_scenario(
        list_train_dataset,
        list_test_dataset,
        seed=seed,
        n_experiences=n_experiences,
        shuffle_order=shuffle_order,
    )


def ConConStrict(
    n_experiences: int,
    *,
    seed: Optional[int] = None,
    shuffle_order: bool = False,
    train_transform: Optional[Any] = _default_train_transform,
    eval_transform: Optional[Any] = _default_eval_transform,
    dataset_root: Optional[Union[str, Path]] = None,
) -> CLScenario:
    """
    Creates a ConCon Strict benchmark.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned benchmark will be a domain-incremental one, where each task
    is a different domain with different confounders. In this setting,
    task-specific confounders may appear in other tasks as random features
    in both positive and negative samples.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    :param dataset_root: The root directory of the dataset.
    :param n_experiences: The number of experiences to use.
    :param seed: The seed to use.
    :param shuffle_order: Whether to shuffle the order of the experiences.
    :param train_transform: The training transform to use.
    :param eval_transform: The evaluation transform to use.

    :returns: The ConCon Strict benchmark.
    """
    assert (
        n_experiences == 3 or n_experiences == 1
    ), "n_experiences must be 1 or 3 for ConCon Disjoint"
    list_train_dataset = []
    list_test_dataset = []

    for i in range(3):
        train_dataset = ConConDataset("strict", i, root=dataset_root, train=True)
        test_dataset = ConConDataset("strict", i, root=dataset_root, train=False)
        train_dataset = _as_taskaware_supervised_classification_dataset(
            train_dataset, transform=train_transform
        )
        test_dataset = _as_taskaware_supervised_classification_dataset(
            test_dataset, transform=eval_transform
        )
        list_train_dataset.append(train_dataset)
        list_test_dataset.append(test_dataset)

    return build_concon_scenario(
        list_train_dataset,
        list_test_dataset,
        seed=seed,
        n_experiences=n_experiences,
        shuffle_order=shuffle_order,
    )


def ConConUnconfounded(
    *,
    train_transform: Optional[Any] = _default_train_transform,
    eval_transform: Optional[Any] = _default_eval_transform,
    dataset_root: Optional[Union[str, Path]] = None,
) -> CLScenario:
    """
    Creates a ConCon Unconfounded benchmark.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned benchmark will only contain one task, where no task-specific
    confounders are present.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    :param dataset_root: The root directory of the dataset.
    :param train_transform: The training transform to use.
    :param eval_transform: The evaluation transform to use.

    :returns: The ConCon Unconfounded benchmark.
    """
    train_dataset = []
    test_dataset = []

    train_dataset.append(
        ConConDataset("unconfounded", 0, root=dataset_root, train=True)
    )
    test_dataset.append(
        ConConDataset("unconfounded", 0, root=dataset_root, train=False)
    )

    train_dataset[0] = _as_taskaware_supervised_classification_dataset(
        train_dataset[0], transform=train_transform
    )

    test_dataset[0] = _as_taskaware_supervised_classification_dataset(
        test_dataset[0], transform=eval_transform
    )

    return benchmark_from_datasets(train=train_dataset, test=test_dataset)


__all__ = [
    "ConConDisjoint",
    "ConConStrict",
    "ConConUnconfounded",
]
