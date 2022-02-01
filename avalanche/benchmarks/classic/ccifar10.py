################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24-06-2020                                                             #
# Author(s): Gabriele Graffieti                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
from pathlib import Path
from typing import Sequence, Optional, Union, Any
from torchvision.datasets import CIFAR10
from torchvision import transforms

from avalanche.benchmarks import nc_benchmark, NCScenario
from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)
from avalanche.benchmarks.datasets import default_dataset_location

_default_cifar10_train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

_default_cifar10_eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)


def SplitCIFAR10(
    n_experiences: int,
    *,
    first_exp_with_half_classes: bool = False,
    return_task_id=False,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    shuffle: bool = True,
    train_transform: Optional[Any] = _default_cifar10_train_transform,
    eval_transform: Optional[Any] = _default_cifar10_eval_transform,
    dataset_root: Union[str, Path] = None
) -> NCScenario:
    """
    Creates a CL benchmark using the CIFAR10 dataset.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of experiences in the current benchmark.
        The value of this parameter should be a divisor of 10 if
        `first_task_with_half_classes` is False, a divisor of 5 otherwise.
    :param first_exp_with_half_classes: A boolean value that indicates if a
        first pretraining step containing half of the classes should be used.
        If it's True, the first experience will use half of the classes (5 for
        cifar10). If this parameter is False, no pretraining step will be
        used and the dataset is simply split into a the number of experiences
        defined by the parameter n_experiences. Defaults to False.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If not None, the ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to false.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default eval transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'cifar10' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """
    cifar_train, cifar_test = _get_cifar10_dataset(dataset_root)

    if return_task_id:
        return nc_benchmark(
            train_dataset=cifar_train,
            test_dataset=cifar_test,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            per_exp_classes={0: 5} if first_exp_with_half_classes else None,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
    else:
        return nc_benchmark(
            train_dataset=cifar_train,
            test_dataset=cifar_test,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            per_exp_classes={0: 5} if first_exp_with_half_classes else None,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )


def _get_cifar10_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location("cifar10")

    train_set = CIFAR10(dataset_root, train=True, download=True)
    test_set = CIFAR10(dataset_root, train=False, download=True)

    return train_set, test_set


if __name__ == "__main__":
    import sys

    benchmark_instance = SplitCIFAR10(5)
    check_vision_benchmark(benchmark_instance)
    sys.exit(0)

__all__ = ["SplitCIFAR10"]
