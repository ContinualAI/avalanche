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

import random
from pathlib import Path
from typing import Sequence, Optional, Union, Any

from torchvision import transforms

from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)

from avalanche.benchmarks.datasets.external_datasets.cifar import (
    get_cifar100_dataset,
    get_cifar10_dataset,
)
from avalanche.benchmarks.utils.classification_dataset import (
    _concat_taskaware_classification_datasets_sequentially,
)

from avalanche.benchmarks import nc_benchmark, NCScenario

_default_cifar100_train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]
)

_default_cifar100_eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]
)


def SplitCIFAR100(
    n_experiences: int,
    *,
    first_exp_with_half_classes: bool = False,
    return_task_id=False,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    shuffle: bool = True,
    class_ids_from_zero_in_each_exp: bool = False,
    class_ids_from_zero_from_first_exp: bool = False,
    train_transform: Optional[Any] = _default_cifar100_train_transform,
    eval_transform: Optional[Any] = _default_cifar100_eval_transform,
    dataset_root: Optional[Union[str, Path]] = None
):
    """
    Creates a CL benchmark using the CIFAR100 dataset.

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

    :param n_experiences: The number of incremental experiences in the current
        benchmark. The value of this parameter should be a divisor of 100 if
        first_task_with_half_classes is False, a divisor of 50 otherwise.
    :param first_exp_with_half_classes: A boolean value that indicates if a
        first pretraining batch containing half of the classes should be used.
        If it's True, a pretraining experience with half of the classes (50 for
        cifar100) is used. If this parameter is False no pretraining task
        will be used, and the dataset is simply split into a the number of
        experiences defined by the parameter n_experiences. Default to False.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to True.
    :param class_ids_from_zero_in_each_exp: If True, original class IDs
        will be mapped to range [0, n_classes_in_exp) for each experience.
        Defaults to False. Mutually exclusive with the
        ``class_ids_from_zero_from_first_exp`` parameter.
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
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'cifar100' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """
    cifar_train, cifar_test = get_cifar100_dataset(dataset_root)

    return nc_benchmark(
        train_dataset=cifar_train,
        test_dataset=cifar_test,
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        fixed_class_order=fixed_class_order,
        shuffle=shuffle,
        per_exp_classes={0: 50} if first_exp_with_half_classes else None,
        class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
        class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


def SplitCIFAR110(
    n_experiences: int,
    *,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    class_ids_from_zero_from_first_exp: bool = False,
    train_transform: Optional[Any] = _default_cifar100_train_transform,
    eval_transform: Optional[Any] = _default_cifar100_eval_transform,
    dataset_root_cifar10: Optional[Union[str, Path]] = None,
    dataset_root_cifar100: Optional[Union[str, Path]] = None
) -> NCScenario:
    """
    Creates a CL benchmark using both the CIFAR100 and CIFAR10 datasets.

    If the datasets are not present in the computer, this method will
    automatically download and store them in the data folder.

    The CIFAR10 dataset is used to create the first experience, while the
    remaining `n_experiences-1` experiences will be created from CIFAR100.

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator will apply a task label 0 to all experiences.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label (always 0 for this specific
    benchmark).

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of experiences for the entire benchmark.
        The first experience will contain the entire CIFAR10 dataset, while the
        other n-1 experiences will be obtained from CIFAR100.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order ONLY for the incremental part, which is based on cifar100. The
        classes must be in range 0-99.
        If None, value of ``seed`` will be used to define the class order for
        the incremental batches on cifar100. If non-None, ``seed`` parameter
        will be ignored. Defaults to None.
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
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root_cifar10: The root path of the CIFAR-10 dataset.
        Defaults to None, which means that the default location for
        'cifar10' will be used.
    :param dataset_root_cifar100: The root path of the CIFAR-100 dataset.
        Defaults to None, which means that the default location for
        'cifar100' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    cifar10_train, cifar10_test = get_cifar10_dataset(dataset_root_cifar10)
    cifar100_train, cifar100_test = get_cifar100_dataset(dataset_root_cifar100)

    (
        cifar_10_100_train,
        cifar_10_100_test,
        _,
    ) = _concat_taskaware_classification_datasets_sequentially(
        [cifar10_train, cifar100_train], [cifar10_test, cifar100_test]
    )
    # cifar10 classes
    class_order = [_ for _ in range(10)]
    # if a class order is defined (for cifar100) the given class labels are
    # appended to the class_order list, adding 10 to them (since the classes
    # 0-9 are the classes of cifar10).
    if fixed_class_order is not None:
        class_order.extend([c + 10 for c in fixed_class_order])
    else:
        random.seed(seed)
        # random shuffling of the cifar100 classes (labels 10-109)
        cifar_100_class_order = random.sample(range(10, 110), 100)
        class_order.extend(cifar_100_class_order)

    return nc_benchmark(
        cifar_10_100_train,
        cifar_10_100_test,
        n_experiences=n_experiences,
        task_labels=False,
        shuffle=False,
        seed=None,
        fixed_class_order=class_order,
        class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
        per_exp_classes={0: 10},
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


if __name__ == "__main__":
    import sys

    print("Split 100")
    benchmark_instance = SplitCIFAR100(5)
    check_vision_benchmark(benchmark_instance)

    print("Split 110")
    benchmark_instance = SplitCIFAR110(5)
    check_vision_benchmark(benchmark_instance)

    sys.exit(0)


__all__ = ["SplitCIFAR100", "SplitCIFAR110"]
