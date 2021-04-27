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
from typing import Sequence, Optional
from os.path import expanduser
from torchvision.datasets import CIFAR100
from torchvision import transforms

from avalanche.benchmarks.datasets import CIFAR10
from avalanche.benchmarks.utils.avalanche_dataset import \
    concat_datasets_sequentially, train_eval_avalanche_datasets
from avalanche.benchmarks import nc_benchmark, NCScenario

_default_cifar100_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

_default_cifar100_eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])


def SplitCIFAR100(n_experiences: int,
                  first_exp_with_half_classes: bool = False,
                  return_task_id=False,
                  seed: Optional[int] = None,
                  fixed_class_order: Optional[Sequence[int]] = None,
                  train_transform=_default_cifar100_train_transform,
                  eval_transform=_default_cifar100_eval_transform):
    """
    Creates a CL scenario using the CIFAR100 dataset.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned scenario will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).

    The scenario instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The scenario API is quite simple and is uniform across all scenario
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of incremental experiences in the current
        scenario. The value of this parameter should be a divisor of 100 if
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

    :returns: A properly initialized :class:`NCScenario` instance.
    """
    cifar_train, cifar_test = _get_cifar100_dataset(
        train_transform, eval_transform
    )

    if return_task_id:
        return nc_benchmark(
            train_dataset=cifar_train,
            test_dataset=cifar_test,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            per_exp_classes={0: 50} if first_exp_with_half_classes else None,
            class_ids_from_zero_in_each_exp=True)
    else:
        return nc_benchmark(
            train_dataset=cifar_train,
            test_dataset=cifar_test,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            per_exp_classes={0: 50} if first_exp_with_half_classes else None)


def SplitCIFAR110(
        n_experiences: int,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        train_transform=_default_cifar100_train_transform,
        eval_transform=_default_cifar100_eval_transform) -> NCScenario:
    """
    Creates a CL scenario using both the CIFAR100 and CIFAR10 datasets.

    If the datasets are not present in the computer, this method will
    automatically download and store them in the data folder.

    The CIFAR10 dataset is used to create the first experience, while the
    remaining `n_experiences-1` experiences will be created from CIFAR100.

    The returned scenario will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator will apply a task label "0" to all experiences.

    The scenario instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label (always "0" for this specific
    benchmark).

    The scenario API is quite simple and is uniform across all scenario
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of experiences for the entire scenario.
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

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    cifar10_train, cifar10_test = _get_cifar10_dataset(
        train_transform, eval_transform)

    cifar100_train, cifar100_test = _get_cifar100_dataset(
        train_transform, eval_transform)

    cifar_10_100_train, cifar_10_100_test, _ = concat_datasets_sequentially(
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
        cifar_10_100_train, cifar_10_100_test,
        n_experiences=n_experiences,
        task_labels=False,
        shuffle=False,
        seed=None,
        fixed_class_order=class_order,
        per_exp_classes={0: 10})


def _get_cifar10_dataset(train_transformation, eval_transformation):
    train_set = CIFAR10(expanduser("~") + "/.avalanche/data/cifar10/",
                        train=True, download=True)

    test_set = CIFAR10(expanduser("~") + "/.avalanche/data/cifar10/",
                       train=False, download=True)

    return train_eval_avalanche_datasets(
        train_set, test_set, train_transformation, eval_transformation)


def _get_cifar100_dataset(train_transformation, eval_transformation):
    train_set = CIFAR100(expanduser("~") + "/.avalanche/data/cifar100/",
                         train=True, download=True)

    test_set = CIFAR100(expanduser("~") + "/.avalanche/data/cifar100/",
                        train=False, download=True)

    return train_eval_avalanche_datasets(
        train_set, test_set, train_transformation, eval_transformation)


__all__ = [
    'SplitCIFAR100',
    'SplitCIFAR110'
]
