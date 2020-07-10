################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24-06-2020                                                             #
# Author(s): Gabriele Graffieti, Vincenzo Lomonaco                             #
# E-mail: contact@continualai.org                                              #
# Website: wwww.continualai.org                                                #
################################################################################

""" This module implements an high-level function to create the classic
Fashion MNIST split CL scenario. """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from typing import Sequence, Optional
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from avalanche.benchmarks.scenarios.new_classes.scenario_creation import \
    create_nc_single_dataset_sit_scenario, \
    create_nc_single_dataset_multi_task_scenario

_default_cifar10_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
])

_default_cifar10_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])


def create_fmnist_benchmark(incremental_steps: int,
                            first_batch_with_half_classes: bool = False,
                            return_task_id=False,
                            seed: Optional[int] = None,
                            fixed_class_order: Optional[Sequence[int]] = None,
                            train_transform=_default_cifar10_train_transform,
                            test_transform=_default_cifar10_test_transform
                            ):
    """
    Creates a CL scenario using the Fashion MNIST dataset.
    If the dataset is not present in the computer the method automatically
    download it and store the data in the data folder.

    :param incremental_steps: The number of incremental steps in the current
        scenario. If the first step is a "pretrain" step and it contains
        half of the classes, the number of incremental steps is the number of
        tasks performed after the pretraining task.
        The value of this parameter should be a divisor of 10 if
        first_task_with_half_classes if false, a divisor of 5 otherwise.
    :param first_batch_with_half_classes: A boolean value that indicates if a
        first pretraining batch containing half of the classes should be used.
        If it's True, a pretrain batch with half of the classes (5 for
        cifar100) is used, and a number of incremental tasks, given by the
        parameter incremental_task is constructed. If this paramenter is False
        no pretraining task will be used, and the dataset is simply split into
        a the number of steps defined by the parameter incremental_steps.
        Default to False.
    :param return_task_id: if True, for every step the task id is returned and
        the Scenario is Multi Task. This means that the scenario returned
        will be of type ``NCMultiTaskScenario``. If false the task index is
        not returned (default to 0 for every batch) and the returned scenario
        is of type ``NCSingleTaskScenario``.
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
    :param test_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.

    :returns: A :class:`NCMultiTaskScenario` instance initialized for the the
        MT scenario using CIFAR10 if the parameter ``return_task_id`` is True,
        a :class:`NCSingleTaskScenario` initialized for the SIT scenario using
        CIFAR10 otherwise.
    """
    cifar_train, cifar_test = _get_fmnist_dataset(train_transform,
                                                  test_transform)
    total_steps = incremental_steps + 1 if first_batch_with_half_classes \
        else incremental_steps
    if return_task_id:
        return create_nc_single_dataset_multi_task_scenario(
            train_dataset=cifar_train,
            test_dataset=cifar_test,
            n_tasks=total_steps,
            seed=seed,
            fixed_class_order=fixed_class_order,
            per_task_classes={0: 5} if first_batch_with_half_classes else None)
    else:
        return create_nc_single_dataset_sit_scenario(
            train_dataset=cifar_train,
            test_dataset=cifar_test,
            n_batches=total_steps,
            seed=seed,
            fixed_class_order=fixed_class_order,
            per_batch_classes={0: 5} if first_batch_with_half_classes else None
        )


def _get_fmnist_dataset(train_transformation, test_transformation):
    train_set = FashionMNIST('./data/fmnist', train=True,
                             download=True, transform=train_transformation)
    test_set = FashionMNIST('./data/fmnist', train=False,
                            download=True, transform=test_transformation)
    return train_set, test_set


if __name__ == "__main__":

    nc_scenario = create_fmnist_benchmark(incremental_steps=10)

    for i, batch in enumerate(nc_scenario):
        print(i, batch)
