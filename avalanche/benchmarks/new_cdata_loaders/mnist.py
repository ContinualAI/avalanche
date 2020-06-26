#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 26-06-2020                                                             #
# Author(s): Gabriele Graffieti                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


from typing import Optional, Sequence

from torchvision.datasets import MNIST
from torchvision import transforms
from avalanche.benchmarks.scenarios.new_classes.scenario_creation import \
    create_nc_single_dataset_sit_scenario, \
    create_nc_single_dataset_multi_task_scenario


_default_mnist_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

_default_mnist_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def create_split_mnist_benchmark(
        incremental_steps: int,
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        train_transform=_default_mnist_train_transform,
        test_transform=_default_mnist_test_transform):
    """
    Creates a CL scenario using the MNIST dataset.
    This helper create the basic split MNIST scenario, where the 10 classes of
    the MNIST dataset are evenly splitted into the given nuber of tasks.
    If the dataset is not present in the computer the method automatically
    download it and store the data in the data folder.

    :param incremental_steps: The number of incremental steps in the current
        scenario.
        The value of this parameter should be a divisor of 10.
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
        MT split MNIST scenario if the parameter ``return_task_id`` is True,
        a :class:`NCSingleTaskScenario` initialized for the SIT split MNIST
        scenario otherwise.
    """

    mnist_train, mnist_test = _get_mnist_dataset(train_transform,
                                                 test_transform)
    if return_task_id:
        return create_nc_single_dataset_multi_task_scenario(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            n_tasks=incremental_steps,
            seed=seed,
            fixed_class_order=fixed_class_order,
        )
    else:
        return create_nc_single_dataset_sit_scenario(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            n_batches=incremental_steps,
            seed=seed,
            fixed_class_order=fixed_class_order,
        )


def _get_mnist_dataset(train_transformation, test_transformation):
    mnist_train = MNIST(root="./data/MNIST", train=True,
                        download=True, transform=train_transformation)
    mnist_test = MNIST(root="./data/MNIST", train=False,
                       download=True, transform=test_transformation)
    return mnist_train, mnist_test
