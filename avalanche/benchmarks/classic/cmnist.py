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
from os.path import expanduser
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from avalanche.benchmarks.scenarios.new_classes.scenario_creation import \
    create_nc_single_dataset_sit_scenario, \
    create_nc_single_dataset_multi_task_scenario, \
    create_nc_multi_dataset_multi_task_scenario
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import \
    NCMultiTaskScenario


_default_mnist_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

_default_mnist_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def SplitMNIST(
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


def PermutedMNIST(
        incremental_steps: int,
        seed: Optional[int] = None,
        train_transform=_default_mnist_train_transform,
        test_transform=_default_mnist_test_transform) -> NCMultiTaskScenario:

    """
    This helper create a permuted MNIST scenario: where a given number of random
    pixel permutations is used to permute the MNIST images in
    `incremental_steps` different manners, creating an equal number of tasks.
    Each task is composed of all the original MNIST 10 classes, but the pixel
    in the images are permuted in different ways in every task.
    If the dataset is not present in the computer the method automatically
    download it and store the data in the data folder.

    :param incremental_steps: The number of incremental tasks in the current
        scenario. It indicates how many different permutations of the MNIST
        dataset have to be created.
        The value of this parameter should be a divisor of 10.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param train_transform: The transformation to apply to the training data
        before the random permutation, e.g. a random crop, a normalization or a
        concatenation of different transformations (see torchvision.transform
        documentation for a comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param test_transform: The transformation to apply to the test data
        before the random permutation, e.g. a random crop, a normalization or a
        concatenation of different transformations (see torchvision.transform
        documentation for a comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.

    :returns: A :class:`NCMultiTaskScenario` instance initialized for the the
        MT permuted MNIST scenario.
    """

    list_train_dataset = []
    list_test_dataset = []
    rng_permute = np.random.RandomState(seed)

    # for every incremental step
    for _ in range(incremental_steps):
        # choose a random permutation of the pixels in the image
        idx_permute = torch.from_numpy(rng_permute.permutation(784)).type(
            torch.int64)

        # add the permutation to the default dataset transformation
        train_transform_list = train_transform.transforms.copy()
        train_transform_list.append(
            transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1, 28, 28))
        )
        new_train_transform = transforms.Compose(train_transform_list)

        test_transform_list = test_transform.transforms.copy()
        test_transform_list.append(
            transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1, 28, 28))
        )
        new_test_transform = transforms.Compose(test_transform_list)

        # get the datasets with the constructed transformation
        permuted_train, permuted_test = _get_mnist_dataset(new_train_transform,
                                                           new_test_transform)
        list_train_dataset.append(permuted_train)
        list_test_dataset.append(permuted_test)

    return create_nc_multi_dataset_multi_task_scenario(
        train_dataset_list=list_train_dataset,
        test_dataset_list=list_test_dataset,
        shuffle=False,
        classes_ids_from_zero_in_each_task=True
    )


def RotatedMNIST(
        incremental_steps: int,
        seed: Optional[int] = None,
        rotations_list: Optional[Sequence[int]] = None,
        train_transform=_default_mnist_train_transform,
        test_transform=_default_mnist_test_transform) -> NCMultiTaskScenario:

    """
    This helper create a rotated MNIST scenario: where a given number of random
    rotations are used to rotate the MNIST images in
    `incremental_steps` different manners, creating an equal number of tasks.
    Each task is composed of all the original MNIST 10 classes, but the images
    are rotated in different ways and using different values in every task.
    If the dataset is not present in the computer the method automatically
    download it and store the data in the data folder.

    :param incremental_steps: The number of incremental tasks in the current
        scenario. It indicates how many different rotations of the MNIST
        dataset have to be created.
        The value of this parameter should be a divisor of 10.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param rotations_list: A list of rotations values in degrees (from -180 to
        180) used to define the rotations. The rotation specified in position
        0 of the list will be applieed to the task 0, the rotation specified in
        position 1 will be applyed to task 1 and so on.
        If None, value of ``seed`` will be used to define the rotations.
        If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param train_transform: The transformation to apply to the training data
        after the random rotation, e.g. a random crop, a normalization or a
        concatenation of different transformations (see torchvision.transform
        documentation for a comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param test_transform: The transformation to apply to the test data
        after the random rotation, e.g. a random crop, a normalization or a
        concatenation of different transformations (see torchvision.transform
        documentation for a comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.

    :returns: A :class:`NCMultiTaskScenario` instance initialized for the the
        MT rotated MNIST scenario.
    """

    assert len(rotations_list) == incremental_steps, "The number of rotations" \
                                                     " should match the number"\
                                                     " of incremental steps."
    assert all(-180 <= rotations_list[i] <= 180
               for i in range(len(rotations_list))), "The value of a rotation" \
                                                     " should be between -180" \
                                                     " and 180 degrees."

    list_train_dataset = []
    list_test_dataset = []
    rng_rotate = np.random.RandomState(seed)

    # for every incremental step
    for step in range(incremental_steps):
        if rotations_list is not None:
            rotation_angle = rotations_list[step]
        else:
            # choose a random rotation of the pixels in the image
            rotation_angle = rng_rotate.randint(-180, 181)

        # add the rotation to the default dataset transformation in position
        # 0, since it works on images and not tensors.
        train_transform_list = train_transform.transforms.copy()
        train_transform_list.insert(
            0,
            transforms.RandomRotation(degrees=(rotation_angle, rotation_angle))
        )
        new_train_transform = transforms.Compose(train_transform_list)

        test_transform_list = test_transform.transforms.copy()
        test_transform_list.insert(
            0,
            transforms.RandomRotation(degrees=(rotation_angle, rotation_angle))
        )
        new_test_transform = transforms.Compose(test_transform_list)

        # get the datasets with the constructed transformation
        rotated_train, rotated_test = _get_mnist_dataset(new_train_transform,
                                                         new_test_transform)
        list_train_dataset.append(rotated_train)
        list_test_dataset.append(rotated_test)

    return create_nc_multi_dataset_multi_task_scenario(
        train_dataset_list=list_train_dataset,
        test_dataset_list=list_test_dataset,
        shuffle=False,
        classes_ids_from_zero_in_each_task=True
    )


def _get_mnist_dataset(train_transformation, test_transformation):
    mnist_train = MNIST(root=expanduser("~")+"/.avalanche/data/mnist/",
                        train=True,
                        download=True, transform=train_transformation)
    mnist_test = MNIST(root=expanduser("~")+"/.avalanche/data/mnist/",
                       train=False,
                       download=True, transform=test_transformation)
    return mnist_train, mnist_test
