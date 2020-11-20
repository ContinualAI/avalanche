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


from typing import Optional, Sequence, Union, Any
from os.path import expanduser
import torch
from PIL.Image import Image
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize, \
    RandomRotation

from avalanche.benchmarks import NCScenario, nc_scenario
from avalanche.benchmarks.utils import train_test_transformation_datasets

_default_mnist_train_transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

_default_mnist_test_transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])


class PixelsPermutation(object):
    """
    Apply a fixed permutation to the pixels of the given image.

    Works with both Tensors and PIL images. Returns an object of the same type
    of the input element.
    """

    def __init__(self, index_permutation: Sequence[int]):
        self.permutation = index_permutation
        self._to_tensor = ToTensor()
        self._to_image = ToPILImage()

    def __call__(self, img: Union[Image, Tensor]):
        is_image = isinstance(img, Image)
        if (not is_image) and (not isinstance(img, Tensor)):
            raise ValueError('Invalid input: must be a PIL image or a Tensor')

        if is_image:
            img = self._to_tensor(img)

        img = img.view(-1)[self.permutation].view(*img.shape)

        if is_image:
            img = self._to_image(img)

        return img


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

    mnist_train, mnist_test = _get_mnist_dataset(
        train_transform, test_transform)

    if return_task_id:
        return nc_scenario(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            n_steps=incremental_steps,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            class_ids_from_zero_in_each_step=True)
    else:
        return nc_scenario(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            n_steps=incremental_steps,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order)


def PermutedMNIST(
        incremental_steps: int,
        seed: Optional[int] = None,
        train_transform: Any = _default_mnist_train_transform,
        test_transform: Any = _default_mnist_test_transform) -> NCScenario:
    """
    This helper create a permuted MNIST scenario: where a given number of random
    pixel permutations is used to permute the MNIST images in
    ``incremental_steps`` different manners, creating an equal number of tasks.
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

        permutation = PixelsPermutation(idx_permute)

        mnist_train, mnist_test = _get_mnist_dataset(permutation, permutation)

        # Freeze the permutation, then add the user defined transformations
        permuted_train = mnist_train \
            .freeze_transforms() \
            .add_transforms(train_transform)

        permuted_test = mnist_test \
            .freeze_transforms() \
            .add_transforms(test_transform)

        list_train_dataset.append(permuted_train)
        list_test_dataset.append(permuted_test)

    return nc_scenario(
        list_train_dataset,
        list_test_dataset,
        n_steps=len(list_train_dataset),
        task_labels=True,
        shuffle=False,
        class_ids_from_zero_in_each_step=True,
        one_dataset_per_step=True)


def RotatedMNIST(
        incremental_steps: int,
        seed: Optional[int] = None,
        rotations_list: Optional[Sequence[int]] = None,
        train_transform=_default_mnist_train_transform,
        test_transform=_default_mnist_test_transform) -> NCScenario:
    """
    This helper create a rotated MNIST scenario: where a given number of random
    rotations are used to rotate the MNIST images in
    ``incremental_steps`` different manners, creating an equal number of tasks.
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

        rotation = RandomRotation(degrees=(rotation_angle, rotation_angle))

        mnist_train, mnist_test = _get_mnist_dataset(rotation, rotation)

        # Freeze the rotation, then add the user defined transformations
        rotated_train = mnist_train \
            .freeze_transforms() \
            .add_transforms(train_transform)

        rotated_test = mnist_test \
            .freeze_transforms() \
            .add_transforms(test_transform)

        list_train_dataset.append(rotated_train)
        list_test_dataset.append(rotated_test)

    return nc_scenario(
        list_train_dataset,
        list_test_dataset,
        n_steps=len(list_train_dataset),
        task_labels=True,
        shuffle=False,
        class_ids_from_zero_in_each_step=True,
        one_dataset_per_step=True)


def _get_mnist_dataset(train_transformation, test_transformation):
    train_set = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                      train=True, download=True)

    test_set = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                     train=False, download=True)

    return train_test_transformation_datasets(
        train_set, test_set, train_transformation, test_transformation)


__all__ = ['SplitMNIST', 'PermutedMNIST', 'RotatedMNIST']


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    scenario = PermutedMNIST(5, train_transform=None, test_transform=None)
    for i, (img, label) in enumerate(scenario.train_stream[0].dataset):
        plt.imshow(np.asarray(img))
        plt.show()
        if ((i+1) % 5) == 0:
            break
