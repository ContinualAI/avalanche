################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 13-02-2021                                                             #
# Author(s): Jary Pomponi                                                      #
################################################################################

from typing import Optional, Sequence, Any, Union
from os.path import expanduser
import torch
from torch import Tensor
from torchvision.transforms import ToTensor, Compose, Normalize, \
    ToPILImage, RandomRotation
from PIL.Image import Image

from avalanche.benchmarks import nc_scenario, NCScenario
from avalanche.benchmarks.datasets.omniglot import Omniglot
from avalanche.benchmarks.utils import train_test_transformation_datasets
import numpy as np

_default_omniglot_train_transform = Compose([
    ToTensor(),
    Normalize((0.9221,), (0.2681,))
])

_default_omniglot_test_transform = Compose([
    ToTensor(),
    Normalize((0.9221,), (0.2681,))
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


def SplitOmniglot(
        n_experiences: int,
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        train_transform=_default_omniglot_train_transform,
        test_transform=_default_omniglot_test_transform):
    """
    Creates a CL scenario using the OMNIGLOT dataset.
    This helper create the basic split OMNIGLOT scenario,
    where the 1623 classes of the OMNIGLOT dataset are evenly splitted into the
    given nuber of tasks.
    If the dataset is not present in the computer the method automatically
    download it and store the data in the data folder.

    :param n_experiences: The number of incremental experiences in the current
        scenario. The value of this parameter should be a divisor of 10.
    :param return_task_id: if True, for every experience the task id is returned
        and the Scenario is Multi Task. This means that the scenario returned
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
        MT split OMNIGLOT scenario if the parameter ``return_task_id`` is True,
        a :class:`NCSingleTaskScenario` initialized for the SIT split OMNIGLOT
        scenario otherwise.
    """

    omniglot_train, omniglot_test = _get_omniglot_dataset(train_transform,
                                                          test_transform)

    if return_task_id:
        return nc_scenario(
            train_dataset=omniglot_train,
            test_dataset=omniglot_test,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            class_ids_from_zero_in_each_exp=True)
    else:
        return nc_scenario(
            train_dataset=omniglot_train,
            test_dataset=omniglot_test,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order)


def PermutedOmniglot(
        n_experiences: int,
        seed: Optional[int] = None,
        train_transform: Any = _default_omniglot_train_transform,
        test_transform: Any = _default_omniglot_test_transform) -> NCScenario:
    """
    This helper create a permuted OMNIGLOT scenario: where a given number of
    random pixel permutations is used to permute the OMNIGLOT images in
    ``n_experiences`` different manners, creating an equal number of tasks.
    Each task is composed of all the original OMNIGLOT classes, but the pixel
    in the images are permuted in different ways in every task.
    If the dataset is not present in the computer the method automatically
    download it and store the data in the data folder.

    :param n_experiences: The number of experiences (tasks) in the current
        scenario. It indicates how many different permutations of the MNIST
        dataset have to be created.
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
        MT permuted OMNIGLOT scenario.
    """

    list_train_dataset = []
    list_test_dataset = []
    rng_permute = np.random.RandomState(seed)

    # for every incremental experience
    for _ in range(n_experiences):
        # choose a random permutation of the pixels in the image
        idx_permute = torch.from_numpy(rng_permute.permutation(784)).type(
            torch.int64)

        permutation = PixelsPermutation(idx_permute)

        omniglot_train, omniglot_test = _get_omniglot_dataset(permutation,
                                                              permutation)

        # Freeze the permutation, then add the user defined transformations
        permuted_train = omniglot_train \
            .freeze_transforms() \
            .add_transforms(train_transform)

        permuted_test = omniglot_test \
            .freeze_transforms() \
            .add_transforms(test_transform)

        list_train_dataset.append(permuted_train)
        list_test_dataset.append(permuted_test)

    return nc_scenario(
        list_train_dataset,
        list_test_dataset,
        n_experiences=len(list_train_dataset),
        task_labels=True,
        shuffle=False,
        class_ids_from_zero_in_each_exp=True,
        one_dataset_per_exp=True)


def RotatedOmniglot(
        n_experiences: int,
        seed: Optional[int] = None,
        rotations_list: Optional[Sequence[int]] = None,
        train_transform=_default_omniglot_train_transform,
        test_transform=_default_omniglot_test_transform) -> NCScenario:
    """
    This helper create a rotated OMNIGLOT scenario: where a given number of
    random rotations are used to rotate the OMNIGLOT images in
    ``n_experiences`` different manners, creating an equal number of tasks.
    Each task is composed of all the original OMNIGLOT classes, but the images
    are rotated in different ways and using different values in every task.
    If the dataset is not present in the computer the method automatically
    download it and store the data in the data folder.

    :param n_experiences: The number of experiences (tasks) in the current
        scenario. It indicates how many different rotations of the OMNIGLOT
        dataset have to be created.
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
        MT rotated OMNIGLOT scenario.
    """

    if rotations_list is None:
        rng_rotate = np.random.RandomState(seed)
        rotations_list = [rng_rotate.randint(-180, 181) for _ in range(
            n_experiences)]
    else:
        assert len(rotations_list) == n_experiences, "The number of rotations" \
                                               " should match the number" \
                                               " of incremental experiences."
    assert all(-180 <= rotations_list[i] <= 180
               for i in range(len(rotations_list))), "The value of a rotation" \
                                                     " should be between -180" \
                                                     " and 180 degrees."

    list_train_dataset = []
    list_test_dataset = []

    # for every incremental experience
    for experience in range(n_experiences):
        rotation_angle = rotations_list[experience]

        rotation = RandomRotation(degrees=(rotation_angle, rotation_angle))

        omniglot_train, omniglot_test = _get_omniglot_dataset(rotation,
                                                              rotation)

        # Freeze the rotation, then add the user defined transformations
        rotated_train = omniglot_train \
            .freeze_transforms() \
            .add_transforms(train_transform)

        rotated_test = omniglot_test \
            .freeze_transforms() \
            .add_transforms(test_transform)

        list_train_dataset.append(rotated_train)
        list_test_dataset.append(rotated_test)

    return nc_scenario(
        list_train_dataset,
        list_test_dataset,
        n_experiences=len(list_train_dataset),
        task_labels=True,
        shuffle=False,
        class_ids_from_zero_in_each_exp=True,
        one_dataset_per_exp=True)


def _get_omniglot_dataset(train_transformation, test_transform):
    train = Omniglot(root=expanduser("~") + "/.avalanche/data/omniglot/",
                     train=True, download=True)
    test = Omniglot(root=expanduser("~") + "/.avalanche/data/omniglot/",
                    train=False, download=True)

    return train_test_transformation_datasets(
                                    train_dataset=train,
                                    test_dataset=test,
                                    train_transformation=train_transformation,
                                    test_transformation=test_transform)


__all__ = [
    'SplitOmniglot',
    'PermutedOmniglot',
    'RotatedOmniglot'
]

if __name__ == '__main__':
    _get_omniglot_dataset(_default_omniglot_train_transform,
                          _default_omniglot_test_transform)
    rot = RotatedOmniglot(n_experiences=10, seed=1)
