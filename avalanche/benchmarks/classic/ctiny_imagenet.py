################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-05-2020                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

from torchvision import transforms
from avalanche.benchmarks.datasets import TinyImagenet
from avalanche.benchmarks.generators import nc_scenario
from avalanche.benchmarks.utils import train_test_transformation_datasets

_default_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
])

_default_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])


def SplitTinyImageNet(n_experiences=10, return_task_id=False, seed=0,
                      fixed_class_order=None,
                      train_transform=_default_train_transform,
                      test_transform=_default_test_transform):
    """
    Creates a CL scenario using the Tiny ImageNet dataset.
    If the dataset is not present in the computer the method automatically
    download it and store the data in the data folder.

    :param n_experiences: The number of experiences in the current scenario.
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
        MT scenario using CIFAR10 if the parameter ``return_task_id`` is True,
        a :class:`NCSingleTaskScenario` initialized for the SIT scenario using
        CIFAR10 otherwise.
        """

    train_set, test_set = _get_tiny_imagenet_dataset(
        train_transform, test_transform)

    if return_task_id:
        return nc_scenario(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            class_ids_from_zero_in_each_exp=True)
    else:
        return nc_scenario(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order)


def _get_tiny_imagenet_dataset(train_transformation, test_transformation):
    train_set = TinyImagenet(train=True)

    test_set = TinyImagenet(train=False)

    return train_test_transformation_datasets(
        train_set, test_set, train_transformation, test_transformation)


__all__ = [
    'SplitTinyImageNet'
]
