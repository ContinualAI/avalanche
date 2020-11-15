################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24-06-2020                                                             #
# Author(s): Gabriele Graffieti                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from typing import Sequence, Optional
from os.path import expanduser
from torchvision.datasets import CIFAR10
from torchvision import transforms

from avalanche.benchmarks import nc_scenario

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


def SplitCIFAR10(incremental_steps: int,
                 first_batch_with_half_classes: bool = False,
                 return_task_id=False,
                 seed: Optional[int] = None,
                 fixed_class_order: Optional[Sequence[int]] = None,
                 train_transform=_default_cifar10_train_transform,
                 test_transform=_default_cifar10_test_transform
                 ):
    """
    Creates a CL scenario using the CIFAR10 dataset.
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
    cifar_train, cifar_test = _get_cifar10_dataset(train_transform,
                                                   test_transform)
    total_steps = incremental_steps + 1 if first_batch_with_half_classes \
        else incremental_steps
    if return_task_id:
        return nc_scenario(
            train_dataset=cifar_train,
            test_dataset=cifar_test,
            n_steps=total_steps,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            per_step_classes={0: 5} if first_batch_with_half_classes else None,
            class_ids_from_zero_in_each_step=True)
    else:
        return nc_scenario(
            train_dataset=cifar_train,
            test_dataset=cifar_test,
            n_steps=total_steps,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            per_step_classes={0: 5} if first_batch_with_half_classes else None
        )


def _get_cifar10_dataset(train_transformation, test_transformation):
    cifar_train = CIFAR10(expanduser("~") + "/.avalanche/data/cifar10/",
                          train=True,
                          download=True, transform=train_transformation)
    cifar_test = CIFAR10(expanduser("~") + "/.avalanche/data/cifar10/",
                         train=False,
                         download=True, transform=test_transformation)
    return cifar_train, cifar_test
