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
from torchvision.datasets import CIFAR100
from torchvision import transforms
from avalanche.benchmarks.scenarios.new_classes.scenario_creation import \
    create_nc_single_dataset_sit_scenario, \
    create_nc_single_dataset_multi_task_scenario
from avalanche.benchmarks.scenarios.new_classes.nc_scenario \
    import NCMultiTaskScenario, NCSingleTaskScenario


def create_multi_task_cifar100(incremental_tasks: int,
                               first_task_with_half_classes: bool = False,
                               seed: Optional[int] = None,
                               fixed_class_order: Optional[Sequence[int]] = None
                               ) -> NCMultiTaskScenario:
    """
    Creates a "New Classes - Multi Task" scenario using the CIFAR100 dataset.
    If the dataset is not present in the computer the method automatically
    download it and store the data in the data folder.

    :param incremental_tasks: The number of incremental tasks in the current
        scenario. If the first task is a "pretrain" task and it contains
        half of the classes, the number of incremental tasks is the number of
        tasks performed after the pretraining task.
        The value of this parameter should be a divisor of 100 if
        first_task_with_half_classes if false, a divisor of 50 otherwise.
    :param first_task_with_half_classes: A boolean value that indicates if a
        first pretraining task containing half of the classes should be used.
        If it's True, a pretrain task with half of the classes (50 for
        cifar100) is used, and a number of incremental tasks, given by the
        parameter incremental_task is constructed. If this paramenter is False
        no pretraining task will be used, and the dataset is simply split into
        a the number of tasks defined by the parameter incremental_tasks.
        Default to False.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.

    :returns: A :class:`NCMultiTaskScenario` instance initialized for the the
        MT scenario using CIFAR100.
    """
    cifar_train, cifar_test = __download_cifar100()
    total_tasks = incremental_tasks + 1 if first_task_with_half_classes \
        else incremental_tasks
    return create_nc_single_dataset_multi_task_scenario(
        train_dataset=cifar_train,
        test_dataset=cifar_test,
        n_tasks=total_tasks,
        seed=seed,
        fixed_class_order=fixed_class_order,
        per_task_classes={0: 50} if first_task_with_half_classes else None
    )


def create_single_task_cifar100(incremental_batches: int,
                                first_batch_with_half_classes: bool = False,
                                seed: Optional[int] = None,
                                fixed_class_order: Optional[Sequence[int]] =
                                None) -> NCSingleTaskScenario:
    """
    Creates a "New Classes - Single Incremental Task" scenario using the
    CIFAR100 dataset.
    If the dataset is not present in the computer the method automatically
    download it and store the data in the data folder.

    :param incremental_batches: The number of incremental batches in the current
        scenario. If the first batch is a "pretrain" batch and it contains
        half of the classes, the number of incremental batches is the number of
        batches performed after the pretraining task.
        The value of this parameter should be a divisor of 100 if
        first_batch_with_half_classes if false, a divisor of 50 otherwise.
    :param first_batch_with_half_classes: A boolean value that indicates if a
        first pretraining batch containing half of the classes should be used.
        If it's True, a pretrain batch with half of the classes (50 for
        cifar100) is used, and a number of incremental batches, given by the
        parameter incremental_batches is constructed. If this paramenter is
        False no pretraining batch will be used, and the dataset is simply
        split into a the number of batches defined by the parameter
        incremental_batchs.
        Default to False.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.

    :returns: A :class:`NCSingleTaskScenario` instance initialized for the the
        SIT scenario using CIFAR100.
    """
    cifar_train, cifar_test = __download_cifar100()
    total_tasks = incremental_batches + 1 if first_batch_with_half_classes \
        else incremental_batches
    return create_nc_single_dataset_sit_scenario(
        train_dataset=cifar_train,
        test_dataset=cifar_test,
        n_batches=total_tasks,
        seed=seed,
        fixed_class_order=fixed_class_order,
        per_batch_classes={0: 50} if first_batch_with_half_classes else None
    )


def __download_cifar100():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    cifar_train = CIFAR100('./data/cifar10', train=True,
                           download=True, transform=train_transform)
    cifar_test = CIFAR100('./data/cifar10', train=False,
                          download=True, transform=test_transform)
    return cifar_train, cifar_test
