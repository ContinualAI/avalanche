################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
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

from typing import Sequence, Optional
from os.path import expanduser
from torchvision.datasets import FashionMNIST
from torchvision import transforms

from avalanche.benchmarks import nc_benchmark

_default_fmnist_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

_default_fmnist_eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])


def SplitFMNIST(n_experiences: int,
                first_batch_with_half_classes: bool = False,
                return_task_id=False,
                seed: Optional[int] = None,
                fixed_class_order: Optional[Sequence[int]] = None,
                train_transform=_default_fmnist_train_transform,
                eval_transform=_default_fmnist_eval_transform):
    """
    Creates a CL scenario using the Fashion MNIST dataset.

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

    :param n_experiences: The number of experiences in the current
        scenario. If the first experience is a "pretraining" step and it
        contains half of the classes. The value of this parameter should be a
        divisor of 10 if first_task_with_half_classes if false, a divisor of 5
        otherwise.
    :param first_batch_with_half_classes: A boolean value that indicates if a
        first pretraining batch containing half of the classes should be used.
        If it's True, a pretraining batch with half of the classes (5 for
        cifar100) is used. If this parameter is False no pretraining task
        will be used, and the dataset is simply split into
        a the number of experiences defined by the parameter n_experiences.
        Default to False.
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

    fmnist_train, fmnist_test = _get_fmnist_dataset()

    if return_task_id:
        return nc_benchmark(
            train_dataset=fmnist_train,
            test_dataset=fmnist_test,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            per_exp_classes={0: 5} if first_batch_with_half_classes else None,
            train_transform=train_transform,
            eval_transform=eval_transform)
    else:
        return nc_benchmark(
            train_dataset=fmnist_train,
            test_dataset=fmnist_test,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            per_exp_classes={0: 5} if first_batch_with_half_classes else None,
            train_transform=train_transform,
            eval_transform=eval_transform)


def _get_fmnist_dataset():
    train_set = FashionMNIST(expanduser("~") + "/.avalanche/data/fashionmnist/",
                             train=True, download=True)
    test_set = FashionMNIST(expanduser("~") + "/.avalanche/data/fashionmnist/",
                            train=False, download=True)
    return train_set, test_set


__all__ = [
    'SplitFMNIST'
]

if __name__ == "__main__":

    nc_benchmark = SplitFMNIST(n_experiences=10)

    for i, batch in enumerate(nc_benchmark.train_stream):
        print(i, batch)
