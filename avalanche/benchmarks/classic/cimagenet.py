#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-05-2020                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

import sys

from avalanche.benchmarks.datasets import ImageNet
from avalanche.benchmarks.generators import NCScenario

from torchvision import transforms

_default_train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
])

_default_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])


def SplitImageNet(root,
                  incremental_steps=10,
                  classes_first_batch=None,
                  return_task_id=False,
                  seed=0,
                  fixed_class_order=None,
                  train_transform=_default_train_transform,
                  test_transform=_default_test_transform):
    """
    Creates a CL scenario using the Tiny ImageNet dataset.
    If the dataset is not present in the computer the method automatically
    download it and store the data in the data folder.

    :param root: Base path where Imagenet data are stored.
    :param incremental_steps: The number of incremental steps in the current
        scenario.
    :param classes_first_batch: Number of classes in the first batch.
    Usually this is set to 500. Default to None.
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

    train_set = ImageNet(root, solit="train", transform=train_transform)
    test_set = ImageNet(root, split="val", transform=test_transform)

    if classes_first_batch is not None:
        per_step_classes = {0: classes_first_batch}
    else:
        per_step_classes = None

    if return_task_id:
        return NCScenario(
            train_dataset=train_set,
            test_dataset=test_set,
            n_steps=incremental_steps,
            per_step_classes=per_step_classes,
            seed=seed,
            fixed_class_order=fixed_class_order,
            multi_task=True
        )
    else:
        return NCScenario(
            train_dataset=train_set,
            test_dataset=test_set,
            n_steps=incremental_steps,
            per_step_classes=per_step_classes,
            seed=seed,
            fixed_class_order=fixed_class_order,
            multi_task=False
        )

if __name__ == "__main__":

    scenario = SplitImageNet("/home/admin/data/imagenet/2012")
    for step in scenario:
        print("step: ", step.current_step)
        print("classes number: ", len(step.classes_in_this_task))
        print("classes: ", step.classes_in_this_task)