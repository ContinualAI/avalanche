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

from avalanche.benchmarks.datasets import CUB200
from avalanche.benchmarks import nc_scenario

from torchvision import transforms

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


def SplitCUB200(root,
                n_steps=11,
                classes_first_batch=100,
                return_task_id=False,
                seed=0,
                fixed_class_order=None,
                shuffle=False,
                train_transform=_default_train_transform,
                test_transform=_default_test_transform):
    """
    Creates a CL scenario using the Tiny ImageNet dataset.
    If the dataset is not present in the computer the method automatically
    download it and store the data in the data folder.

    :param root: Base path where Imagenet data are stored.
    :param n_steps: The number of steps in the current scenario.
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
    :param shuffle: If true, the class order in the incremental steps is
        randomly shuffled. Default to false.
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

    train_set, test_set = _get_cub200_dataset(
        root, train_transform, test_transform)

    if classes_first_batch is not None:
        per_step_classes = {0: classes_first_batch}
    else:
        per_step_classes = None

    if return_task_id:
        return nc_scenario(
            train_dataset=train_set,
            test_dataset=test_set,
            n_steps=n_steps,
            task_labels=True,
            per_step_classes=per_step_classes,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            one_dataset_per_step=True)
    else:
        return nc_scenario(
            train_dataset=train_set,
            test_dataset=test_set,
            n_steps=n_steps,
            task_labels=False,
            per_step_classes=per_step_classes,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle)


def _get_cub200_dataset(root, train_transformation, test_transformation):
    train_set = CUB200(root, train=True)
    test_set = CUB200(root, train=False)

    return train_test_transformation_datasets(
        train_set, test_set, train_transformation, test_transformation)


__all__ = [
    'SplitCUB200'
]

if __name__ == "__main__":
    scenario = SplitCUB200("~/.avalanche/data/cub200/")
    for step in scenario.train_stream:
        print("step: ", step.current_step)
        print("classes number: ", len(step.classes_in_this_step))
        print("classes: ", step.classes_in_this_step)

