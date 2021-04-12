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

from avalanche.benchmarks.datasets import CUB200
from avalanche.benchmarks import nc_scenario

from torchvision import transforms

from avalanche.benchmarks.utils import train_eval_avalanche_datasets

_default_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
])

_default_eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])


def SplitCUB200(root,
                n_experiences=11,
                classes_first_batch=100,
                return_task_id=False,
                seed=0,
                fixed_class_order=None,
                shuffle=False,
                train_transform=_default_train_transform,
                eval_transform=_default_eval_transform):
    """
    Creates a CL scenario using the Cub-200 dataset.

    If the dataset is not present in the computer, **this method will NOT be
    able automatically download** and store it.

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

    :param root: Base path where Cub-200 data is stored.
    :param n_experiences: The number of experiences in the current scenario.
        Defaults to 11.
    :param classes_first_batch: Number of classes in the first batch.
        Usually this is set to 500. Defaults to 100.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to false.
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

    train_set, test_set = _get_cub200_dataset(
        root, train_transform, eval_transform)

    if classes_first_batch is not None:
        per_exp_classes = {0: classes_first_batch}
    else:
        per_exp_classes = None

    if return_task_id:
        return nc_scenario(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=True,
            per_exp_classes=per_exp_classes,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            one_dataset_per_exp=True)
    else:
        return nc_scenario(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=False,
            per_exp_classes=per_exp_classes,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle)


def _get_cub200_dataset(root, train_transformation, eval_transformation):
    train_set = CUB200(root, train=True)
    test_set = CUB200(root, train=False)

    return train_eval_avalanche_datasets(
        train_set, test_set, train_transformation, eval_transformation)


__all__ = [
    'SplitCUB200'
]

if __name__ == "__main__":
    scenario = SplitCUB200("~/.avalanche/data/CUB_200_2011/")
    for exp in scenario.train_stream:
        print("Experience: ", exp.current_experience)
        print("classes number: ", len(exp.classes_in_this_experience))
        print("classes: ", exp.classes_in_this_experience)
