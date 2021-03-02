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


from avalanche.benchmarks.datasets import ImageNet
from avalanche.benchmarks import nc_scenario

from torchvision import transforms

from avalanche.benchmarks.utils import train_eval_avalanche_datasets

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

_default_train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

_default_eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


def SplitImageNet(root,
                  n_experiences=10,
                  per_exp_classes=None,
                  return_task_id=False,
                  seed=0,
                  fixed_class_order=None,
                  train_transform=_default_train_transform,
                  eval_transform=_default_eval_transform):
    """
    Creates a CL scenario using the Tiny ImageNet dataset.

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

    :param root: Base path where Imagenet data is stored.
    :param n_experiences: The number of experiences in the current scenario.
    :param per_exp_classes: Is not None, a dictionary whose keys are
        (0-indexed) experience IDs and their values are the number of classes
        to include in the respective experiences. The dictionary doesn't
        have to contain a key for each experience! All the remaining exps
        will contain an equal amount of the remaining classes. The
        remaining number of classes must be divisible without remainder
        by the remaining number of experiences. For instance,
        if you want to include 50 classes in the first experience
        while equally distributing remaining classes across remaining
        experiences, just pass the "{0: 50}" dictionary as the
        per_experience_classes parameter. Defaults to None.
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

    train_set, test_set = _get_imagenet_dataset(
        root, train_transform, eval_transform)

    if return_task_id:
        return nc_scenario(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=True,
            per_exp_classes=per_exp_classes,
            seed=seed,
            fixed_class_order=fixed_class_order,
            class_ids_from_zero_in_each_exp=True)
    else:
        return nc_scenario(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=False,
            per_exp_classes=per_exp_classes,
            seed=seed,
            fixed_class_order=fixed_class_order)


def _get_imagenet_dataset(root, train_transformation, eval_transformation):
    train_set = ImageNet(root, split="train")

    test_set = ImageNet(root, split="val")

    return train_eval_avalanche_datasets(
        train_set, test_set, train_transformation, eval_transformation)


__all__ = [
    'SplitImageNet'
]

if __name__ == "__main__":
    scenario = SplitImageNet("/ssd2/datasets/imagenet/")
    for exp in scenario.train_stream:
        print("experience: ", exp.current_experience)
        print("classes number: ", len(exp.classes_in_this_experience))
        print("classes: ", exp.classes_in_this_experience)
