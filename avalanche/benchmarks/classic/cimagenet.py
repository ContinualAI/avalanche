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

from avalanche.benchmarks.utils import train_test_transformation_datasets

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

_default_train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

_default_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


def SplitImageNet(root,
                  n_steps=10,
                  per_step_classes=None,
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
    :param n_steps: The number of  steps in the current scenario.
    :param per_step_classes: Is not None, a dictionary whose keys are
        (0-indexed) step IDs and their values are the number of classes
        to include in the respective steps. The dictionary doesn't
        have to contain a key for each step! All the remaining steps
        will contain an equal amount of the remaining classes. The
        remaining number of classes must be divisible without remainder
        by the remaining number of steps. For instance,
        if you want to include 50 classes in the first step
        while equally distributing remaining classes across remaining
        steps, just pass the "{0: 50}" dictionary as the
        per_step_classes parameter. Defaults to None.
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

    train_set, test_set = _get_imagenet_dataset(
        root, train_transform, test_transform)

    if return_task_id:
        return nc_scenario(
            train_dataset=train_set,
            test_dataset=test_set,
            n_steps=n_steps,
            task_labels=True,
            per_step_classes=per_step_classes,
            seed=seed,
            fixed_class_order=fixed_class_order,
            class_ids_from_zero_in_each_step=True)
    else:
        return nc_scenario(
            train_dataset=train_set,
            test_dataset=test_set,
            n_steps=n_steps,
            task_labels=False,
            per_step_classes=per_step_classes,
            seed=seed,
            fixed_class_order=fixed_class_order)


def _get_imagenet_dataset(root, train_transformation, test_transformation):
    train_set = ImageNet(root, split="train")

    test_set = ImageNet(root, split="val")

    return train_test_transformation_datasets(
        train_set, test_set, train_transformation, test_transformation)


__all__ = [
    'SplitImageNet'
]

if __name__ == "__main__":
    scenario = SplitImageNet("/ssd2/datasets/imagenet/")
    for step in scenario.train_stream:
        print("step: ", step.current_step)
        print("classes number: ", len(step.classes_in_this_step))
        print("classes: ", step.classes_in_this_step)
