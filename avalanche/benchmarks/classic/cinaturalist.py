################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-05-2020                                                             #
# Author: Matthias De Lange                                                    #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################
from pathlib import Path
from typing import Union, Any, Optional

from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)
from avalanche.benchmarks.datasets import (
    INATURALIST2018,
    default_dataset_location,
)
from avalanche.benchmarks import nc_benchmark

from torchvision import transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

_default_train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)

_default_eval_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)


def SplitInaturalist(
    *,
    super_categories=None,
    return_task_id=False,
    download=False,
    seed=0,
    train_transform: Optional[Any] = _default_train_transform,
    eval_transform: Optional[Any] = _default_eval_transform,
    dataset_root: Union[str, Path] = None
):
    """
    Creates a CL benchmark using the iNaturalist2018 dataset.
    A selection of supercategories (by default 10) define the experiences.
    Note that the supercategories are highly imbalanced in the number of classes
    and the amount of data available.

    If the dataset is not present in the computer, **this method will
    automatically download** and store it if `download=True`
    (120Gtrain/val).

    To parse the dataset jsons you need to install an additional dependency:
    "pycocotools". You can install it like this:

        "conda install -c conda-forge pycocotools"

    Implementation is based on the CL survey
    (https://ieeexplore.ieee.org/document/9349197) but differs slightly.
    The survey uses only the original iNaturalist2018 training dataset split
    into 70/10/20 for train/val/test streams. This method instead uses the full
    iNaturalist2018 training set to make the `train_stream`, whereas the
    `test_stream` is defined by the original iNaturalist2018 validation data.

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param super_categories: The list of supercategories which define the
    tasks, i.e. each task consists of all classes in a super-category.
    :param download: If true and the dataset is not present in the computer,
    this method will automatically download and store it. This will take 120G
    for the train/val set.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
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
    :param dataset_root: The root path of the dataset.
        Defaults to None, which means that the default location for
        'inatuarlist2018' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    # Categories with > 100 datapoints
    if super_categories is None:
        super_categories = [
            "Amphibia",
            "Animalia",
            "Arachnida",
            "Aves",
            "Fungi",
            "Insecta",
            "Mammalia",
            "Mollusca",
            "Plantae",
            "Reptilia",
        ]

    train_set, test_set = _get_inaturalist_dataset(
        dataset_root, super_categories, download=download
    )
    per_exp_classes, fixed_class_order = _get_split(super_categories, train_set)

    if return_task_id:
        return nc_benchmark(
            fixed_class_order=fixed_class_order,
            per_exp_classes=per_exp_classes,
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=len(super_categories),
            task_labels=True,
            seed=seed,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
    else:
        return nc_benchmark(
            fixed_class_order=fixed_class_order,
            per_exp_classes=per_exp_classes,
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=len(super_categories),
            task_labels=False,
            seed=seed,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )


def _get_inaturalist_dataset(dataset_root, super_categories, download):
    if dataset_root is None:
        dataset_root = default_dataset_location("inatuarlist2018")

    train_set = INATURALIST2018(
        dataset_root, split="train", supcats=super_categories, download=download
    )
    test_set = INATURALIST2018(
        dataset_root, split="val", supcats=super_categories, download=download
    )

    return train_set, test_set


def _get_split(super_categories, train_set):
    """Get number of classes per experience, and
    the total order of the classes."""
    per_exp_classes, fixed_class_order = {}, []
    for idx, supcat in enumerate(super_categories):
        new_cats = list(train_set.cats_per_supcat[supcat])
        fixed_class_order += new_cats
        per_exp_classes[idx] = len(new_cats)
    return per_exp_classes, fixed_class_order


__all__ = ["SplitInaturalist"]


if __name__ == "__main__":
    import sys

    benchmark_instance = SplitInaturalist()
    check_vision_benchmark(benchmark_instance, show_without_transforms=False)
    sys.exit(0)
