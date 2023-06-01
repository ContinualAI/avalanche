################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 13-02-2021                                                             #
# Author(s): Jary Pomponi, Antonio Carta                                       #
################################################################################
from pathlib import Path
from typing import Optional, Sequence, Any, Union
from torch import Tensor
from torchvision.transforms import (
    ToTensor,
    Compose,
    Normalize,
    ToPILImage,
)
from PIL.Image import Image

from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.datasets.omniglot import Omniglot


_default_omniglot_train_transform = Compose(
    [ToTensor(), Normalize((0.9221,), (0.2681,))]
)

_default_omniglot_eval_transform = Compose(
    [ToTensor(), Normalize((0.9221,), (0.2681,))]
)


class PixelsPermutation(object):
    """Apply a fixed permutation to the pixels of the given image.

    Works with both Tensors and PIL images. Returns an object of the same type
    of the input element.
    """

    def __init__(self, index_permutation: Sequence[int]):
        self.permutation = index_permutation
        self._to_tensor = ToTensor()
        self._to_image = ToPILImage()

    def __call__(self, img: Union[Image, Tensor]):
        is_image = isinstance(img, Image)
        if (not is_image) and (not isinstance(img, Tensor)):
            raise ValueError("Invalid input: must be a PIL image or a Tensor")

        image_as_tensor: Tensor
        if is_image:
            image_as_tensor = self._to_tensor(img)
        else:
            image_as_tensor = img

        image_as_tensor = image_as_tensor.view(-1)[self.permutation].view(
            *image_as_tensor.shape
        )

        if is_image:
            img = self._to_image(image_as_tensor)
        else:
            img = image_as_tensor

        return img


def SplitAlphabetOmniglot(
    n_experiences: int,
    *,
    return_task_id=False,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    class_ids_from_zero_from_first_exp: bool = False,
    shuffle: bool = True,
    train_transform: Optional[Any] = _default_omniglot_train_transform,
    eval_transform: Optional[Any] = _default_omniglot_eval_transform,
    dataset_root: Optional[Union[str, Path]] = None
):
    """Class-incremental OMNIGLOT with the alphabet used as target.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned benchmark will return experiences containing all patterns of a
    subset of alphabets (class-incremental scenario).

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of incremental experiences in the current
        benchmark. The value of this parameter should be a divisor of 10.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param class_ids_from_zero_from_first_exp: If True, original class IDs
        will be remapped so that they will appear as having an ascending
        order. For instance, if the resulting class order after shuffling
        (or defined by fixed_class_order) is [23, 34, 11, 7, 6, ...] and
        class_ids_from_zero_from_first_exp is True, then all the patterns
        belonging to class 23 will appear as belonging to class "0",
        class "34" will be mapped to "1", class "11" to "2" and so on.
        This is very useful when drawing confusion matrices and when dealing
        with algorithms with dynamic head expansion. Defaults to False.
        Mutually exclusive with the ``class_ids_from_zero_in_each_exp``
        parameter.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to True.
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
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'omniglot' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    omniglot_train, omniglot_test = _get_omniglot_dataset(dataset_root)
    return nc_benchmark(
        train_dataset=omniglot_train,
        test_dataset=omniglot_test,
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        fixed_class_order=fixed_class_order,
        shuffle=shuffle,
        class_ids_from_zero_in_each_exp=False,
        class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


def SplitOmniglot(
    n_experiences: int,
    *,
    return_task_id=False,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    shuffle: bool = True,
    class_ids_from_zero_in_each_exp: bool = False,
    class_ids_from_zero_from_first_exp: bool = False,
    train_transform: Optional[Any] = _default_omniglot_train_transform,
    eval_transform: Optional[Any] = _default_omniglot_eval_transform,
    dataset_root: Optional[Union[str, Path]] = None
):
    """
    Creates a CL benchmark using the OMNIGLOT dataset.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc.

    By default, an equal amount of classes will be assigned to each experience.
    OMNIGLOT consists of 964 classes, which means that the number of
    experiences can be 1, 2, 4, 241, 482, 964.

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

    :param n_experiences: The number of incremental experiences in the current
        benchmark. The value of this parameter should be a divisor of 10.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to True.
    :param class_ids_from_zero_in_each_exp: If True, original class IDs
        will be mapped to range [0, n_classes_in_exp) for each experience.
        Defaults to False. Mutually exclusive with the
        ``class_ids_from_zero_from_first_exp`` parameter.
    :param class_ids_from_zero_from_first_exp: If True, original class IDs
        will be remapped so that they will appear as having an ascending
        order. For instance, if the resulting class order after shuffling
        (or defined by fixed_class_order) is [23, 34, 11, 7, 6, ...] and
        class_ids_from_zero_from_first_exp is True, then all the patterns
        belonging to class 23 will appear as belonging to class "0",
        class "34" will be mapped to "1", class "11" to "2" and so on.
        This is very useful when drawing confusion matrices and when dealing
        with algorithms with dynamic head expansion. Defaults to False.
        Mutually exclusive with the ``class_ids_from_zero_in_each_exp``
        parameter.
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
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'omniglot' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    omniglot_train, omniglot_test = _get_omniglot_dataset(dataset_root)
    return nc_benchmark(
        train_dataset=omniglot_train,
        test_dataset=omniglot_test,
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        fixed_class_order=fixed_class_order,
        shuffle=shuffle,
        class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
        class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


def _get_omniglot_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location("omniglot")

    train = Omniglot(root=dataset_root, train=True, download=True)
    test = Omniglot(root=dataset_root, train=False, download=True)

    return train, test


__all__ = ["SplitOmniglot"]

if __name__ == "__main__":
    import sys

    print("Split Omniglot")
    benchmark_instance = SplitOmniglot(4, train_transform=None, eval_transform=None)
    check_vision_benchmark(benchmark_instance)

    sys.exit(0)
