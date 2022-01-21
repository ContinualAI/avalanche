################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 22-06-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

""" This module contains DEPRECATED mid-level benchmark generators.
Please use the ones found in generic_benchmark_creation.
"""

import warnings
from pathlib import Path
from typing import Sequence, Union, SupportsInt, Any, Tuple

from torch import Tensor

from avalanche.benchmarks.utils import (
    AvalancheTensorDataset,
    SupportedDataset,
    datasets_from_paths,
    AvalancheDataset,
)
from avalanche.benchmarks.utils import datasets_from_filelists
from .generic_cl_scenario import GenericCLScenario
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence
from ..utils.avalanche_dataset import AvalancheDatasetType


def create_multi_dataset_generic_scenario(
    train_dataset_list: Sequence[SupportedDataset],
    test_dataset_list: Sequence[SupportedDataset],
    task_labels: Sequence[int],
    complete_test_set_only: bool = False,
    train_transform=None,
    train_target_transform=None,
    eval_transform=None,
    eval_target_transform=None,
    dataset_type: AvalancheDatasetType = None,
) -> GenericCLScenario:
    """
    This helper function is DEPRECATED in favor of
    `create_multi_dataset_generic_benchmark`.

    Creates a generic scenario given a list of datasets and the respective task
    labels. Each training dataset will be considered as a separate training
    experience. Contents of the datasets will not be changed, including the
    targets.

    When loading the datasets from a set of fixed filelist, consider using
    the :func:`create_generic_scenario_from_filelists` helper method instead.

    In its base form, this function accepts a list of test datsets that must
    contain the same amount of datasets of the training list.
    Those pairs are then used to create the "past", "cumulative"
    (a.k.a. growing) and "future" test sets. However, in certain Continual
    Learning scenarios only the concept of "complete" test set makes sense. In
    that case, the ``complete_test_set_only`` should be set to True (see the
    parameter description for more info).

    Beware that pattern transformations must already be included in the
    datasets (when needed).

    :param train_dataset_list: A list of training datasets.
    :param test_dataset_list: A list of test datasets.
    :param task_labels: A list of task labels. Must contain the same amount of
        elements of the ``train_dataset_list`` parameter. For
        Single-Incremental-Task (a.k.a. Task-Free) scenarios, this is usually
        a list of zeros. For Multi Task scenario, this is usually a list of
        ascending task labels (starting from 0).
    :param complete_test_set_only: If True, only the complete test set will
        be returned by the scenario. This means that the ``test_dataset_list``
        parameter must be list with a single element (the complete test set).
        Defaults to False, which means that ``train_dataset_list`` and
        ``test_dataset_list`` must contain the same amount of datasets.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param train_target_transform: The transformation to apply to training
        patterns targets. Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_target_transform: The transformation to apply to test
        patterns targets. Defaults to None.
    :param dataset_type: The type of the dataset. Defaults to None, which
        means that the type will be obtained from the input datasets. If input
        datasets are not instances of :class:`AvalancheDataset`, the type
        UNDEFINED will be used.

    :returns: A :class:`GenericCLScenario` instance.
    """

    warnings.warn(
        "create_multi_dataset_generic_scenario is deprecated in favor"
        " of create_multi_dataset_generic_benchmark.",
        DeprecationWarning,
    )

    transform_groups = dict(
        train=(train_transform, train_target_transform),
        eval=(eval_transform, eval_target_transform),
    )

    if complete_test_set_only:
        if len(test_dataset_list) != 1:
            raise ValueError(
                "Test must contain 1 element when"
                "complete_test_set_only is True"
            )
    else:
        if len(test_dataset_list) != len(train_dataset_list):
            raise ValueError(
                "Train and test lists must define the same "
                " amount of experiences"
            )

    train_t_labels = []
    train_dataset_list = list(train_dataset_list)
    for dataset_idx in range(len(train_dataset_list)):
        dataset = train_dataset_list[dataset_idx]
        train_t_labels.append(task_labels[dataset_idx])
        train_dataset_list[dataset_idx] = AvalancheDataset(
            dataset,
            task_labels=ConstantSequence(
                task_labels[dataset_idx], len(dataset)
            ),
            transform_groups=transform_groups,
            initial_transform_group="train",
            dataset_type=dataset_type,
        )

    test_t_labels = []
    test_dataset_list = list(test_dataset_list)
    for dataset_idx in range(len(test_dataset_list)):
        dataset = test_dataset_list[dataset_idx]

        test_t_label = task_labels[dataset_idx]
        if complete_test_set_only:
            test_t_label = 0

        test_t_labels.append(test_t_label)

        test_dataset_list[dataset_idx] = AvalancheDataset(
            dataset,
            task_labels=ConstantSequence(test_t_label, len(dataset)),
            transform_groups=transform_groups,
            initial_transform_group="eval",
            dataset_type=dataset_type,
        )

    return GenericCLScenario(
        stream_definitions={
            "train": (train_dataset_list, train_t_labels),
            "test": (test_dataset_list, test_t_labels),
        },
        complete_test_set_only=complete_test_set_only,
    )


def create_generic_scenario_from_filelists(
    root: Union[str, Path],
    train_file_lists: Sequence[Union[str, Path]],
    test_file_lists: Union[Union[str, Path], Sequence[Union[str, Path]]],
    task_labels: Sequence[int],
    complete_test_set_only: bool = False,
    train_transform=None,
    train_target_transform=None,
    eval_transform=None,
    eval_target_transform=None,
) -> GenericCLScenario:
    """
    This helper function is DEPRECATED in favor of
    `create_generic_benchmark_from_filelists`.

    Creates a generic scenario given a list of filelists and the respective task
    labels. A separate dataset will be created for each filelist and each of
    those training datasets will be considered a separate training experience.

    In its base form, this function accepts a list of filelists for the test
    datsets that must contain the same amount of elements of the training list.
    Those pairs of datasets are then used to create the "past", "cumulative"
    (a.k.a. growing) and "future" test sets. However, in certain Continual
    Learning scenarios only the concept of "complete" test set makes sense. In
    that case, the ``complete_test_set_only`` should be set to True (see the
    parameter description for more info).

    This helper functions is the best shot when loading Caffe-style dataset
    based on filelists.

    The resulting benchmark instance and the intermediate datasets used to
    populate it will be of type CLASSIFICATION.

    :param root: The root path of the dataset.
    :param train_file_lists: A list of filelists describing the
        paths of the training patterns for each experience.
    :param test_file_lists: A list of filelists describing the
        paths of the test patterns for each experience.
    :param task_labels: A list of task labels. Must contain the same amount of
        elements of the ``train_file_lists`` parameter. For
        Single-Incremental-Task (a.k.a. Task-Free) scenarios, this is usually
        a list of zeros. For Multi Task scenario, this is usually a list of
        ascending task labels (starting from 0).
    :param complete_test_set_only: If True, only the complete test set will
        be returned by the scenario. This means that the ``test_file_lists``
        parameter must be list with a single element (the complete test set).
        Alternatively, can be a plain string or :class:`Path` object.
        Defaults to False, which means that ``train_file_lists`` and
        ``test_file_lists`` must contain the same amount of filelists paths.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param train_target_transform: The transformation to apply to training
        patterns targets. Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_target_transform: The transformation to apply to test
        patterns targets. Defaults to None.

    :returns: A :class:`GenericCLScenario` instance.
    """

    warnings.warn(
        "create_generic_scenario_from_filelists is deprecated in "
        "favor of create_generic_benchmark_from_filelists.",
        DeprecationWarning,
    )

    train_datasets, test_dataset = datasets_from_filelists(
        root,
        train_file_lists,
        test_file_lists,
        complete_test_set_only=complete_test_set_only,
    )

    return create_multi_dataset_generic_scenario(
        train_datasets,
        test_dataset,
        task_labels,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        eval_transform=eval_transform,
        eval_target_transform=eval_target_transform,
        complete_test_set_only=complete_test_set_only,
        dataset_type=AvalancheDatasetType.CLASSIFICATION,
    )


FileAndLabel = Tuple[Union[str, Path], int]


def create_generic_scenario_from_paths(
    train_list_of_files: Sequence[Sequence[FileAndLabel]],
    test_list_of_files: Union[
        Sequence[FileAndLabel], Sequence[Sequence[FileAndLabel]]
    ],
    task_labels: Sequence[int],
    complete_test_set_only: bool = False,
    train_transform=None,
    train_target_transform=None,
    eval_transform=None,
    eval_target_transform=None,
    dataset_type: AvalancheDatasetType = AvalancheDatasetType.UNDEFINED,
) -> GenericCLScenario:
    """
    This helper function is DEPRECATED in favor of
    `create_generic_benchmark_from_paths`.

    Creates a generic scenario given a sequence of lists of files. A separate
    dataset will be created for each list. Each of those training datasets
    will be considered a separate training experience.

    This is very similar to `create_generic_scenario_from_filelists`, with the
    main difference being that `create_generic_scenario_from_filelists`
    accepts, for each experience, a file list formatted in Caffe-style.
    On the contrary, this accepts a list of tuples where each tuple contains
    two elements: the full path to the pattern and its label.
    Optionally, the tuple may contain a third element describing the bounding
    box of the element to crop. This last bounding box may be useful when trying
    to extract the part of the image depicting the desired element.

    In its base form, this function accepts a list for the test datasets that
    must contain the same amount of elements of the training list.
    Those pairs of datasets are then used to create the "past", "cumulative"
    (a.k.a. growing) and "future" test sets. However, in certain Continual
    Learning scenarios only the concept of "complete" test set makes sense. In
    that case, the ``complete_test_set_only`` should be set to True (see the
    parameter description for more info).

    The label of each pattern doesn't have to be an int.

    :param train_list_of_files: A list of lists. Each list describes the paths
        and labels of patterns to include in that training experience, as
        tuples. Each tuple must contain two elements: the full path to the
        pattern and its class label. Optionally, the tuple may contain a
        third element describing the bounding box to use for cropping (top,
        left, height, width).
    :param test_list_of_files: A list of lists. Each list describes the paths
        and labels of patterns to include in that test experience, as tuples.
        Each tuple must contain two elements: the full path to the pattern
        and its class label. Optionally, the tuple may contain a third element
        describing the bounding box to use for cropping (top, left, height,
        width).
    :param task_labels: A list of task labels. Must contain the same amount of
        elements of the ``train_file_lists`` parameter. For
        Single-Incremental-Task (a.k.a. Task-Free) scenarios, this is usually
        a list of zeros. For Multi Task scenario, this is usually a list of
        ascending task labels (starting from 0).
    :param complete_test_set_only: If True, only the complete test set will
        be returned by the scenario. This means that the ``test_list_of_files``
        parameter must define a single experience (the complete test set).
        Defaults to False, which means that ``train_list_of_files`` and
        ``test_list_of_files`` must contain the same amount of paths.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param train_target_transform: The transformation to apply to training
        patterns targets. Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_target_transform: The transformation to apply to test
        patterns targets. Defaults to None.
    :param dataset_type: The type of the dataset. Defaults to UNDEFINED.

    :returns: A :class:`GenericCLScenario` instance.
    """

    warnings.warn(
        "create_generic_scenario_from_paths is deprecated in favor"
        " of create_generic_benchmark_from_paths.",
        DeprecationWarning,
    )

    train_datasets, test_dataset = datasets_from_paths(
        train_list_of_files,
        test_list_of_files,
        complete_test_set_only=complete_test_set_only,
    )

    return create_multi_dataset_generic_scenario(
        train_datasets,
        test_dataset,
        task_labels,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        eval_transform=eval_transform,
        eval_target_transform=eval_target_transform,
        complete_test_set_only=complete_test_set_only,
        dataset_type=dataset_type,
    )


def create_generic_scenario_from_tensor_lists(
    train_tensors: Sequence[Sequence[Any]],
    test_tensors: Sequence[Sequence[Any]],
    task_labels: Sequence[int],
    *,
    complete_test_set_only: bool = False,
    train_transform=None,
    train_target_transform=None,
    eval_transform=None,
    eval_target_transform=None,
    dataset_type: AvalancheDatasetType = None
) -> GenericCLScenario:
    """
    This helper function is DEPRECATED in favor of
    `create_generic_benchmark_from_tensor_lists`.

    Creates a generic scenario given lists of Tensors. A separate dataset will
    be created from each Tensor tuple (x, y, z, ...) and each of those training
    datasets will be considered a separate training experience. Using this
    helper function is the lowest-level way to create a Continual Learning
    scenario. When possible, consider using higher level helpers.

    Experiences are defined by passing lists of tensors as the `train_tensors`
    and `test_tensors` parameter. Those parameters must be lists containing
    sub-lists of tensors, one for each experience. Each tensor defines the value
    of a feature ("x", "y", "z", ...) for all patterns of that experience.

    By default the second tensor of each experience will be used to fill the
    `targets` value (label of each pattern).

    In its base form, the test lists must contain the same amount of elements of
    the training lists. Those pairs of datasets are then used to create the
    "past", "cumulative" (a.k.a. growing) and "future" test sets.
    However, in certain Continual Learning scenarios only the concept of
    "complete" test set makes sense. In that case, the
    ``complete_test_set_only`` should be set to True (see the parameter
    description for more info).

    :param train_tensors: A list of lists. The first list must contain the
        tensors for the first training experience (one tensor per feature), the
        second list must contain the tensors for the second training experience,
        and so on.
    :param test_tensors: A list of lists. The first list must contain the
        tensors for the first test experience (one tensor per feature), the
        second list must contain the tensors for the second test experience,
        and so on. When using `complete_test_set_only`, this parameter
        must be a list containing a single sub-list for the single test
        experience.
    :param task_labels: A list of task labels. Must contain a task label for
        each experience. For Single-Incremental-Task (a.k.a. Task-Free)
        scenarios, this is usually a list of zeros. For Multi Task scenario,
        this is usually a list of ascending task labels (starting from 0).
    :param complete_test_set_only: If True, only the complete test set will
        be returned by the scenario. This means that ``test_tensors`` must
        define a single experience. Defaults to False, which means that
        ``train_tensors`` and ``test_tensors`` must define the same
        amount of experiences.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param train_target_transform: The transformation to apply to training
        patterns targets. Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_target_transform: The transformation to apply to test
        patterns targets. Defaults to None.
    :param dataset_type: The type of the dataset. Defaults to None, which
        means that the type will be obtained from the input datasets. If input
        datasets are not instances of :class:`AvalancheDataset`, the type
        UNDEFINED will be used.

    :returns: A :class:`GenericCLScenario` instance.
    """

    warnings.warn(
        "create_generic_scenario_from_tensor_lists is deprecated in "
        "favor of create_generic_benchmark_from_tensor_lists.",
        DeprecationWarning,
    )

    train_datasets = [
        AvalancheTensorDataset(*exp_tensors, dataset_type=dataset_type)
        for exp_tensors in train_tensors
    ]

    test_datasets = [
        AvalancheTensorDataset(*exp_tensors, dataset_type=dataset_type)
        for exp_tensors in test_tensors
    ]

    return create_multi_dataset_generic_scenario(
        train_datasets,
        test_datasets,
        task_labels,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        eval_transform=eval_transform,
        eval_target_transform=eval_target_transform,
        complete_test_set_only=complete_test_set_only,
        dataset_type=dataset_type,
    )


def create_generic_scenario_from_tensors(
    train_data_x: Sequence[Any],
    train_data_y: Sequence[Sequence[SupportsInt]],
    test_data_x: Union[Any, Sequence[Any]],
    test_data_y: Union[Any, Sequence[Sequence[SupportsInt]]],
    task_labels: Sequence[int],
    complete_test_set_only: bool = False,
    train_transform=None,
    train_target_transform=None,
    eval_transform=None,
    eval_target_transform=None,
    dataset_type: AvalancheDatasetType = AvalancheDatasetType.UNDEFINED,
) -> GenericCLScenario:
    """
    This helper function is DEPRECATED in favor of
    `create_generic_benchmark_from_tensor_lists`.

    Please consider using :func:`create_generic_scenario_from_tensor_lists`
    instead. When switching to the new function, please keep in mind that the
    format of the parameters is completely different!

    Creates a generic scenario given lists of Tensors and the respective task
    labels. A separate dataset will be created from each Tensor pair (x + y)
    and each of those training datasets will be considered a separate
    training experience. Contents of the datasets will not be changed, including
    the targets. Using this helper function is the lower level way to create a
    Continual Learning scenario. When possible, consider using higher level
    helpers.

    By default the second tensor of each experience will be used to fill the
    `targets` value (label of each pattern).

    In its base form, the test lists must contain the same amount of elements of
    the training lists. Those pairs of datasets are then used to create the
    "past", "cumulative" (a.k.a. growing) and "future" test sets.
    However, in certain Continual Learning scenarios only the concept of
    "complete" test set makes sense. In that case, the
    ``complete_test_set_only`` should be set to True (see the parameter
    description for more info).

    :param train_data_x: A list of Tensors (one per experience) containing the
        patterns of the training sets.
    :param train_data_y: A list of Tensors or int lists containing the
        labels of the patterns of the training sets. Must contain the same
        number of elements of ``train_datasets_x``.
    :param test_data_x: A Tensor or a list of Tensors (one per experience)
        containing the patterns of the test sets.
    :param test_data_y: A Tensor or a list of Tensors or int lists containing
        the labels of the patterns of the test sets. Must contain the same
        number of elements of ``test_datasets_x``.
    :param task_labels: A list of task labels. Must contain the same amount of
        elements of the ``train_datasets_x`` parameter. For
        Single-Incremental-Task (a.k.a. Task-Free) scenarios, this is usually
        a list of zeros. For Multi Task scenario, this is usually a list of
        ascending task labels (starting from 0).
    :param complete_test_set_only: If True, only the complete test set will
        be returned by the scenario. This means that ``test_data_x`` and
        ``test_data_y`` must define a single experience. Defaults to False,
        which means that ``train_data_*`` and ``test_data_*`` must define the
        same amount of experiences.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param train_target_transform: The transformation to apply to training
        patterns targets. Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_target_transform: The transformation to apply to test
        patterns targets. Defaults to None.
    :param dataset_type: The type of the dataset. Defaults to UNDEFINED.

    :returns: A :class:`GenericCLScenario` instance.
    """

    warnings.warn(
        "create_generic_scenario_from_tensors is deprecated in favor "
        "of create_generic_benchmark_from_tensor_lists.",
        DeprecationWarning,
    )

    if len(train_data_x) != len(train_data_y):
        raise ValueError(
            "train_data_x and train_data_y must contain"
            " the same amount of elements"
        )

    if type(test_data_x) != type(test_data_y):
        raise ValueError(
            "test_data_x and test_data_y must be of" " the same type"
        )

    if isinstance(test_data_x, Tensor):
        test_data_x = [test_data_x]
        test_data_y = [test_data_y]
    else:
        if len(test_data_x) != len(test_data_y):
            raise ValueError(
                "test_data_x and test_data_y must contain"
                " the same amount of elements"
            )

    exp_train_first_structure = []
    exp_test_first_structure = []
    for exp_idx in range(len(train_data_x)):
        exp_x = train_data_x[exp_idx]
        exp_y = train_data_y[exp_idx]

        exp_train_first_structure.append([exp_x, exp_y])

    for exp_idx in range(len(test_data_x)):
        exp_x = test_data_x[exp_idx]
        exp_y = test_data_y[exp_idx]

        exp_test_first_structure.append([exp_x, exp_y])

    return create_generic_scenario_from_tensor_lists(
        train_tensors=exp_train_first_structure,
        test_tensors=exp_test_first_structure,
        task_labels=task_labels,
        complete_test_set_only=complete_test_set_only,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        eval_transform=eval_transform,
        eval_target_transform=eval_target_transform,
        dataset_type=dataset_type,
    )


__all__ = [
    "create_multi_dataset_generic_scenario",
    "create_generic_scenario_from_filelists",
    "create_generic_scenario_from_paths",
    "create_generic_scenario_from_tensor_lists",
    "create_generic_scenario_from_tensors",
]
