################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-05-2020                                                             #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

""" In this module the high-level benchmark generators are listed. They are
based on the methods already implemented in the "scenario" module. For the
specific generators we have: "New Classes" (NC) and "New Instances" (NI); For
the generic ones: filelist_scenario, tensor_scenario, dataset_scenario.
"""

from pathlib import Path
from typing import Sequence, Optional, Dict, SupportsInt, Union, Any, List, \
    Tuple

import torch

from avalanche.benchmarks.scenarios.generic_cl_scenario import GenericCLScenario
from avalanche.benchmarks.scenarios.generic_scenario_creation import *
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import \
    NCScenario
from avalanche.benchmarks.scenarios.new_instances.ni_scenario import NIScenario
from avalanche.benchmarks.utils import \
    concat_datasets_sequentially, as_transformation_dataset
from avalanche.benchmarks.utils.transform_dataset import SupportedDataset


def nc_scenario(
        train_dataset: Union[
            Sequence[SupportedDataset], SupportedDataset],
        test_dataset: Union[
            Sequence[SupportedDataset], SupportedDataset],
        n_steps: int,
        task_labels: bool,
        *,
        shuffle: bool = True,
        seed: Optional[int] = None,
        fixed_class_order: Sequence[int] = None,
        per_step_classes: Dict[int, int] = None,
        class_ids_from_zero_from_first_step: bool = False,
        class_ids_from_zero_in_each_step: bool = False,
        one_dataset_per_step: bool = False,
        reproducibility_data: Dict[str, Any] = None) -> NCScenario:

    """
    This method is the high-level specific scenario generator for the
    "New Classes" (NC) case. Given a sequence of train and test datasets creates
    the continual stream of data as a series of steps. Each step will contain
    all the patterns belonging to a certain set of classes and a class won't be
    assigned to more than one step.

    The ``task_labels`` parameter determines if each incremental step has
    an increasing task label or if, at the contrary, a default task label "0"
    has to be assigned to all steps. This can be useful when differentiating
    between Single-Incremental-Task and Multi-Task scenarios.

    There are other important parameters that can be specified in order to tweak
    the behaviour of the resulting scenario. Please take a few minutes to read
    and understand them as they may save you a lot of work.

    This generator features an integrated reproducibility mechanism that allows
    the user to store and later re-load a scenario. For more info see the
    ``reproducibility_data`` parameter.

    :param train_dataset: A list of training datasets, or a single dataset.
    :param test_dataset: A list of test datasets, or a single test dataset.
    :param n_steps: The number of incremental steps. This is not used when
        using multiple train/test datasets with the ``one_dataset_per_step``
        parameter set to True.
    :param task_labels: If True, each step will have an ascending task
            label. If False, the task label will be 0 for all the steps.
    :param shuffle: If True, the class (or step) order will be shuffled.
        Defaults to True.
    :param seed: If ``shuffle`` is True and seed is not None, the class (or
        step) order will be shuffled according to the seed. When None, the
        current PyTorch random number generator state will be used. Defaults to
        None.
    :param fixed_class_order: If not None, the class order to use (overrides
        the shuffle argument). Very useful for enhancing reproducibility.
        Defaults to None.
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
    :param class_ids_from_zero_from_first_step: If True, original class IDs
        will be remapped so that they will appear as having an ascending
        order. For instance, if the resulting class order after shuffling
        (or defined by fixed_class_order) is [23, 34, 11, 7, 6, ...] and
        class_ids_from_zero_from_first_step is True, then all the patterns
        belonging to class 23 will appear as belonging to class "0",
        class "34" will be mapped to "1", class "11" to "2" and so on.
        This is very useful when drawing confusion matrices and when dealing
        with algorithms with dynamic head expansion. Defaults to False.
        Mutually exclusive with the ``class_ids_from_zero_in_each_step``
        parameter.
    :param class_ids_from_zero_in_each_step: If True, original class IDs
        will be mapped to range [0, n_classes_in_step) for each step.
        Defaults to False. Mutually exclusive with the
        ``class_ids_from_zero_from_first_step`` parameter.
    :param one_dataset_per_step: available only when multiple train-test
        datasets are provided. If True, each dataset will be treated as a step.
        Mutually exclusive with the ``per_step_classes`` and
        ``fixed_class_order`` parameters. Overrides the ``n_steps`` parameter.
        Defaults to False.
    :param reproducibility_data: If not None, overrides all the other
        scenario definition options. This is usually a dictionary containing
        data used to reproduce a specific experiment. One can use the
        ``get_reproducibility_data`` method to get (and even distribute)
        the experiment setup so that it can be loaded by passing it as this
        parameter. In this way one can be sure that the same specific
        experimental setup is being used (for reproducibility purposes).
        Beware that, in order to reproduce an experiment, the same train and
        test datasets must be used. Defaults to None.

    :return: A :class:`NCMultiTaskScenario` or :class:`NCSingleTaskScenario`
        instance initialized for the the SIT or MT scenario.
    """

    if class_ids_from_zero_from_first_step and class_ids_from_zero_in_each_step:
        raise ValueError('Invalid mutually exclusive options '
                         'class_ids_from_zero_from_first_step and '
                         'classes_ids_from_zero_in_each_step set at the '
                         'same time')

    if isinstance(train_dataset, list) or isinstance(train_dataset, tuple):
        # Multi-dataset setting

        if len(train_dataset) != len(test_dataset):
            raise ValueError('Train/test dataset lists must contain the '
                             'exact same number of datasets')

        if per_step_classes and one_dataset_per_step:
            raise ValueError(
                'Both per_step_classes and one_dataset_per_step are'
                'used, but those options are mutually exclusive')

        if fixed_class_order and one_dataset_per_step:
            raise ValueError(
                'Both fixed_class_order and one_dataset_per_step are'
                'used, but those options are mutually exclusive')

        seq_train_dataset, seq_test_dataset, mapping = \
            concat_datasets_sequentially(train_dataset, test_dataset)

        if one_dataset_per_step:
            # If one_dataset_per_step is True, each dataset will be treated as
            # a step. In this scenario, shuffle refers to the step order,
            # not to the class one.
            fixed_class_order, per_step_classes = \
                _one_dataset_per_step_class_order(mapping, shuffle, seed)

            # We pass a fixed_class_order to the NCGenericScenario
            # constructor, so we don't need shuffling.
            shuffle = False
            seed = None

            # Overrides n_steps (and per_step_classes, already done)
            n_steps = len(train_dataset)
        train_dataset, test_dataset = seq_train_dataset, seq_test_dataset

    # Datasets should be instances of TransformationDataset
    train_dataset = as_transformation_dataset(train_dataset).train()
    test_dataset = as_transformation_dataset(test_dataset).eval()

    return NCScenario(train_dataset, test_dataset, n_steps, task_labels,
                      shuffle, seed, fixed_class_order, per_step_classes,
                      class_ids_from_zero_from_first_step,
                      class_ids_from_zero_in_each_step,
                      reproducibility_data)


def ni_scenario(
        train_dataset: Union[
            Sequence[SupportedDataset], SupportedDataset],
        test_dataset: Union[
            Sequence[SupportedDataset], SupportedDataset],
        n_steps: int,
        *,
        task_labels: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
        balance_steps: bool = False,
        min_class_patterns_in_step: int = 0,
        fixed_step_assignment: Optional[Sequence[Sequence[int]]] = None,
        reproducibility_data: Optional[Dict[str, Any]] = None) \
        -> NIScenario:
    """
    This method is the high-level specific scenario generator for the
    "New Instances" (NI) case. Given a sequence of train and test datasets
    creates the continual stream of data as a series of steps. Each step will
    contain patterns belonging to the same classes.

    The ``task_labels`` parameter determines if each incremental step has
    an increasing task label or if, at the contrary, a default task label "0"
    has to be assigned to all steps. This can be useful when differentiating
    between Single-Incremental-Task and Multi-Task scenarios.

    There are other important parameters that can be specified in order to tweak
    the behaviour of the resulting scenario. Please take a few minutes to read
    and understand them as they may save you a lot of work.

    This generator features an integrated reproducibility mechanism that allows
    the user to store and later re-load a scenario. For more info see the
    ``reproducibility_data`` parameter.

    :param train_dataset: A list of training datasets, or a single dataset.
    :param test_dataset: A list of test datasets, or a single test dataset.
    :param n_steps: The number of steps.
    :param task_labels: If True, each step will have an ascending task
            label. If False, the task label will be 0 for all the steps.
    :param shuffle: If True, patterns order will be shuffled.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param balance_steps: If True, pattern of each class will be equally
        spread across all steps. If False, patterns will be assigned to
        steps in a complete random way. Defaults to False.
    :param min_class_patterns_in_step: The minimum amount of patterns of
        every class that must be assigned to every step. Compatible with
        the ``balance_steps`` parameter. An exception will be raised if
        this constraint can't be satisfied. Defaults to 0.
    :param fixed_step_assignment: If not None, the pattern assignment
        to use. It must be a list with an entry for each step. Each entry
        is a list that contains the indexes of patterns belonging to that
        step. Overrides the ``shuffle``, ``balance_steps`` and
        ``min_class_patterns_in_step`` parameters.
    :param reproducibility_data: If not None, overrides all the other
        scenario definition options, including ``fixed_step_assignment``.
        This is usually a dictionary containing data used to
        reproduce a specific experiment. One can use the
        ``get_reproducibility_data`` method to get (and even distribute)
        the experiment setup so that it can be loaded by passing it as this
        parameter. In this way one can be sure that the same specific
        experimental setup is being used (for reproducibility purposes).
        Beware that, in order to reproduce an experiment, the same train and
        test datasets must be used. Defaults to None.

    :return: A :class:`NIScenario` instance.
    """

    seq_train_dataset, seq_test_dataset = train_dataset, test_dataset
    if isinstance(train_dataset, list) or isinstance(train_dataset, tuple):
        if len(train_dataset) != len(test_dataset):
            raise ValueError('Train/test dataset lists must contain the '
                             'exact same number of datasets')

        seq_train_dataset, seq_test_dataset, _ = \
            concat_datasets_sequentially(train_dataset, test_dataset)

    # Datasets should be instances of TransformationDataset
    seq_train_dataset = as_transformation_dataset(seq_train_dataset).train()
    seq_test_dataset = as_transformation_dataset(seq_test_dataset).eval()

    return NIScenario(
        seq_train_dataset, seq_test_dataset,
        n_steps,
        task_labels,
        shuffle=shuffle, seed=seed,
        balance_steps=balance_steps,
        min_class_patterns_in_step=min_class_patterns_in_step,
        fixed_step_assignment=fixed_step_assignment,
        reproducibility_data=reproducibility_data)


def dataset_scenario(
        train_dataset_list: Sequence[SupportedDataset],
        test_dataset_list: Sequence[SupportedDataset],
        task_labels: Sequence[int],
        complete_test_set_only: bool = False) -> GenericCLScenario:
    """
    Creates a generic scenario given a list of datasets and the respective task
    labels. Each training dataset will be considered as a separate training
    step. Contents of the datasets will not be changed, including the targets.

    When loading the datasets from a set of fixed file lists, consider using
    the :func:`filelist_scenario` helper method instead.

    In its base form, this function accepts a list of test datasets that must
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

    :returns: A :class:`GenericCLScenario` instance.
    """

    return create_multi_dataset_generic_scenario(
        train_dataset_list=train_dataset_list,
        test_dataset_list=test_dataset_list,
        task_labels=task_labels,
        complete_test_set_only=complete_test_set_only
    )


def filelist_scenario(
        root: Union[str, Path],
        train_file_lists: Sequence[Union[str, Path]],
        test_file_lists: Union[Union[str, Path], Sequence[Union[str, Path]]],
        task_labels: Sequence[int],
        complete_test_set_only: bool = False,
        train_transform=None, train_target_transform=None,
        test_transform=None, test_target_transform=None) -> GenericCLScenario:
    """
    Creates a generic scenario given a list of filelists and the respective task
    labels. A separate dataset will be created for each filelist and each of
    those training datasets will be considered a separate training step.
    Contents of the datasets will not be changed, including the targets.

    In its base form, this function accepts a list of filelists for the test
    datsets that must contain the same amount of elements of the training list.
    Those pairs of datasets are then used to create the "past", "cumulative"
    (a.k.a. growing) and "future" test sets. However, in certain Continual
    Learning scenarios only the concept of "complete" test set makes sense. In
    that case, the ``complete_test_set_only`` should be set to True (see the
    parameter description for more info).

    This helper functions is the best shot when loading Caffe-style dataset
    based on filelists.

    :param root: The root path of the dataset.
    :param train_file_lists: A list of filelists describing the
        paths of the training patterns for each step.
    :param test_file_lists: A list of filelists describing the
        paths of the test patterns for each step.
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
    :param train_transform: The transformation to apply to training patterns.
        Defaults to None.
    :param train_target_transform: The transformation to apply to training
        patterns targets. Defaults to None.
    :param test_transform: The transformation to apply to test patterns.
        Defaults to None.
    :param test_target_transform: The transformation to apply to test
        patterns targets. Defaults to None.

    :returns: A :class:`GenericCLScenario` instance.
    """

    return create_generic_scenario_from_filelists(
        root=root,
        train_file_lists=train_file_lists,
        test_file_lists=test_file_lists,
        task_labels=task_labels,
        complete_test_set_only=complete_test_set_only,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        test_transform=test_transform,
        test_target_transform=test_target_transform
    )


FileAndLabel = Tuple[Union[str, Path], int]


def paths_scenario(
        train_list_of_files: Sequence[Sequence[FileAndLabel]],
        test_list_of_files: Union[Sequence[FileAndLabel],
                                  Sequence[Sequence[FileAndLabel]]],
        task_labels: Sequence[int],
        complete_test_set_only: bool = False,
        train_transform=None, train_target_transform=None,
        test_transform=None, test_target_transform=None) -> GenericCLScenario:
    """
    Creates a generic scenario given a list of files and class labels.
    A separate dataset will be created for each list and each of
    those training datasets will be considered a separate training step.
    Contents of the datasets will not be changed, including the targets.

    This is very similar to `filelist_scenario`, with the main difference being
    that `filelist_scenario` accepts, for each step, a file list formatted in
    Caffe-style. On the contrary, this accepts a list of tuples where each tuple
    contains two elements: the full path to the pattern and its label.
    Optionally, the tuple may contain a third element describing the bounding
    box of the element to crop. This last bounding box may be useful when trying
    to extract the part of the image depicting the desired element.

    In its base form, this function accepts a list of lists of tuples for the
    test datsets that must contain the same amount of lists of the training
    list. Those pairs of datasets are then used to create the "past",
    "cumulative" (a.k.a. growing) and "future" test sets. However, in certain
    Continual Learning scenarios only the concept of "complete" test set makes
    sense. In that case, the ``complete_test_set_only`` should be set to True
    (see the parameter description for more info).

    :param train_list_of_files: A list of lists. Each list describes the paths
        and labels of patterns to include in that training step as tuples. Each
        tuple must contain two elements: the full path to the pattern and its
        class label. Optionally, the tuple may contain a third element
        describing the bounding box to use for cropping (top, left, height,
        width).
    :param test_list_of_files: A list of lists. Each list describes the paths
        and labels of patterns to include in that test step as tuples. Each
        tuple must contain two elements: the full path to the pattern and its
        class label. Optionally, the tuple may contain a third element
        describing the bounding box to use for cropping (top, left, height,
        width).
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
    :param train_transform: The transformation to apply to training patterns.
        Defaults to None.
    :param train_target_transform: The transformation to apply to training
        patterns targets. Defaults to None.
    :param test_transform: The transformation to apply to test patterns.
        Defaults to None.
    :param test_target_transform: The transformation to apply to test
        patterns targets. Defaults to None.

    :returns: A :class:`GenericCLScenario` instance.
    """

    return create_generic_scenario_from_paths(
        train_list_of_files=train_list_of_files,
        test_list_of_files=test_list_of_files,
        task_labels=task_labels,
        complete_test_set_only=complete_test_set_only,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        test_transform=test_transform,
        test_target_transform=test_target_transform
    )


def tensor_scenario(
        train_data_x: Sequence[Any],
        train_data_y: Sequence[Sequence[SupportsInt]],
        test_data_x: Union[Any, Sequence[Any]],
        test_data_y: Union[Any, Sequence[Sequence[SupportsInt]]],
        task_labels: Sequence[int],
        complete_test_set_only: bool = False,
        train_transform=None, train_target_transform=None,
        test_transform=None, test_target_transform=None) -> GenericCLScenario:
    """
    Creates a generic scenario given lists of Tensors and the respective task
    labels. A separate dataset will be created from each Tensor pair (x + y)
    and each of those training datasets will be considered a separate
    training step. Contents of the datasets will not be changed, including the
    targets. Using this helper function is the lower level way to create a
    Continual Learning scenario. When possible, consider using higher level
    helpers.

    In its base form, the test lists must contain the same amount of elements of
    the training lists. Those pairs of datasets are then used to create the
    "past", "cumulative" (a.k.a. growing) and "future" test sets.
    However, in certain Continual Learning scenarios only the concept of
    "complete" test set makes sense. In that case, the
    ``complete_test_set_only`` should be set to True (see the parameter
    description for more info).

    :param train_data_x: A list of Tensors (one per step) containing the
        patterns of the training sets.
    :param train_data_y: A list of Tensors or int lists containing the
        labels of the patterns of the training sets. Must contain the same
        number of elements of ``train_datasets_x``.
    :param test_data_x: A Tensor or a list of Tensors (one per step) containing
        the patterns of the test sets.
    :param test_data_y: A Tensor or a list of Tensors or int lists containing
        the labels of the patterns of the test sets. Must contain the same
        number of elements of ``test_datasets_x``.
    :param task_labels: A list of task labels. Must contain the same amount of
        elements of the ``train_datasets_x`` parameter. For
        Single-Incremental-Task (a.k.a. Task-Free) scenarios, this is usually
        a list of zeros. For Multi Task scenario, this is usually a list of
        ascending task labels (starting from 0).
    :param complete_test_set_only: If True, only the complete test set will
        be returned by the scenario. This means that the ``test_datasets_x`` and
        ``test_datasets_y`` parameters must be lists with a single element
        (the complete test set). Defaults to False, which means that
        ``train_file_lists`` and ``test_file_lists`` must contain the same
        amount of filelists paths.
    :param train_transform: The transformation to apply to training patterns.
        Defaults to None.
    :param train_target_transform: The transformation to apply to training
        patterns targets. Defaults to None.
    :param test_transform: The transformation to apply to test patterns.
        Defaults to None.
    :param test_target_transform: The transformation to apply to test
        patterns targets. Defaults to None.

    :returns: A :class:`GenericCLScenario` instance.
    """

    return create_generic_scenario_from_tensors(
        train_data_x=train_data_x,
        train_data_y=train_data_y,
        test_data_x=test_data_x,
        test_data_y=test_data_y,
        task_labels=task_labels,
        complete_test_set_only=complete_test_set_only,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        test_transform=test_transform,
        test_target_transform=test_target_transform
    )


def _one_dataset_per_step_class_order(
        class_list_per_step: Sequence[Sequence[int]],
        shuffle: bool, seed: Union[int, None]) -> (List[int], Dict[int, int]):
    """
    Utility function that shuffles the class order by keeping classes from the
    same step together. Each step is defined by a different entry in the
    class_list_per_step parameter.

    :param class_list_per_step: A list of class lists, one for each step
    :param shuffle: If True, the step order will be shuffled. If False,
        this function will return the concatenation of lists from the
        class_list_per_step parameter.
    :param seed: If not None, an integer used to initialize the random
        number generator.

    :returns: A class order that keeps class IDs from the same step together
        (adjacent).
    """
    dataset_order = list(range(len(class_list_per_step)))
    if shuffle:
        if seed is not None:
            torch.random.manual_seed(seed)
        dataset_order = torch.as_tensor(dataset_order)[
            torch.randperm(len(dataset_order))].tolist()
    fixed_class_order = []
    classes_per_step = {}
    for dataset_position, dataset_idx in enumerate(dataset_order):
        fixed_class_order.extend(class_list_per_step[dataset_idx])
        classes_per_step[dataset_position] = \
            len(class_list_per_step[dataset_idx])
    return fixed_class_order, classes_per_step


__all__ = [
    'nc_scenario',
    'ni_scenario',
    'dataset_scenario',
    'filelist_scenario',
    'paths_scenario',
    'tensor_scenario'
]
