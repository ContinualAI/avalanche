################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-05-2020                                                             #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

from typing import Sequence, Optional, Dict, SupportsInt, Union, Any
from pathlib import Path
from avalanche.benchmarks.scenarios.new_classes.scenario_creation import \
    create_nc_single_dataset_sit_scenario, \
    create_nc_single_dataset_multi_task_scenario, \
    create_nc_multi_dataset_sit_scenario, \
    create_nc_multi_dataset_multi_task_scenario
from avalanche.benchmarks.scenarios.new_instances.scenario_creation import \
    create_ni_multi_dataset_sit_scenario, \
    create_ni_single_dataset_sit_scenario
from avalanche.benchmarks.scenarios.generic_scenario_creation import *
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import \
    NCMultiTaskScenario, NCSingleTaskScenario
from avalanche.benchmarks.scenarios.new_instances.ni_scenario import NIScenario
from avalanche.benchmarks.scenarios.generic_cl_scenario import GenericCLScenario
from avalanche.training.utils import IDatasetWithTargets

""" In this module the high-level scenario generators are listed. They are 
based on the methods already implemented in the "scenario" module. For the 
specific generators we have:"New Classes" (NC) and "New Classes and Instances" (
NIC); For the  generic ones: FilelistScenario, TensorScenario, DatasetScenario.
."""


def NCScenario(
        train_dataset: Union[
            Sequence[IDatasetWithTargets], IDatasetWithTargets],
        test_dataset: Union[
            Sequence[IDatasetWithTargets], IDatasetWithTargets],
        n_steps: int,
        multi_task: bool =True,
        shuffle: bool = True,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        per_step_classes: Optional[Dict[int, int]] = None,
        classes_ids_from_zero: bool = True,
        remap_class_ids: bool = False,
        one_dataset_per_batch: bool = False,
        reproducibility_data: Optional[Dict[str, Any]] = None) \
        -> Union[NCMultiTaskScenario, NCSingleTaskScenario]:

    """
    This method is the high-level specific scenario generator for the
    "New Classes" (NC) case. Given a sequence of train and test datasets creates
    the continual stream of data as a series of steps (task or batches),
    highly tunable through its parameters.

    The main parameter ``multi_task`` determines if the scenario is a
    Single-Incremental-Task scenario o a Multi-task one. This in turn enable
    other important options specifying the behavious of each of those.

    :param train_dataset: A list of training datasets, or a single dataset.
    :param test_dataset: A list of test datasets, or a single test dataset.
    :param n_steps: The number of batches or tasks. This is not used in the
        case of multiple train/test datasets and when ``multi_task`` is set to
        True.
    :param multi_task: True if the scenario is Multi-Task, False if it is a
        Single-Incremental-Task scenario.
    :param shuffle: If True, class order will be shuffled.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: This parameter is valid only if a single
        train-test dataset is provided. A list of class IDs used to define the
        classorder. If None, values of shuffle and seed will be used to define
        the class order. If non-None, shuffle and seed parameters will be
        ignored. Defaults to None.
    :param per_step_classes: not available with multiple train-test
        datasets``multi_task`` is set to True. Is not None, a dictionary
        whose keys are (0-indexed) task IDs and their values are the number
        of classes to include in the respective batches. The dictionary doesn't
        have to contain a key for each task! All the remaining batches
        will contain an equal amount of the remaining classes. The
        remaining number of classes must be divisible without remainder
        by the remaining number of steps. For instance,
        if you want to include 50 classes in the first task while equally
        distributing remaining classes across remaining batches,
        just pass the "{0: 50}" dictionary as the per_task_classes
        parameter. Defaults to None.
    :param classes_ids_from_zero: This parametes is valid
        only when ``multi_task`` is set to True. If True, original class IDs
        will be mapped to range [0, n_classes_in_task) for each step. If False,
        each class will keep its original ID as defined in the input
        datasets. Defaults to True.
    :param remap_class_ids: This parameter is only valid when a single
        train/test is given and ``multi_task`` is set to False.
        If True, original class IDs will be remapped so that they will appear
        as having an ascending order. For instance, if the resulting class
        order after shuffling (or defined by fixed_class_order) is [23, 34,
        11, 7, 6, ...] and remap_class_indexes is True, then all the patterns
        belonging to class 23 will appear as belonging to class "0",
        class "34" will be mapped to "1", class "11" to "2" and so on. This
        is very useful when drawing confusion matrices and when dealing with
        algorithms with dynamic head expansion. Defaults to False.
    :param one_dataset_per_batch: available only when multile train-test
        datasets are provided and ``multi_task`` is set to False. If True, each
        dataset will be treated as a batch. Mutually exclusive with the
        per_task_classes parameter. Overrides the n_batches parameter.
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

    if isinstance(train_dataset, list) or isinstance(train_dataset, tuple):
        # we are in multi-datasets setting
        if multi_task:
            scenario = create_nc_multi_dataset_multi_task_scenario(
                train_dataset_list=train_dataset,
                test_dataset_list=test_dataset,
                shuffle=shuffle,
                seed=seed,
                classes_ids_from_zero_in_each_task=classes_ids_from_zero,
                reproducibility_data=reproducibility_data
            )
        else:
            scenario = create_nc_multi_dataset_sit_scenario(
                train_dataset_list=train_dataset,
                test_dataset_list=test_dataset,
                n_batches=n_steps, shuffle=shuffle,
                seed=seed, per_batch_classes=per_step_classes,
                one_dataset_per_batch=one_dataset_per_batch,
                reproducibility_data=reproducibility_data
            )

    else:
        # we are working with a single input dataset
        if multi_task:
            scenario = create_nc_single_dataset_multi_task_scenario(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_tasks=n_steps,
                seed=seed,
                fixed_class_order=fixed_class_order,
                per_task_classes=per_step_classes,
                classes_ids_from_zero_in_each_task=classes_ids_from_zero,
                reproducibility_data=reproducibility_data
            )
        else:
            scenario = create_nc_single_dataset_sit_scenario(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_batches=n_steps, shuffle=shuffle,
                seed=seed, fixed_class_order=fixed_class_order,
                per_batch_classes=per_step_classes,
                remap_class_ids=remap_class_ids,
                reproducibility_data=reproducibility_data
            )

    return scenario


def NIScenario(
        train_dataset: Union[
            Sequence[IDatasetWithTargets], IDatasetWithTargets],
        test_dataset: Union[
            Sequence[IDatasetWithTargets], IDatasetWithTargets],
        n_batches: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
        balance_batches: bool = False,
        min_class_patterns_in_batch: int = 0,
        fixed_batch_assignment: Optional[Sequence[Sequence[int]]] = None,
        reproducibility_data: Optional[Dict[str, Any]] = None) \
        -> NIScenario:
    """
    This method is the high-level specific scenario generator for the
    "New Instances" (NI) case. Given a sequence of train and test datasets
    creates the continual stream of data as a series of steps (task or batches),
    highly tunable through its parameters.

    :param train_dataset: A list of training datasets, or a single dataset.
    :param test_dataset: A list of test datasets, or a single test dataset.
    :param n_batches: The number of batches.
    :param shuffle: If True, patterns order will be shuffled.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param balance_batches: If True, pattern of each class will be equally
            spread across all batches. If False, patterns will be assigned to
            batches in a complete random way. Defaults to False.
    :param min_class_patterns_in_batch: The minimum amount of patterns of
        every class that must be assigned to every batch. Compatible with
        the ``balance_batches`` parameter. An exception will be raised if
        this constraint can't be satisfied. Defaults to 0.
    :param fixed_batch_assignment: only available when a single train-test
        dataset is given. If not None, the pattern assignment
        to use. It must be a list with an entry for each batch. Each entry
        is a list that contains the indexes of patterns belonging to that
        batch. Overrides the ``shuffle``, ``balance_batches`` and
        ``min_class_patterns_in_batch`` parameters.
    :param reproducibility_data: If not None, overrides all the other
        scenario definition options, including ``fixed_batch_assignment``.
        This is usually a dictionary containing data used to
        reproduce a specific experiment. One can use the scenario's
        ``get_reproducibility_data`` method to get (and even distribute)
        the experiment setup so that it can be loaded by passing it as this
        parameter. In this way one can be sure that the same specific
        experimental setup is being used (for reproducibility purposes).
        Beware that, in order to reproduce an experiment, the same train and
        test datasets must be used. Defaults to None.

    :return: A :class:`NIScenario` instance.
    """

    if isinstance(train_dataset, list) or isinstance(train_dataset, tuple):
        scenario = create_ni_multi_dataset_sit_scenario(
            train_dataset_list=train_dataset,
            test_dataset_list=test_dataset,
            n_batches=n_batches,
            shuffle=shuffle,
            seed=seed,
            balance_batches=balance_batches,
            min_class_patterns_in_batch=min_class_patterns_in_batch,
            reproducibility_data=reproducibility_data
        )
    else:
        scenario = create_ni_single_dataset_sit_scenario(
            train_dataset_list=train_dataset,
            test_dataset_list=test_dataset,
            n_batches=n_batches,
            shuffle=shuffle,
            seed=seed,
            balance_batches=balance_batches,
            min_class_patterns_in_batch=min_class_patterns_in_batch,
            fixed_batch_assignment=fixed_batch_assignment,
            reproducibility_data=reproducibility_data
        )

    return scenario


def DatasetScenario(
        train_dataset_list: Sequence[IDatasetWithTargets],
        test_dataset_list: Sequence[IDatasetWithTargets],
        task_labels: Sequence[int],
        complete_test_set_only: bool = False) -> GenericCLScenario:
    """
    Creates a generic scenario given a list of datasets and the respective task
    labels. Each training dataset will be considered as a separate training
    step. Contents of the datasets will not be changed, including the targets.

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

    :returns: A :class:`GenericCLScenario` instance.
    """

    return create_multi_dataset_generic_scenario(
        train_dataset_list=train_dataset_list,
        test_dataset_list=test_dataset_list,
        task_labels=task_labels,
        complete_test_set_only=complete_test_set_only
    )


def FilelistDataset(
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


def TensorScenario(
        train_data_x: Sequence[Any],
        train_data_y: Sequence[Sequence[SupportsInt]],
        test_data_x: Sequence[Any],
        test_data_y: Sequence[Sequence[SupportsInt]],
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
    :param test_data_x: A list of Tensors (one per step) containing the
        patterns of the test sets.
    :param test_data_y: A list of Tensors or int lists containing the
        labels of the patterns of the test sets. Must contain the same
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


__all__ = ['NCScenario', 'NIScenario', 'DatasetScenario', 'FilelistDataset',
           'TensorScenario']
