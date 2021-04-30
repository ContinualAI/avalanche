################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 16-04-2021                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

""" In this module the high-level benchmark generators are listed. They are
based on the methods already implemented in the "scenario" module. For the
specific generators we have: "New Classes" (NC) and "New Instances" (NI); For
the generic ones: filelist_benchmark, tensors_benchmark, dataset_benchmark
and paths_benchmark.
"""
from typing import Sequence, Optional, Dict, Union, Any, List

import torch

from avalanche.benchmarks.scenarios.generic_benchmark_creation import *
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import \
    NCScenario
from avalanche.benchmarks.scenarios.new_instances.ni_scenario import NIScenario
from avalanche.benchmarks.utils import concat_datasets_sequentially
from avalanche.benchmarks.utils.avalanche_dataset import SupportedDataset, \
    AvalancheDataset, AvalancheDatasetType


def nc_benchmark(
        train_dataset: Union[
            Sequence[SupportedDataset], SupportedDataset],
        test_dataset: Union[
            Sequence[SupportedDataset], SupportedDataset],
        n_experiences: int,
        task_labels: bool,
        *,
        shuffle: bool = True,
        seed: Optional[int] = None,
        fixed_class_order: Sequence[int] = None,
        per_exp_classes: Dict[int, int] = None,
        class_ids_from_zero_from_first_exp: bool = False,
        class_ids_from_zero_in_each_exp: bool = False,
        one_dataset_per_exp: bool = False,
        train_transform=None,
        eval_transform=None,
        reproducibility_data: Dict[str, Any] = None) -> NCScenario:
    """
    This is the high-level benchmark instances generator for the
    "New Classes" (NC) case. Given a sequence of train and test datasets creates
    the continual stream of data as a series of experiences. Each experience
    will contain all the instances belonging to a certain set of classes and a
    class won't be assigned to more than one experience.

    This is the reference helper function for creating instances of Class- or
    Task-Incremental benchmarks.

    The ``task_labels`` parameter determines if each incremental experience has
    an increasing task label or if, at the contrary, a default task label "0"
    has to be assigned to all experiences. This can be useful when
    differentiating between Single-Incremental-Task and Multi-Task scenarios.

    There are other important parameters that can be specified in order to tweak
    the behaviour of the resulting scenario. Please take a few minutes to read
    and understand them as they may save you a lot of work.

    This generator features a integrated reproducibility mechanism that allows
    the user to store and later re-load a scenario. For more info see the
    ``reproducibility_data`` parameter.

    :param train_dataset: A list of training datasets, or a single dataset.
    :param test_dataset: A list of test datasets, or a single test dataset.
    :param n_experiences: The number of incremental experience. This is not used
        when using multiple train/test datasets with the ``one_dataset_per_exp``
        parameter set to True.
    :param task_labels: If True, each experience will have an ascending task
            label. If False, the task label will be 0 for all the experiences.
    :param shuffle: If True, the class (or experience) order will be shuffled.
        Defaults to True.
    :param seed: If ``shuffle`` is True and seed is not None, the class (or
        experience) order will be shuffled according to the seed. When None, the
        current PyTorch random number generator state will be used. Defaults to
        None.
    :param fixed_class_order: If not None, the class order to use (overrides
        the shuffle argument). Very useful for enhancing reproducibility.
        Defaults to None.
    :param per_exp_classes: Is not None, a dictionary whose keys are
        (0-indexed) experience IDs and their values are the number of classes
        to include in the respective experiences. The dictionary doesn't
        have to contain a key for each experience! All the remaining experiences
        will contain an equal amount of the remaining classes. The
        remaining number of classes must be divisible without remainder
        by the remaining number of experiences. For instance,
        if you want to include 50 classes in the first experience
        while equally distributing remaining classes across remaining
        experiences, just pass the "{0: 50}" dictionary as the
        per_experience_classes parameter. Defaults to None.
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
    :param class_ids_from_zero_in_each_exp: If True, original class IDs
        will be mapped to range [0, n_classes_in_exp) for each experience.
        Defaults to False. Mutually exclusive with the
        ``class_ids_from_zero_from_first_exp`` parameter.
    :param one_dataset_per_exp: available only when multiple train-test
        datasets are provided. If True, each dataset will be treated as a
        experience. Mutually exclusive with the ``per_experience_classes`` and
        ``fixed_class_order`` parameters. Overrides the ``n_experiences`` 
        parameter. Defaults to False.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param reproducibility_data: If not None, overrides all the other
        scenario definition options. This is usually a dictionary containing
        data used to reproduce a specific experiment. One can use the
        ``get_reproducibility_data`` method to get (and even distribute)
        the experiment setup so that it can be loaded by passing it as this
        parameter. In this way one can be sure that the same specific
        experimental setup is being used (for reproducibility purposes).
        Beware that, in order to reproduce an experiment, the same train and
        test datasets must be used. Defaults to None.

    :return: A properly initialized :class:`NCScenario` instance.
    """

    if class_ids_from_zero_from_first_exp and class_ids_from_zero_in_each_exp:
        raise ValueError('Invalid mutually exclusive options '
                         'class_ids_from_zero_from_first_exp and '
                         'classes_ids_from_zero_in_each_exp set at the '
                         'same time')

    if isinstance(train_dataset, list) or isinstance(train_dataset, tuple):
        # Multi-dataset setting

        if len(train_dataset) != len(test_dataset):
            raise ValueError('Train/test dataset lists must contain the '
                             'exact same number of datasets')

        if per_exp_classes and one_dataset_per_exp:
            raise ValueError(
                'Both per_experience_classes and one_dataset_per_exp are'
                'used, but those options are mutually exclusive')

        if fixed_class_order and one_dataset_per_exp:
            raise ValueError(
                'Both fixed_class_order and one_dataset_per_exp are'
                'used, but those options are mutually exclusive')

        seq_train_dataset, seq_test_dataset, mapping = \
            concat_datasets_sequentially(train_dataset, test_dataset)

        if one_dataset_per_exp:
            # If one_dataset_per_exp is True, each dataset will be treated as
            # a experience. In this scenario, shuffle refers to the experience
            # order, not to the class one.
            fixed_class_order, per_exp_classes = \
                _one_dataset_per_exp_class_order(mapping, shuffle, seed)

            # We pass a fixed_class_order to the NCGenericScenario
            # constructor, so we don't need shuffling.
            shuffle = False
            seed = None

            # Overrides n_experiences (and per_experience_classes, already done)
            n_experiences = len(train_dataset)
        train_dataset, test_dataset = seq_train_dataset, seq_test_dataset

    transform_groups = dict(
        train=(train_transform, None),
        eval=(eval_transform, None)
    )

    # Datasets should be instances of AvalancheDataset
    train_dataset = AvalancheDataset(
        train_dataset,
        transform_groups=transform_groups,
        initial_transform_group='train',
        dataset_type=AvalancheDatasetType.CLASSIFICATION)

    test_dataset = AvalancheDataset(
        test_dataset,
        transform_groups=transform_groups,
        initial_transform_group='eval',
        dataset_type=AvalancheDatasetType.CLASSIFICATION)

    return NCScenario(train_dataset, test_dataset, n_experiences, task_labels,
                      shuffle, seed, fixed_class_order, per_exp_classes,
                      class_ids_from_zero_from_first_exp,
                      class_ids_from_zero_in_each_exp,
                      reproducibility_data)


def ni_benchmark(
        train_dataset: Union[
            Sequence[SupportedDataset], SupportedDataset],
        test_dataset: Union[
            Sequence[SupportedDataset], SupportedDataset],
        n_experiences: int,
        *,
        task_labels: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
        balance_experiences: bool = False,
        min_class_patterns_in_exp: int = 0,
        fixed_exp_assignment: Optional[Sequence[Sequence[int]]] = None,
        train_transform=None,
        eval_transform=None,
        reproducibility_data: Optional[Dict[str, Any]] = None) \
        -> NIScenario:
    """
    This is the high-level benchmark instances generator for the
    "New Instances" (NI) case. Given a sequence of train and test datasets
    creates the continual stream of data as a series of experiences.

    This is the reference helper function for creating instances of
    Domain-Incremental benchmarks.

    The ``task_labels`` parameter determines if each incremental experience has
    an increasing task label or if, at the contrary, a default task label "0"
    has to be assigned to all experiences. This can be useful when
    differentiating between Single-Incremental-Task and Multi-Task scenarios.

    There are other important parameters that can be specified in order to tweak
    the behaviour of the resulting scenario. Please take a few minutes to read
    and understand them as they may save you a lot of work.

    This generator features an integrated reproducibility mechanism that allows
    the user to store and later re-load a scenario. For more info see the
    ``reproducibility_data`` parameter.

    :param train_dataset: A list of training datasets, or a single dataset.
    :param test_dataset: A list of test datasets, or a single test dataset.
    :param n_experiences: The number of experiences.
    :param task_labels: If True, each experience will have an ascending task
            label. If False, the task label will be 0 for all the experiences.
    :param shuffle: If True, patterns order will be shuffled.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param balance_experiences: If True, pattern of each class will be equally
        spread across all experiences. If False, patterns will be assigned to
        experiences in a complete random way. Defaults to False.
    :param min_class_patterns_in_exp: The minimum amount of patterns of
        every class that must be assigned to every experience. Compatible with
        the ``balance_experiences`` parameter. An exception will be raised if
        this constraint can't be satisfied. Defaults to 0.
    :param fixed_exp_assignment: If not None, the pattern assignment
        to use. It must be a list with an entry for each experience. Each entry
        is a list that contains the indexes of patterns belonging to that
        experience. Overrides the ``shuffle``, ``balance_experiences`` and
        ``min_class_patterns_in_exp`` parameters.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param reproducibility_data: If not None, overrides all the other
        scenario definition options, including ``fixed_exp_assignment``.
        This is usually a dictionary containing data used to
        reproduce a specific experiment. One can use the
        ``get_reproducibility_data`` method to get (and even distribute)
        the experiment setup so that it can be loaded by passing it as this
        parameter. In this way one can be sure that the same specific
        experimental setup is being used (for reproducibility purposes).
        Beware that, in order to reproduce an experiment, the same train and
        test datasets must be used. Defaults to None.

    :return: A properly initialized :class:`NIScenario` instance.
    """

    seq_train_dataset, seq_test_dataset = train_dataset, test_dataset
    if isinstance(train_dataset, list) or isinstance(train_dataset, tuple):
        if len(train_dataset) != len(test_dataset):
            raise ValueError('Train/test dataset lists must contain the '
                             'exact same number of datasets')

        seq_train_dataset, seq_test_dataset, _ = \
            concat_datasets_sequentially(train_dataset, test_dataset)

    transform_groups = dict(
        train=(train_transform, None),
        eval=(eval_transform, None)
    )

    # Datasets should be instances of AvalancheDataset
    seq_train_dataset = AvalancheDataset(
        seq_train_dataset,
        transform_groups=transform_groups,
        initial_transform_group='train',
        dataset_type=AvalancheDatasetType.CLASSIFICATION)

    seq_test_dataset = AvalancheDataset(
        seq_test_dataset,
        transform_groups=transform_groups,
        initial_transform_group='eval',
        dataset_type=AvalancheDatasetType.CLASSIFICATION)

    return NIScenario(
        seq_train_dataset, seq_test_dataset,
        n_experiences,
        task_labels,
        shuffle=shuffle, seed=seed,
        balance_experiences=balance_experiences,
        min_class_patterns_in_exp=min_class_patterns_in_exp,
        fixed_exp_assignment=fixed_exp_assignment,
        reproducibility_data=reproducibility_data)


# Here we define some high-level APIs an alias of their mid-level counterparts.
# This was done mainly because the implementation for the mid-level API is now
# quite stable and not particularly complex.
dataset_benchmark = create_multi_dataset_generic_benchmark
filelist_benchmark = create_generic_benchmark_from_filelists
paths_benchmark = create_generic_benchmark_from_paths
tensors_benchmark = create_generic_benchmark_from_tensor_lists


def _one_dataset_per_exp_class_order(
        class_list_per_exp: Sequence[Sequence[int]],
        shuffle: bool, seed: Union[int, None]) -> (List[int], Dict[int, int]):
    """
    Utility function that shuffles the class order by keeping classes from the
    same experience together. Each experience is defined by a different entry in
    the class_list_per_exp parameter.

    :param class_list_per_exp: A list of class lists, one for each experience
    :param shuffle: If True, the experience order will be shuffled. If False,
        this function will return the concatenation of lists from the
        class_list_per_exp parameter.
    :param seed: If not None, an integer used to initialize the random
        number generator.

    :returns: A class order that keeps class IDs from the same experience
        together (adjacent).
    """
    dataset_order = list(range(len(class_list_per_exp)))
    if shuffle:
        if seed is not None:
            torch.random.manual_seed(seed)
        dataset_order = torch.as_tensor(dataset_order)[
            torch.randperm(len(dataset_order))].tolist()
    fixed_class_order = []
    classes_per_exp = {}
    for dataset_position, dataset_idx in enumerate(dataset_order):
        fixed_class_order.extend(class_list_per_exp[dataset_idx])
        classes_per_exp[dataset_position] = \
            len(class_list_per_exp[dataset_idx])
    return fixed_class_order, classes_per_exp


__all__ = [
    'nc_benchmark',
    'ni_benchmark',
    'dataset_benchmark',
    'filelist_benchmark',
    'paths_benchmark',
    'tensors_benchmark'
]
