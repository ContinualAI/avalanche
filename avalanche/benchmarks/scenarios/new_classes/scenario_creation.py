################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

from typing import Sequence, Optional, Dict, List, Union

import torch


from avalanche.training.utils.transform_dataset import TransformationSubset, \
    IDatasetWithTargets, ConcatDatasetWithTargets
from .nc_generic_scenario import NCGenericScenario
from .nc_scenario import NCMultiTaskScenario, NCSingleTaskScenario


def concat_datasets_sequentially(
        train_dataset_list: Sequence[IDatasetWithTargets],
        test_dataset_list: Sequence[IDatasetWithTargets]):
    """
    Concatenates a list of datasets. This is completely different from
    :class:`ConcatDataset`, in which datasets are merged together without
    other processing. Instead, this function re-maps the datasets class IDs.
    For instance:
    let the dataset[0] contain patterns of 3 different classes,
    let the dataset[1] contain patterns of 2 different classes, then class IDs
    will be mapped as follows:

    dataset[0] class "0" -> new class ID is "0"

    dataset[0] class "1" -> new class ID is "1"

    dataset[0] class "2" -> new class ID is "2"

    dataset[1] class "0" -> new class ID is "3"

    dataset[1] class "1" -> new classID is "4"

    ... -> ...

    dataset[N-1] class "C-1" -> new class ID is "overall_n_classes-1"

    In contract, using PyTorch ConcatDataset:

    dataset[0] class "0" -> ID is "0"

    dataset[0] class "1" -> ID is "1"

    dataset[0] class "2" -> ID is "2"

    dataset[1] class "0" -> ID is "0"

    dataset[1] class "1" -> ID is "1"

    Note: ``train_dataset_list`` and ``test_dataset_list`` must have the same
    number of datasets.

    :param train_dataset_list: A list of training datasets
    :param test_dataset_list: A list of test datasets

    :returns: A concatenated dataset.
    """
    remapped_train_datasets = []
    remapped_test_datasets = []
    next_remapped_idx = 0

    # Obtain the number of classes of each dataset
    # Here we use the training set to detect the class number
    #
    # We should consider merging classes from the test set too
    classes_per_dataset = [
        len(torch.unique(
            torch.cat((torch.as_tensor(train_dataset_list[dataset_idx].targets),
                      torch.as_tensor(test_dataset_list[dataset_idx].targets)))
            )) for dataset_idx in range(len(train_dataset_list))
    ]

    new_class_ids_per_dataset = []
    for dataset_idx in range(len(train_dataset_list)):
        # The class IDs for this dataset will be in range
        # [n_classes_in_previous_datasets,
        #       n_classes_in_previous_datasets + classes_in_this_dataset)
        class_mapping = list(
            range(next_remapped_idx,
                  next_remapped_idx + classes_per_dataset[dataset_idx]))
        new_class_ids_per_dataset.append(class_mapping)

        train_set = train_dataset_list[dataset_idx]
        test_set = test_dataset_list[dataset_idx]

        # TransformationSubset is used to apply the class IDs transformation.
        # Remember, the class_mapping parameter must be a list in which:
        # new_class_id = class_mapping[original_class_id]
        remapped_train_datasets.append(
            TransformationSubset(train_set, None,
                                 class_mapping=class_mapping))
        remapped_test_datasets.append(
            TransformationSubset(test_set, None,
                                 class_mapping=class_mapping))
        next_remapped_idx += classes_per_dataset[dataset_idx]

    return ConcatDatasetWithTargets(remapped_train_datasets), \
        ConcatDatasetWithTargets(remapped_test_datasets), \
        new_class_ids_per_dataset


def create_nc_single_dataset_sit_scenario(
        train_dataset: IDatasetWithTargets,
        test_dataset: IDatasetWithTargets,
        n_batches: int, shuffle: bool = True,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        per_batch_classes: Optional[Dict[int, int]] = None,
        remap_class_ids: bool = False) -> NCSingleTaskScenario:
    """
    Creates a "New Classes - Single Incremental Task" scenario given a couple
    of train and test datasets.

    :param train_dataset: The training dataset.
    :param test_dataset: A list of test dataset.
    :param n_batches: The number of batches.
    :param shuffle: If True, class order will be shuffled.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, values of ``shuffle`` and ``seed`` will be used to
        define the class order. If non-None, ``shuffle`` and ``seed`` parameters
        will be ignored. Defaults to None.
    :param per_batch_classes: Is not None, a dictionary whose keys are
        (0-indexed) batch IDs and their values are the number of classes
        to include in the respective batches. The dictionary doesn't
        have to contain a key for each batch! All the remaining batches
        will contain an equal amount of the remaining classes. The
        remaining number of classes must be divisible without remainder
        by the remaining number of batches. For instance,
        if you want to include 50 classes in the first batch while equally
        distributing remaining classes across remaining batches,
        just pass the "{0: 50}" dictionary as the ``per_batch_classes``
        parameter. Defaults to None.
    :param remap_class_ids: If True, original class IDs will be
        remapped so that they will appear as having an ascending order.
        For instance, if the resulting class order after shuffling
        (or defined by fixed_class_order) is [23, 34, 11, 7, 6, ...] and
        remap_class_indexes is True, then all the patterns belonging to
        class 23 will appear as belonging to class "0", class "34" will
        be mapped to "1", class "11" to "2" and so on. This is very
        useful when drawing confusion matrices and when dealing with
        algorithms with dynamic head expansion. Defaults to False.

    :returns: A :class:`NCMultiTaskScenario` instance initialized for the the
        SIT scenario.
    """

    base_scenario = NCGenericScenario(
        train_dataset, test_dataset,
        n_batches=n_batches, shuffle=shuffle, seed=seed,
        fixed_class_order=fixed_class_order,
        per_batch_classes=per_batch_classes,
        remap_class_indexes=remap_class_ids)

    return NCSingleTaskScenario(base_scenario)


def create_nc_single_dataset_multi_task_scenario(
        train_dataset: IDatasetWithTargets,
        test_dataset: IDatasetWithTargets,
        n_tasks: int, shuffle: bool = True,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        per_task_classes: Optional[Dict[int, int]] = None,
        classes_ids_from_zero_in_each_task: bool = True) \
        -> NCMultiTaskScenario:
    """
    Creates a "New Classes - Multi Task" scenario given a couple
    of train and test datasets.

    :param train_dataset: The training dataset.
    :param test_dataset: A list of test dataset.
    :param n_tasks: The number of batches.
    :param shuffle: If True, class order will be shuffled.
    :param seed: A valid int used to initialize the random number generator. Can
        be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, values of shuffle and seed will be used to define the
        class order. If non-None, shuffle and seed parameters will be ignored.
        Defaults to None.
    :param per_task_classes: Is not None, a dictionary whose keys are
        (0-indexed) task IDs and their values are the number of classes
        to include in the respective batches. The dictionary doesn't
        have to contain a key for each task! All the remaining batches
        will contain an equal amount of the remaining classes. The
        remaining number of classes must be divisible without remainder
        by the remaining number of batches. For instance,
        if you want to include 50 classes in the first task while equally
        distributing remaining classes across remaining batches,
        just pass the "{0: 50}" dictionary as the per_task_classes
        parameter. Defaults to None.
    :param classes_ids_from_zero_in_each_task: If True, original class IDs will
        be mapped to range [0, n_classes_in_task) for each task. If False,
        each class will keep its original ID as defined in the input
        datasets. Defaults to True.

    :returns: A :class:`NCMultiTaskScenario` instance.
    """

    base_scenario = NCGenericScenario(
        train_dataset, test_dataset,
        n_batches=n_tasks, shuffle=shuffle, seed=seed,
        fixed_class_order=fixed_class_order,
        per_batch_classes=per_task_classes,
        remap_class_indexes=False)

    return NCMultiTaskScenario(
        base_scenario,
        classes_ids_from_zero_in_each_task=classes_ids_from_zero_in_each_task)


def _one_dataset_per_batch_class_order(
        class_list_per_batch: Sequence[Sequence[int]],
        shuffle: bool, seed: Union[int, None]) -> (List[int], Dict[int, int]):
    """
    Utility function that shuffles the class order by keeping classes from the
    same batch together. Each batch is defined by a different entry in the
    class_list_per_batch parameter.

    Args:
    :param class_list_per_batch: A list of class lists, one for each batch
    :param shuffle: If True, the batch order will be shuffled. If False,
        this function will return the concatenation of lists from the
        class_list_per_batch parameter.
    :param seed: If not None, an integer used to initialize the random
        number generator.

    :returns: A class order that keeps class IDs from the same batch together
        (adjacent).
    """
    dataset_order = list(range(len(class_list_per_batch)))
    if shuffle:
        if seed is not None:
            torch.random.manual_seed(seed)
        dataset_order = torch.as_tensor(dataset_order)[
            torch.randperm(len(dataset_order))].tolist()
    fixed_class_order = []
    classes_per_batch = {}
    for dataset_position, dataset_idx in enumerate(dataset_order):
        fixed_class_order.extend(class_list_per_batch[dataset_idx])
        classes_per_batch[dataset_position] = \
            len(class_list_per_batch[dataset_idx])
    return fixed_class_order, classes_per_batch


def create_nc_multi_dataset_sit_scenario(
        train_dataset_list: Sequence[IDatasetWithTargets],
        test_dataset_list: Sequence[IDatasetWithTargets],
        n_batches: int, shuffle: bool = True,
        seed: Optional[int] = None,
        per_batch_classes: Optional[Dict[int, int]] = None,
        one_dataset_per_batch: bool = False) \
        -> NCSingleTaskScenario:
    """
    Creates a "New Classes - Single Incremental Task" scenario given a list of
    datasets and the number of batches. The datasets will be merged together.

    Note: train_dataset_list and test_dataset_list must have the same number of
    datasets.

    :param train_dataset_list: A list of training datasets.
    :param test_dataset_list: A list of test datasets.
    :param n_batches: The number of batches.
    :param shuffle: If True and one_dataset_per_batch is False, class order
        will be shuffled. If True and one_dataset_per_batch is True,
        batch order will be shuffled.
    :param seed: A valid int used to initialize the random number generator. Can
        be None.
    :param per_batch_classes: Is not None, a dictionary whose keys are
        (0-indexed) batch IDs and their values are the number of classes
        to include in the respective batches. The dictionary doesn't
        have to contain a key for each batch! All the remaining batches
        will contain an equal amount of the remaining classes. The
        remaining number of classes must be divisible without remainder
        by the remaining number of batches. Defaults to None. This
        parameter is mutually exclusive with one_dataset_per_batch.
    :param one_dataset_per_batch: If True, each dataset will be treated as a
        batch. Mutually exclusive with the per_task_classes parameter.
        Overrides the n_batches parameter.

    :returns: A :class:`NCMultiTaskScenario` instance initialized for the
        the SIT scenario.
    """
    if len(train_dataset_list) != len(test_dataset_list):
        raise ValueError('Train/test dataset lists must contain the '
                         'exact same number of datasets')

    if per_batch_classes and one_dataset_per_batch:
        raise ValueError('Both per_task_classes and one_dataset_per_batch are'
                         'True, but those options are mutually exclusive')

    seq_train_dataset, seq_test_dataset, mapping = \
        concat_datasets_sequentially(train_dataset_list, test_dataset_list)

    fixed_class_order = None

    if one_dataset_per_batch:
        # If one_dataset_per_batch is True, each dataset will be treated as
        # a batch. In this scenario, shuffle refers to the batch order,
        # not to the class one.
        fixed_class_order, per_batch_classes = \
            _one_dataset_per_batch_class_order(mapping, shuffle, seed)

        # We pass a fixed_class_order to the NCGenericGenericScenario
        # constructor, so we don't need shuffling.
        shuffle = False
        seed = None

        # Overrides n_batches (and per_batch_classes, already done)
        n_batches = len(train_dataset_list)

    base_scenario = NCGenericScenario(
        seq_train_dataset, seq_test_dataset,
        n_batches=n_batches, shuffle=shuffle, seed=seed,
        fixed_class_order=fixed_class_order,
        per_batch_classes=per_batch_classes)

    return NCSingleTaskScenario(base_scenario)


def create_nc_multi_dataset_multi_task_scenario(
        train_dataset_list: Sequence[IDatasetWithTargets],
        test_dataset_list: Sequence[IDatasetWithTargets],
        shuffle: bool = True, seed: Optional[int] = None,
        classes_ids_from_zero_in_each_task: bool = True) \
        -> NCMultiTaskScenario:
    """
    Creates a "New Classes - Multi Task" scenario given a list of
    datasets and the number of batches. Each dataset will be treated as a task.
    This means that the overall number of batches will be
    len(train_dataset_list).

    Note: train_dataset_list and test_dataset_list must have the same number of
    datasets.

    Args:
    :param train_dataset_list: A list of training datasets
    :param test_dataset_list: A list of test datasets
    :param shuffle: If True, task order will be shuffled. Defaults to True.
    :param seed: A valid int used to initialize the random number generator. Can
        be None.
    :param classes_ids_from_zero_in_each_task: If True, original class IDs will
        be kept as is, that is, in range [0, n_classes_in_task) for each task.
        If False, each class ID will be remapped so that each class ID will
        appear once across all batches. Defaults to True.
        For instance, if the resulting dataset (task) order after shuffling
        is [dataset2, dataset3, dataset0, dataset1] and
        classes_ids_from_zero_in_each_task is False, then all the classes
        belonging to dataset2 will appear as having IDs in range
        [0, n_classes_in_dataset2) while classes in dataset3 will appear
        as having IDs in range [n_classes_in_dataset2,
        n_classes_in_dataset2+n_classes_in_dataset3) and so on.

    :Returns: A :class:`NCMultiTaskScenario` instance.
    """
    if len(train_dataset_list) != len(test_dataset_list):
        raise ValueError('Train/test dataset lists must contain the '
                         'exact same number of datasets')

    # First, concatenate the datasets
    # The resulting dataset will feature class IDs in range
    # [0, n_overall_classes)
    seq_train_dataset, seq_test_dataset, mapping = \
        concat_datasets_sequentially(train_dataset_list, test_dataset_list)

    # We can't just shuffle the class order because each dataset is a task:
    # doing so will result in having classes of different batches mixed
    # together. Use the _one_dataset_per_batch_class_order utility to get a
    # shuffled task order. _one_dataset_per_batch_class_order doesn't shuffle
    # the task internal class order. That is, this utility is suitable when
    # dealing with Multi Task scenarios, not Multi Incremental Task ones!
    fixed_class_order, classes_per_task = _one_dataset_per_batch_class_order(
        mapping, shuffle, seed
    )

    base_scenario = NCGenericScenario(
        seq_train_dataset, seq_test_dataset,
        n_batches=len(train_dataset_list), shuffle=False, seed=None,
        fixed_class_order=fixed_class_order, per_batch_classes=classes_per_task)

    return NCMultiTaskScenario(
        base_scenario,
        classes_ids_from_zero_in_each_task=classes_ids_from_zero_in_each_task)


__all__ = ['create_nc_single_dataset_sit_scenario',
           'create_nc_single_dataset_multi_task_scenario',
           'create_nc_multi_dataset_sit_scenario',
           'create_nc_multi_dataset_multi_task_scenario',
           'concat_datasets_sequentially']
