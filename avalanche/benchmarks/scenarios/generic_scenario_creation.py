from pathlib import Path
from typing import Sequence, Union, SupportsInt

from torch import Tensor

from avalanche.training.utils import IDatasetWithTargets, \
    ConcatDatasetWithTargets, TransformationTensorDataset
from .datasets_from_filelists import datasets_from_filelists
from .generic_cl_scenario import GenericCLScenario


def create_multi_dataset_generic_scenario(
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

    # GenericCLScenario accepts a single training+test sets couple along with
    # the respective list of patterns indexes to include in each step.
    # This means that we have to concat the list of train and test sets
    # and create, for each step, a of indexes.
    # Each dataset describes a different step so the lists of indexes will
    # just be ranges of ascending indexes.
    train_structure = []
    concat_train_dataset = ConcatDatasetWithTargets(train_dataset_list)
    next_idx = 0
    for train_dataset in train_dataset_list:
        end_idx = next_idx + len(train_dataset)
        train_structure.append(range(next_idx, end_idx))
        next_idx = end_idx

    test_structure = []
    if complete_test_set_only:
        # If complete_test_set_only is True, we can leave test_structure = []
        # In this way, GenericCLScenario will consider the whole test set.
        #
        # We don't offer a way to reduce the test set here. However, consider
        # that the user may reduce the test set by creating a subset and passing
        # it to this function directly.
        if len(test_dataset_list) != 1:
            raise ValueError('Test must contain 1 element when'
                             'complete_test_set_only is True')
        concat_test_dataset = test_dataset_list[0]
    else:
        concat_test_dataset = ConcatDatasetWithTargets(test_dataset_list)
        test_structure = []
        next_idx = 0
        for test_dataset in test_dataset_list:
            end_idx = next_idx + len(test_dataset)
            test_structure.append(range(next_idx, end_idx))
            next_idx = end_idx

    # GenericCLScenario constructor will also check that the same amount of
    # train/test sets + task_labels have been defined.
    return GenericCLScenario(
        concat_train_dataset, concat_test_dataset, train_structure,
        test_structure, task_labels,
        return_complete_test_set_only=complete_test_set_only)


def create_generic_scenario_from_filelists(
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

    train_datasets, test_dataset = datasets_from_filelists(
        root, train_file_lists, test_file_lists,
        complete_test_set_only=complete_test_set_only,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        test_transform=test_transform,
        test_target_transform=test_target_transform)

    return create_multi_dataset_generic_scenario(
        train_datasets, test_dataset, task_labels,
        complete_test_set_only=complete_test_set_only)


def create_generic_scenario_from_tensors(
        train_datasets_x: Sequence[Tensor],
        train_datasets_y: Sequence[Sequence[SupportsInt]],
        test_datasets_x: Sequence[Tensor],
        test_datasets_y: Sequence[Sequence[SupportsInt]],
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

    :param train_datasets_x: A list of Tensors (one per step) containing the
        patterns of the training sets.
    :param train_datasets_y: A list of Tensors or int lists containing the
        labels of the patterns of the training sets. Must contain the same
        number of elements of ``train_datasets_x``.
    :param test_datasets_x: A list of Tensors (one per step) containing the
        patterns of the test sets.
    :param test_datasets_y: A list of Tensors or int lists containing the
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

    if len(train_datasets_x) != len(train_datasets_y):
        raise ValueError('train_datasets_x and train_datasets_y must contain'
                         'the same amount of elements')

    if len(test_datasets_x) != len(test_datasets_y):
        raise ValueError('test_datasets_x and test_datasets_y must contain'
                         'the same amount of elements')

    train_datasets = [
        TransformationTensorDataset(
            dataset_x, dataset_y, transform=train_transform,
            target_transform=train_target_transform)
        for dataset_x, dataset_y in
        zip(train_datasets_x, train_datasets_y)]

    test_datasets = [
        TransformationTensorDataset(
            dataset_x, dataset_y, transform=test_transform,
            target_transform=test_target_transform)
        for dataset_x, dataset_y in
        zip(test_datasets_x, test_datasets_y)]

    return create_multi_dataset_generic_scenario(
        train_datasets, test_datasets, task_labels,
        complete_test_set_only=complete_test_set_only)


__all__ = ['create_multi_dataset_generic_scenario',
           'create_generic_scenario_from_filelists',
           'create_generic_scenario_from_tensors']
