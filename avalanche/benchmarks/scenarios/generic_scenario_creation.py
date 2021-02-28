from pathlib import Path
from typing import Sequence, Union, SupportsInt, Any, Tuple

from torch import Tensor

from avalanche.benchmarks.utils import AvalancheTensorDataset, \
    AvalancheConcatDataset, as_transformation_dataset, \
    SupportedDataset, datasets_from_paths
from avalanche.benchmarks.utils import datasets_from_filelists
from .generic_cl_scenario import GenericCLScenario
from avalanche.benchmarks.utils.dataset_utils import LazyConcatTargets,\
    ConstantSequence


def create_multi_dataset_generic_scenario(
        train_dataset_list: Sequence[SupportedDataset],
        test_dataset_list: Sequence[SupportedDataset],
        task_labels: Sequence[int],
        complete_test_set_only: bool = False) -> GenericCLScenario:
    """
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

    :returns: A :class:`GenericCLScenario` instance.
    """

    # GenericCLScenario accepts a single training+test sets couple along with
    # the respective list of patterns indexes to include in each experience.
    # This means that we have to concat the list of train and test sets
    # and create, for each experience, a of indexes.
    # Each dataset describes a different experience so the lists of indexes will
    # just be ranges of ascending indexes.
    train_structure = []
    pattern_train_task_labels = []
    concat_train_dataset = AvalancheConcatDataset(train_dataset_list)
    next_idx = 0
    for dataset_idx, train_dataset in enumerate(train_dataset_list):
        end_idx = next_idx + len(train_dataset)
        train_structure.append(range(next_idx, end_idx))
        pattern_train_task_labels.append(
            ConstantSequence(task_labels[dataset_idx], len(train_dataset)))
        next_idx = end_idx

    pattern_train_task_labels = LazyConcatTargets(pattern_train_task_labels)

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
        concat_test_dataset = as_transformation_dataset(test_dataset_list[0])
        pattern_test_task_labels = ConstantSequence(0, len(concat_test_dataset))
    else:
        concat_test_dataset = AvalancheConcatDataset(test_dataset_list)
        test_structure = []
        pattern_test_task_labels = []
        next_idx = 0
        for dataset_idx, test_dataset in enumerate(test_dataset_list):
            end_idx = next_idx + len(test_dataset)
            test_structure.append(range(next_idx, end_idx))
            pattern_test_task_labels.append(
                ConstantSequence(task_labels[dataset_idx], len(test_dataset)))
            next_idx = end_idx
        pattern_test_task_labels = LazyConcatTargets(pattern_test_task_labels)
    concat_test_dataset = concat_test_dataset.eval()

    task_labels = [[x] for x in task_labels]

    # GenericCLScenario constructor will also check that the same amount of
    # train/test sets + task_labels have been defined.
    return GenericCLScenario(
        concat_train_dataset, concat_test_dataset,
        concat_train_dataset, concat_test_dataset, train_structure,
        test_structure, task_labels,
        pattern_train_task_labels,
        pattern_test_task_labels,
        complete_test_set_only=complete_test_set_only)


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
    those training datasets will be considered a separate training experience.
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


FileAndLabel = Tuple[Union[str, Path], int]


def create_generic_scenario_from_paths(
        train_list_of_files: Sequence[Sequence[FileAndLabel]],
        test_list_of_files: Union[Sequence[FileAndLabel],
                                  Sequence[Sequence[FileAndLabel]]],
        task_labels: Sequence[int],
        complete_test_set_only: bool = False,
        train_transform=None, train_target_transform=None,
        test_transform=None, test_target_transform=None) -> GenericCLScenario:
    """
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

    In its base form, this function accepts a list for the test datsets that
    must contain the same amount of elements of the training list.
    Those pairs of datasets are then used to create the "past", "cumulative"
    (a.k.a. growing) and "future" test sets. However, in certain Continual
    Learning scenarios only the concept of "complete" test set makes sense. In
    that case, the ``complete_test_set_only`` should be set to True (see the
    parameter description for more info).

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

    train_datasets, test_dataset = datasets_from_paths(
        train_list_of_files, test_list_of_files,
        complete_test_set_only=complete_test_set_only,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        test_transform=test_transform,
        test_target_transform=test_target_transform)

    return create_multi_dataset_generic_scenario(
        train_datasets, test_dataset, task_labels,
        complete_test_set_only=complete_test_set_only)


def create_generic_scenario_from_tensors(
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
    training experience. Contents of the datasets will not be changed, including
    the targets. Using this helper function is the lower level way to create a
    Continual Learning scenario. When possible, consider using higher level
    helpers.

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

    if len(train_data_x) != len(train_data_y):
        raise ValueError('train_data_x and train_data_y must contain'
                         ' the same amount of elements')

    if type(test_data_x) != type(test_data_y):
        raise ValueError('test_data_x and test_data_y must be of'
                         ' the same type')

    if isinstance(test_data_x, Tensor):
        test_data_x = [test_data_x]
        test_data_y = [test_data_y]
    else:
        if len(test_data_x) != len(test_data_y):
            raise ValueError('test_data_x and test_data_y must contain'
                             ' the same amount of elements')

    transform_groups = dict(train=(train_transform, train_target_transform),
                            test=(test_transform, test_target_transform))

    train_datasets = [
        AvalancheTensorDataset(
            dataset_x, dataset_y,
            transform_groups=transform_groups,
            initial_transform_group='train')
        for dataset_x, dataset_y in
        zip(train_data_x, train_data_y)]

    test_datasets = [
        AvalancheTensorDataset(
            dataset_x, dataset_y,
            transform_groups=transform_groups,
            initial_transform_group='test')
        for dataset_x, dataset_y in
        zip(test_data_x, test_data_y)]

    return create_multi_dataset_generic_scenario(
        train_datasets, test_datasets, task_labels,
        complete_test_set_only=complete_test_set_only)


__all__ = [
    'create_multi_dataset_generic_scenario',
    'create_generic_scenario_from_filelists',
    'create_generic_scenario_from_paths',
    'create_generic_scenario_from_tensors'
]
