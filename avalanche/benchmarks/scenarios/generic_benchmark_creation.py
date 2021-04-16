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

""" This module contains mid-level benchmark generators.
Consider using the higher-level ones found in benchmark_generators. If none of
them fit your needs, then the helper functions here listed may help.
"""

from pathlib import Path
from typing import Sequence, Union, Any, Tuple, Dict, Optional

from avalanche.benchmarks.utils import AvalancheTensorDataset, \
    SupportedDataset, AvalancheDataset, FilelistDataset, \
    PathsDataset, common_paths_root
from .generic_cl_scenario import GenericCLScenario
from ..utils.avalanche_dataset import AvalancheDatasetType


def create_multi_dataset_generic_benchmark(
        train_datasets: Sequence[SupportedDataset],
        test_datasets: Sequence[SupportedDataset],
        *,
        other_streams_datasets: Dict[str, Sequence[SupportedDataset]] = None,
        complete_test_set_only: bool = False,
        train_transform=None, train_target_transform=None,
        eval_transform=None, eval_target_transform=None,
        other_streams_transforms: Dict[str, Tuple[Any, Any]] = None,
        dataset_type: AvalancheDatasetType = None) -> GenericCLScenario:
    """
    Creates a benchmark instance given a list of datasets. Each dataset will be
    considered as a separate experience.

    Contents of the datasets must already be set, including task labels.
    Transformations will be applied if defined.

    This function allows for the creation of custom streams as well.
    While "train" and "test" datasets must always be set, the experience list
    for other streams can be defined by using the `other_streams_datasets`
    parameter.

    If transformations are defined, they will be applied to the datasets
    of the related stream.

    :param train_datasets: A list of training datasets.
    :param test_datasets: A list of test datasets.
    :param other_streams_datasets: A dictionary describing the content of custom
        streams. Keys must be valid stream names (letters and numbers,
        not starting with a number) while the value must be a list of dataset.
        If this dictionary contains the definition for "train" or "test"
        streams then those definition will override the `train_datasets` and
        `test_datasets` parameters.
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
    :param other_streams_transforms: Transformations to apply to custom
        streams. If no transformations are defined for a custom stream,
        then "train" transformations will be used. This parameter must be a
        dictionary mapping stream names to transformations. The transformations
        must be a two elements tuple where the first element defines the
        X transformation while the second element is the Y transformation.
        Those elements can be None. If this dictionary contains the
        transformations for "train" or "test" streams then those transformations
        will override the `train_transform`, `train_target_transform`,
        `eval_transform` and `eval_target_transform` parameters.
    :param dataset_type: The type of the dataset. Defaults to None, which
        means that the type will be obtained from the input datasets. If input
        datasets are not instances of :class:`AvalancheDataset`, the type
        UNDEFINED will be used.

    :returns: A :class:`GenericCLScenario` instance.
    """

    transform_groups = dict(
        train=(train_transform, train_target_transform),
        eval=(eval_transform, eval_target_transform))

    if other_streams_transforms is not None:
        for stream_name, stream_transforms in other_streams_transforms.items():
            if isinstance(stream_transforms, Sequence):
                if len(stream_transforms) == 1:
                    # Suppose we got only the transformation for X values
                    stream_transforms = (stream_transforms[0], None)
            else:
                # Suppose it's the transformation for X values
                stream_transforms = (stream_transforms, None)

            transform_groups[stream_name] = stream_transforms

    input_streams = dict(
        train=train_datasets,
        test=test_datasets)
    input_streams = {**input_streams, **other_streams_datasets}

    if complete_test_set_only:
        if len(input_streams['test']) != 1:
            raise ValueError('Test stream must contain one experience when'
                             'complete_test_set_only is True')

    stream_definitions = dict()

    for stream_name, dataset_list in input_streams.items():
        initial_transform_group = 'train'
        if stream_name in transform_groups:
            initial_transform_group = stream_name

        stream_datasets = []
        for dataset_idx in range(len(dataset_list)):
            dataset = dataset_list[dataset_idx]
            stream_datasets.append(AvalancheDataset(
                dataset,
                transform_groups=transform_groups,
                initial_transform_group=initial_transform_group,
                dataset_type=dataset_type))
        stream_definitions[stream_name] = (stream_datasets,)

    return GenericCLScenario(
        stream_definitions=stream_definitions,
        complete_test_set_only=complete_test_set_only)


def create_generic_benchmark_from_filelists(
        root: Optional[Union[str, Path]],
        train_file_lists: Sequence[Union[str, Path]],
        test_file_lists: Sequence[Union[str, Path]],
        *,
        other_streams_file_lists: Dict[str, Sequence[Union[str, Path]]] = None,
        task_labels: Sequence[int],
        complete_test_set_only: bool = False,
        train_transform=None, train_target_transform=None,
        eval_transform=None, eval_target_transform=None,
        other_streams_transforms: Dict[str, Tuple[Any, Any]] = None) \
        -> GenericCLScenario:
    """
    Creates a benchmark instance given a list of filelists and the respective
    task labels. A separate dataset will be created for each filelist and each
    of those datasets will be considered a separate experience.

    This helper functions is the best shot when loading Caffe-style dataset
    based on filelists.

    Beware that this helper function is limited is the following two aspects:

    - The resulting benchmark instance and the intermediate datasets used to
      populate it will be of type CLASSIFICATION. There is no way to change
      this.
    - Task labels can only be defined by choosing a single task label for
      each experience (the same task label is applied to all patterns of
      experiences sharing the same position in different streams).

    Despite those constraints, this helper function is usually sufficiently
    powerful to cover most continual learning benchmarks based on file lists.

    When in need to create a similar benchmark instance starting from an
    in-memory list of paths, then the similar helper function
    :func:`create_generic_benchmark_from_paths` can be used.

    When in need to create a benchmark instance in which task labels are defined
    in a more fine-grained way, then consider using
    :func:`create_multi_dataset_generic_benchmark` by passing properly
    initialized :class:`AvalancheDataset` instances.

    :param root: The root path of the dataset. Can be None.
    :param train_file_lists: A list of filelists describing the
        paths of the training patterns for each experience.
    :param test_file_lists: A list of filelists describing the
        paths of the test patterns for each experience.
    :param other_streams_file_lists: A dictionary describing the content of
        custom streams. Keys must be valid stream names (letters and numbers,
        not starting with a number) while the value must be a list of filelists
        (same as `train_file_lists` and `test_file_lists` parameters). If this
        dictionary contains the definition for "train" or "test" streams then
        those definition will  override the `train_file_lists` and
        `test_file_lists` parameters.
    :param task_labels: A list of task labels. Must contain at least a value
        for each experience. Each value describes the task label that will be
        applied to all patterns of a certain experience. For more info on that,
        see the function description.
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
    :param other_streams_transforms: Transformations to apply to custom
        streams. If no transformations are defined for a custom stream,
        then "train" transformations will be used. This parameter must be a
        dictionary mapping stream names to transformations. The transformations
        must be a two elements tuple where the first element defines the
        X transformation while the second element is the Y transformation.
        Those elements can be None. If this dictionary contains the
        transformations for "train" or "test" streams then those transformations
        will override the `train_transform`, `train_target_transform`,
        `eval_transform` and `eval_target_transform` parameters.

    :returns: A :class:`GenericCLScenario` instance.
    """

    input_streams = dict(
        train=train_file_lists,
        test=test_file_lists)
    input_streams = {**input_streams, **other_streams_file_lists}

    stream_definitions = dict()

    for stream_name, file_lists in input_streams.items():
        stream_datasets = []
        for exp_id, f_list in enumerate(file_lists):

            f_list_dataset = FilelistDataset(root, f_list)
            stream_datasets.append(AvalancheDataset(
                f_list_dataset,
                task_labels=task_labels[exp_id]))

        stream_definitions[stream_name] = stream_datasets

    return create_multi_dataset_generic_benchmark(
        [], [],
        other_streams_datasets=stream_definitions,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        eval_transform=eval_transform,
        eval_target_transform=eval_target_transform,
        complete_test_set_only=complete_test_set_only,
        other_streams_transforms=other_streams_transforms,
        dataset_type=AvalancheDatasetType.CLASSIFICATION)


FileAndLabel = Tuple[Union[str, Path], int]


def create_generic_benchmark_from_paths(
        train_lists_of_files: Sequence[Sequence[FileAndLabel]],
        test_lists_of_files: Union[Sequence[FileAndLabel],
                                   Sequence[Sequence[FileAndLabel]]],
        *,
        other_streams_lists_of_files: Dict[str, Sequence[
            Sequence[FileAndLabel]]] = None,
        task_labels: Sequence[int],
        complete_test_set_only: bool = False,
        train_transform=None, train_target_transform=None,
        eval_transform=None, eval_target_transform=None,
        other_streams_transforms: Dict[str, Tuple[Any, Any]] = None,
        dataset_type: AvalancheDatasetType = AvalancheDatasetType.UNDEFINED) \
        -> GenericCLScenario:
    """
    Creates a benchmark instance given a sequence of lists of files. A separate
    dataset will be created for each list. Each of those datasets
    will be considered a separate experience.

    This is very similar to :func:`create_generic_benchmark_from_filelists`,
    with the main difference being that
    :func:`create_generic_benchmark_from_filelists` accepts, for each
    experience, a file list formatted in Caffe-style. On the contrary, this
    accepts a list of tuples where each tuple contains two elements: the full
    path to the pattern and its label. Optionally, the tuple may contain a third
    element describing the bounding box of the element to crop. This last
    bounding box may be useful when trying to extract the part of the image
    depicting the desired element.

    Apart from that, the same limitations of
    :func:`create_generic_benchmark_from_filelists` regarding task labels apply.

    The label of each pattern doesn't have to be an int. Also, a dataset type
    can be defined.

    :param train_lists_of_files: A list of lists. Each list describes the paths
        and labels of patterns to include in that training experience, as
        tuples. Each tuple must contain two elements: the full path to the
        pattern and its class label. Optionally, the tuple may contain a
        third element describing the bounding box to use for cropping (top,
        left, height, width).
    :param test_lists_of_files: A list of lists. Each list describes the paths
        and labels of patterns to include in that test experience, as tuples.
        Each tuple must contain two elements: the full path to the pattern
        and its class label. Optionally, the tuple may contain a third element
        describing the bounding box to use for cropping (top, left, height,
        width).
    :param other_streams_lists_of_files: A dictionary describing the content of
        custom streams. Keys must be valid stream names (letters and numbers,
        not starting with a number) while the value follow the same structure
        of `train_lists_of_files` and `test_lists_of_files` parameters. If this
        dictionary contains the definition for "train" or "test" streams then
        those definition will  override the `train_lists_of_files` and
        `test_lists_of_files` parameters.
    :param task_labels: A list of task labels. Must contain at least a value
        for each experience. Each value describes the task label that will be
        applied to all patterns of a certain experience. For more info on that,
        see the function description.
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
    :param other_streams_transforms: Transformations to apply to custom
        streams. If no transformations are defined for a custom stream,
        then "train" transformations will be used. This parameter must be a
        dictionary mapping stream names to transformations. The transformations
        must be a two elements tuple where the first element defines the
        X transformation while the second element is the Y transformation.
        Those elements can be None. If this dictionary contains the
        transformations for "train" or "test" streams then those transformations
        will override the `train_transform`, `train_target_transform`,
        `eval_transform` and `eval_target_transform` parameters.
    :param dataset_type: The type of the dataset. Defaults to UNDEFINED.

    :returns: A :class:`GenericCLScenario` instance.
    """

    input_streams = dict(
        train=train_lists_of_files,
        test=test_lists_of_files)
    input_streams = {**input_streams, **other_streams_lists_of_files}

    stream_definitions = dict()

    for stream_name, lists_of_files in input_streams.items():
        stream_datasets = []
        for exp_id, list_of_files in enumerate(lists_of_files):
            common_root, exp_paths_list = common_paths_root(list_of_files)
            paths_dataset = PathsDataset(common_root, exp_paths_list)
            stream_datasets.append(AvalancheDataset(
                paths_dataset,
                task_labels=task_labels[exp_id]))

        stream_definitions[stream_name] = stream_datasets

    return create_multi_dataset_generic_benchmark(
        [], [],
        other_streams_datasets=stream_definitions,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        eval_transform=eval_transform,
        eval_target_transform=eval_target_transform,
        complete_test_set_only=complete_test_set_only,
        other_streams_transforms=other_streams_transforms,
        dataset_type=dataset_type)


def create_generic_benchmark_from_tensor_lists(
        train_tensors: Sequence[Sequence[Any]],
        test_tensors: Sequence[Sequence[Any]],
        *,
        other_streams_tensors: Dict[str, Sequence[Sequence[Any]]] = None,
        task_labels: Sequence[int],
        complete_test_set_only: bool = False,
        train_transform=None, train_target_transform=None,
        eval_transform=None, eval_target_transform=None,
        other_streams_transforms: Dict[str, Tuple[Any, Any]] = None,
        dataset_type: AvalancheDatasetType = None) -> GenericCLScenario:
    """
    Creates a benchmark instance given lists of Tensors. A separate dataset will
    be created from each Tensor tuple (x, y, z, ...) and each of those training
    datasets will be considered a separate training experience. Using this
    helper function is the lowest-level way to create a Continual Learning
    scenario. When possible, consider using higher level helpers.

    Experiences are defined by passing lists of tensors as the `train_tensors`,
    `test_tensors` (and `other_streams_tensors`) parameters. Those parameters
    must be lists containing lists of tensors, one list for each experience.
    Each tensor defines the value of a feature ("x", "y", "z", ...) for all
    patterns of that experience.

    By default the second tensor of each experience will be used to fill the
    `targets` value (label of each pattern).

    Beware that task labels can only be defined by choosing a single task label
    for each experience (the same task label is applied to all patterns of
    experiences sharing the same position in different streams).

    When in need to create a benchmark instance in which task labels are defined
    in a more fine-grained way, then consider using
    :func:`create_multi_dataset_generic_benchmark` by passing properly
    initialized :class:`AvalancheDataset` instances.

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
    :param other_streams_tensors: A dictionary describing the content of
        custom streams. Keys must be valid stream names (letters and numbers,
        not starting with a number) while the value follow the same structure
        of `train_tensors` and `test_tensors` parameters. If this
        dictionary contains the definition for "train" or "test" streams then
        those definition will  override the `train_tensors` and `test_tensors`
        parameters.
    :param task_labels: A list of task labels. Must contain at least a value
        for each experience. Each value describes the task label that will be
        applied to all patterns of a certain experience. For more info on that,
        see the function description.
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
    :param other_streams_transforms: Transformations to apply to custom
        streams. If no transformations are defined for a custom stream,
        then "train" transformations will be used. This parameter must be a
        dictionary mapping stream names to transformations. The transformations
        must be a two elements tuple where the first element defines the
        X transformation while the second element is the Y transformation.
        Those elements can be None. If this dictionary contains the
        transformations for "train" or "test" streams then those transformations
        will override the `train_transform`, `train_target_transform`,
        `eval_transform` and `eval_target_transform` parameters.
    :param dataset_type: The type of the dataset. Defaults to UNDEFINED.

    :returns: A :class:`GenericCLScenario` instance.
    """

    input_streams = dict(
        train=train_tensors,
        test=test_tensors)
    input_streams = {**input_streams, **other_streams_tensors}

    stream_definitions = dict()

    for stream_name, list_of_exps_tensors in input_streams.items():
        stream_datasets = []
        for exp_id, exp_tensors in enumerate(list_of_exps_tensors):
            stream_datasets.append(AvalancheTensorDataset(
                *exp_tensors, dataset_type=dataset_type,
                task_labels=task_labels[exp_id]))

        stream_definitions[stream_name] = stream_datasets

    return create_multi_dataset_generic_benchmark(
        [], [],
        other_streams_datasets=stream_definitions,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        eval_transform=eval_transform,
        eval_target_transform=eval_target_transform,
        complete_test_set_only=complete_test_set_only,
        other_streams_transforms=other_streams_transforms,
        dataset_type=dataset_type)


__all__ = [
    'create_multi_dataset_generic_benchmark',
    'create_generic_benchmark_from_filelists',
    'create_generic_benchmark_from_paths',
    'create_generic_benchmark_from_tensor_lists',
]
