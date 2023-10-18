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
from typing import (
    Generator,
    List,
    Mapping,
    Sequence,
    Union,
    Any,
    Tuple,
    Dict,
    Optional,
    Iterable,
    NamedTuple,
)

from avalanche.benchmarks.utils.classification_dataset import (
    _make_taskaware_tensor_classification_dataset,
    _make_taskaware_classification_dataset,
)

from avalanche.benchmarks.utils import (
    SupportedDataset,
    FilelistDataset,
    PathsDataset,
    common_paths_root,
)
from avalanche.benchmarks.utils.classification_dataset import (
    TaskAwareClassificationDataset,
)
from avalanche.benchmarks.scenarios.deprecated.classification_scenario import (
    GenericCLScenario,
)


def create_multi_dataset_generic_benchmark(
    train_datasets: Sequence[SupportedDataset],
    test_datasets: Sequence[SupportedDataset],
    *,
    other_streams_datasets: Optional[Mapping[str, Sequence[SupportedDataset]]] = None,
    complete_test_set_only: bool = False,
    train_transform=None,
    train_target_transform=None,
    eval_transform=None,
    eval_target_transform=None,
    other_streams_transforms: Optional[Mapping[str, Tuple[Any, Any]]] = None
) -> GenericCLScenario:
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
    :param other_streams_datasets: A dictionary describing the content of
        custom streams. Keys must be valid stream names (letters and numbers,
        not starting with a number) while the value must be a list of dataset.
        If this dictionary contains the definition for "train" or "test"
        streams then those definition will override the `train_datasets` and
        `test_datasets` parameters.
    :param complete_test_set_only: If True, only the complete test set will
        be returned by the benchmark. This means that the ``test_dataset_list``
        parameter must be list with a single element (the complete test set).
        Defaults to False.
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
        transformations for "train" or "test" streams then those
        transformations will override the `train_transform`,
        `train_target_transform`, `eval_transform` and
        `eval_target_transform` parameters.

    :returns: A :class:`GenericCLScenario` instance.
    """

    transform_groups = dict(
        train=(train_transform, train_target_transform),
        eval=(eval_transform, eval_target_transform),
    )

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

    input_streams = dict(train=train_datasets, test=test_datasets)

    if other_streams_datasets is not None:
        input_streams = {**input_streams, **other_streams_datasets}

    if complete_test_set_only:
        if len(input_streams["test"]) != 1:
            raise ValueError(
                "Test stream must contain one experience when"
                "complete_test_set_only is True"
            )

    stream_definitions: Dict[
        str, Tuple[Iterable[TaskAwareClassificationDataset]]
    ] = dict()

    for stream_name, dataset_list in input_streams.items():
        initial_transform_group = "train"
        if stream_name in transform_groups:
            initial_transform_group = stream_name

        stream_datasets = []
        for dataset_idx in range(len(dataset_list)):
            dataset = dataset_list[dataset_idx]
            stream_datasets.append(
                _make_taskaware_classification_dataset(
                    dataset,
                    transform_groups=transform_groups,
                    initial_transform_group=initial_transform_group,
                )
            )
        stream_definitions[stream_name] = (stream_datasets,)

    return GenericCLScenario(
        stream_definitions=stream_definitions,
        complete_test_set_only=complete_test_set_only,
    )


def _adapt_lazy_stream(generator, transform_groups, initial_transform_group):
    """
    A simple internal utility to apply transforms and dataset type to all lazily
    generated datasets. Used in the :func:`create_lazy_generic_benchmark`
    benchmark creation helper.

    :return: A datasets in which the proper transformation groups and dataset
        type are applied.
    """

    for dataset in generator:
        dataset = _make_taskaware_classification_dataset(
            dataset,
            transform_groups=transform_groups,
            initial_transform_group=initial_transform_group,
        )
        yield dataset


class LazyStreamDefinition(NamedTuple):
    """
    A simple class that can be used when preparing the parameters for the
    :func:`create_lazy_generic_benchmark` helper.

    This class is a named tuple containing the fields required for defining
    a lazily-created benchmark.

    - exps_generator: The experiences generator. Can be a "yield"-based
      generator, a custom sequence, a standard list or any kind of
      iterable returning :class:`AvalancheDataset`.
    - stream_length: The number of experiences in the stream. Must match the
      number of experiences returned by the generator.
    - exps_task_labels: A list containing the list of task labels of each
      experience. If an experience contains a single task label, a single int
      can be used.
    """

    exps_generator: Iterable[TaskAwareClassificationDataset]
    """
    The experiences generator. Can be a "yield"-based generator, a custom
    sequence, a standard list or any kind of iterable returning
    :class:`AvalancheDataset`.
    """

    stream_length: int
    """
    The number of experiences in the stream. Must match the number of
    experiences returned by the generator
    """

    exps_task_labels: Sequence[Union[int, Iterable[int]]]
    """
    A list containing the list of task labels of each experience.
    If an experience contains a single task label, a single int can be used.
    
    This field is temporary required for internal purposes to support lazy
    streams. This field may become optional in the future.
    """


def create_lazy_generic_benchmark(
    train_generator: LazyStreamDefinition,
    test_generator: LazyStreamDefinition,
    *,
    other_streams_generators: Optional[Dict[str, LazyStreamDefinition]] = None,
    complete_test_set_only: bool = False,
    train_transform=None,
    train_target_transform=None,
    eval_transform=None,
    eval_target_transform=None,
    other_streams_transforms: Optional[Dict[str, Tuple[Any, Any]]] = None
) -> GenericCLScenario:
    """
    Creates a lazily-defined benchmark instance given a dataset generator for
    each stream.

    Generators must return properly initialized instances of
    :class:`AvalancheDataset` which will be used to create experiences.

    The created datasets can have transformations already set.
    However, if transformations are shared across all datasets of the same
    stream, it is recommended to use the `train_transform`, `eval_transform`
    and `other_streams_transforms` parameters, so that transformations groups
    can be correctly applied (transformations are lazily added atop the datasets
    returned by the generators).

    This function allows for the creation of custom streams as well.
    While "train" and "test" streams must be always set, the generators
    for other streams can be defined by using the `other_streams_generators`
    parameter.

    :param train_generator: A proper lazy-generation definition for the training
        stream. It is recommended to pass an instance
        of :class:`LazyStreamDefinition`. See its description for more details.
    :param test_generator: A proper lazy-generation definition for the test
        stream. It is recommended to pass an instance
        of :class:`LazyStreamDefinition`. See its description for more details.
    :param other_streams_generators: A dictionary describing the content of
        custom streams. Keys must be valid stream names (letters and numbers,
        not starting with a number) while the value must be a
        lazy-generation definition (like the ones of the training and
        test streams). If this dictionary contains the definition for
        "train" or "test" streams then those definition will override the
        `train_generator` and `test_generator` parameters.
    :param complete_test_set_only: If True, only the complete test set will
        be returned by the benchmark. This means that the ``test_generator``
        parameter must define a stream with a single experience (the complete
        test set). Defaults to False.
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

    :returns: A lazily-initialized :class:`GenericCLScenario` instance.
    """

    transform_groups = dict(
        train=(train_transform, train_target_transform),
        eval=(eval_transform, eval_target_transform),
    )

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

    input_streams = dict(train=train_generator, test=test_generator)

    if other_streams_generators is not None:
        input_streams = {**input_streams, **other_streams_generators}

    if complete_test_set_only:
        if input_streams["test"][1] != 1:
            raise ValueError(
                "Test stream must contain one experience when"
                "complete_test_set_only is True"
            )

    stream_definitions: Dict[
        str,
        Tuple[
            # Dataset generator + stream length
            Tuple[Generator[TaskAwareClassificationDataset, None, None], int],
            # Task label(s) for each experience
            Iterable[Union[int, Iterable[int]]],
        ],
    ] = dict()

    for stream_name, (
        generator,
        stream_length,
        task_labels,
    ) in input_streams.items():
        initial_transform_group = "train"
        if stream_name in transform_groups:
            initial_transform_group = stream_name

        adapted_stream_generator = _adapt_lazy_stream(
            generator,
            transform_groups,
            initial_transform_group=initial_transform_group,
        )

        stream_definitions[stream_name] = (
            (adapted_stream_generator, stream_length),
            task_labels,
        )

    return GenericCLScenario(
        stream_definitions=stream_definitions,
        complete_test_set_only=complete_test_set_only,
    )


def create_generic_benchmark_from_filelists(
    root: Optional[Union[str, Path]],
    train_file_lists: Sequence[Union[str, Path]],
    test_file_lists: Sequence[Union[str, Path]],
    *,
    other_streams_file_lists: Optional[Dict[str, Sequence[Union[str, Path]]]] = None,
    task_labels: Sequence[int],
    complete_test_set_only: bool = False,
    train_transform=None,
    train_target_transform=None,
    eval_transform=None,
    eval_target_transform=None,
    other_streams_transforms: Optional[Dict[str, Tuple[Any, Any]]] = None
) -> GenericCLScenario:
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
        be returned by the benchmark. This means that the ``test_file_lists``
        parameter must be list with a single element (the complete test set).
        Alternatively, can be a plain string or :class:`Path` object.
        Defaults to False.
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

    input_streams = dict(train=train_file_lists, test=test_file_lists)

    if other_streams_file_lists is not None:
        input_streams = {**input_streams, **other_streams_file_lists}

    stream_definitions: Dict[str, Sequence[TaskAwareClassificationDataset]] = dict()

    for stream_name, file_lists in input_streams.items():
        stream_datasets: List[TaskAwareClassificationDataset] = []
        for exp_id, f_list in enumerate(file_lists):
            f_list_dataset = FilelistDataset(root, f_list)
            stream_datasets.append(
                _make_taskaware_classification_dataset(
                    f_list_dataset, task_labels=task_labels[exp_id]
                )
            )

        stream_definitions[stream_name] = stream_datasets

    return create_multi_dataset_generic_benchmark(
        [],
        [],
        other_streams_datasets=stream_definitions,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        eval_transform=eval_transform,
        eval_target_transform=eval_target_transform,
        complete_test_set_only=complete_test_set_only,
        other_streams_transforms=other_streams_transforms,
    )


FileAndLabel = Union[
    Tuple[Union[str, Path], int], Tuple[Union[str, Path], int, Sequence]
]


def create_generic_benchmark_from_paths(
    train_lists_of_files: Sequence[Sequence[FileAndLabel]],
    test_lists_of_files: Sequence[Sequence[FileAndLabel]],
    *,
    other_streams_lists_of_files: Optional[
        Dict[str, Sequence[Sequence[FileAndLabel]]]
    ] = None,
    task_labels: Sequence[int],
    complete_test_set_only: bool = False,
    train_transform=None,
    train_target_transform=None,
    eval_transform=None,
    eval_target_transform=None,
    other_streams_transforms: Optional[Dict[str, Tuple[Any, Any]]] = None
) -> GenericCLScenario:
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
        be returned by the benchmark. This means that the ``test_list_of_files``
        parameter must define a single experience (the complete test set).
        Defaults to False.
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

    input_streams = dict(train=train_lists_of_files, test=test_lists_of_files)

    if other_streams_lists_of_files is not None:
        input_streams = {**input_streams, **other_streams_lists_of_files}

    stream_definitions: Dict[str, Sequence[TaskAwareClassificationDataset]] = dict()

    for stream_name, lists_of_files in input_streams.items():
        stream_datasets: List[TaskAwareClassificationDataset] = []
        for exp_id, list_of_files in enumerate(lists_of_files):
            common_root, exp_paths_list = common_paths_root(list_of_files)
            paths_dataset: PathsDataset[Any, int] = PathsDataset(
                common_root, exp_paths_list
            )
            stream_datasets.append(
                _make_taskaware_classification_dataset(
                    paths_dataset, task_labels=task_labels[exp_id]
                )
            )

        stream_definitions[stream_name] = stream_datasets

    return create_multi_dataset_generic_benchmark(
        [],
        [],
        other_streams_datasets=stream_definitions,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        eval_transform=eval_transform,
        eval_target_transform=eval_target_transform,
        complete_test_set_only=complete_test_set_only,
        other_streams_transforms=other_streams_transforms,
    )


def create_generic_benchmark_from_tensor_lists(
    train_tensors: Sequence[Sequence[Any]],
    test_tensors: Sequence[Sequence[Any]],
    *,
    other_streams_tensors: Optional[Dict[str, Sequence[Sequence[Any]]]] = None,
    task_labels: Sequence[int],
    complete_test_set_only: bool = False,
    train_transform=None,
    train_target_transform=None,
    eval_transform=None,
    eval_target_transform=None,
    other_streams_transforms: Optional[Dict[str, Tuple[Any, Any]]] = None
) -> GenericCLScenario:
    """
    Creates a benchmark instance given lists of Tensors. A separate dataset will
    be created from each Tensor tuple (x, y, z, ...) and each of those training
    datasets will be considered a separate training experience. Using this
    helper function is the lowest-level way to create a Continual Learning
    benchmark. When possible, consider using higher level helpers.

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
        be returned by the benchmark. This means that ``test_tensors`` must
        define a single experience. Defaults to False.
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

    input_streams = dict(train=train_tensors, test=test_tensors)

    if other_streams_tensors is not None:
        input_streams = {**input_streams, **other_streams_tensors}

    stream_definitions: Dict[str, Sequence[TaskAwareClassificationDataset]] = dict()

    for stream_name, list_of_exps_tensors in input_streams.items():
        stream_datasets: List[TaskAwareClassificationDataset] = []
        for exp_id, exp_tensors in enumerate(list_of_exps_tensors):
            stream_datasets.append(
                _make_taskaware_tensor_classification_dataset(
                    *exp_tensors, task_labels=task_labels[exp_id]
                )
            )

        stream_definitions[stream_name] = stream_datasets

    return create_multi_dataset_generic_benchmark(
        [],
        [],
        other_streams_datasets=stream_definitions,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        eval_transform=eval_transform,
        eval_target_transform=eval_target_transform,
        complete_test_set_only=complete_test_set_only,
        other_streams_transforms=other_streams_transforms,
    )


__all__ = [
    "create_multi_dataset_generic_benchmark",
    "LazyStreamDefinition",
    "create_lazy_generic_benchmark",
    "create_generic_benchmark_from_filelists",
    "create_generic_benchmark_from_paths",
    "create_generic_benchmark_from_tensor_lists",
]
