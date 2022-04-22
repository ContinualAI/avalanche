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
from functools import partial
from itertools import tee
from typing import (Any, Callable, Dict, Generator, Iterable, List, Optional,
                    Sequence, Set, Tuple, Union)

import torch
from avalanche.benchmarks import (ClassificationExperience,
                                  ClassificationStream, GenericCLScenario)
from avalanche.benchmarks.scenarios.classification_scenario import (
    StreamUserDef, TStreamsUserDict)
from avalanche.benchmarks.scenarios.generic_benchmark_creation import *
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCScenario
from avalanche.benchmarks.scenarios.new_instances.ni_scenario import NIScenario
from avalanche.benchmarks.utils import concat_datasets_sequentially
from avalanche.benchmarks.utils.avalanche_dataset import (
    AvalancheConcatDataset, AvalancheDataset, AvalancheDatasetType,
    AvalancheSubset, SupportedDataset)


def nc_benchmark(
    train_dataset: Union[Sequence[SupportedDataset], SupportedDataset],
    test_dataset: Union[Sequence[SupportedDataset], SupportedDataset],
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
    reproducibility_data: Dict[str, Any] = None,
) -> NCScenario:
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
    the behaviour of the resulting benchmark. Please take a few minutes to read
    and understand them as they may save you a lot of work.

    This generator features a integrated reproducibility mechanism that allows
    the user to store and later re-load a benchmark. For more info see the
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
        benchmark definition options. This is usually a dictionary containing
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
        raise ValueError(
            "Invalid mutually exclusive options "
            "class_ids_from_zero_from_first_exp and "
            "classes_ids_from_zero_in_each_exp set at the "
            "same time"
        )

    if isinstance(train_dataset, list) or isinstance(train_dataset, tuple):
        # Multi-dataset setting

        if len(train_dataset) != len(test_dataset):
            raise ValueError(
                "Train/test dataset lists must contain the "
                "exact same number of datasets"
            )

        if per_exp_classes and one_dataset_per_exp:
            raise ValueError(
                "Both per_experience_classes and one_dataset_per_exp are"
                "used, but those options are mutually exclusive"
            )

        if fixed_class_order and one_dataset_per_exp:
            raise ValueError(
                "Both fixed_class_order and one_dataset_per_exp are"
                "used, but those options are mutually exclusive"
            )

        (
            seq_train_dataset,
            seq_test_dataset,
            mapping,
        ) = concat_datasets_sequentially(train_dataset, test_dataset)

        if one_dataset_per_exp:
            # If one_dataset_per_exp is True, each dataset will be treated as
            # a experience. In this benchmark, shuffle refers to the experience
            # order, not to the class one.
            (
                fixed_class_order,
                per_exp_classes,
            ) = _one_dataset_per_exp_class_order(mapping, shuffle, seed)

            # We pass a fixed_class_order to the NCGenericScenario
            # constructor, so we don't need shuffling.
            shuffle = False
            seed = None

            # Overrides n_experiences (and per_experience_classes, already done)
            n_experiences = len(train_dataset)
        train_dataset, test_dataset = seq_train_dataset, seq_test_dataset

    transform_groups = dict(
        train=(train_transform, None), eval=(eval_transform, None)
    )

    # Datasets should be instances of AvalancheDataset
    train_dataset = AvalancheDataset(
        train_dataset,
        transform_groups=transform_groups,
        initial_transform_group="train",
        dataset_type=AvalancheDatasetType.CLASSIFICATION,
    )

    test_dataset = AvalancheDataset(
        test_dataset,
        transform_groups=transform_groups,
        initial_transform_group="eval",
        dataset_type=AvalancheDatasetType.CLASSIFICATION,
    )

    return NCScenario(
        train_dataset,
        test_dataset,
        n_experiences,
        task_labels,
        shuffle,
        seed,
        fixed_class_order,
        per_exp_classes,
        class_ids_from_zero_from_first_exp,
        class_ids_from_zero_in_each_exp,
        reproducibility_data,
    )


def ni_benchmark(
    train_dataset: Union[Sequence[SupportedDataset], SupportedDataset],
    test_dataset: Union[Sequence[SupportedDataset], SupportedDataset],
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
    reproducibility_data: Optional[Dict[str, Any]] = None,
) -> NIScenario:
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
    the behaviour of the resulting benchmark. Please take a few minutes to read
    and understand them as they may save you a lot of work.

    This generator features an integrated reproducibility mechanism that allows
    the user to store and later re-load a benchmark. For more info see the
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
        benchmark definition options, including ``fixed_exp_assignment``.
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
            raise ValueError(
                "Train/test dataset lists must contain the "
                "exact same number of datasets"
            )

        seq_train_dataset, seq_test_dataset, _ = concat_datasets_sequentially(
            train_dataset, test_dataset
        )

    transform_groups = dict(
        train=(train_transform, None), eval=(eval_transform, None)
    )

    # Datasets should be instances of AvalancheDataset
    seq_train_dataset = AvalancheDataset(
        seq_train_dataset,
        transform_groups=transform_groups,
        initial_transform_group="train",
        dataset_type=AvalancheDatasetType.CLASSIFICATION,
    )

    seq_test_dataset = AvalancheDataset(
        seq_test_dataset,
        transform_groups=transform_groups,
        initial_transform_group="eval",
        dataset_type=AvalancheDatasetType.CLASSIFICATION,
    )

    return NIScenario(
        seq_train_dataset,
        seq_test_dataset,
        n_experiences,
        task_labels,
        shuffle=shuffle,
        seed=seed,
        balance_experiences=balance_experiences,
        min_class_patterns_in_exp=min_class_patterns_in_exp,
        fixed_exp_assignment=fixed_exp_assignment,
        reproducibility_data=reproducibility_data,
    )


# Here we define some high-level APIs an alias of their mid-level counterparts.
# This was done mainly because the implementation for the mid-level API is now
# quite stable and not particularly complex.
dataset_benchmark = create_multi_dataset_generic_benchmark
filelist_benchmark = create_generic_benchmark_from_filelists
paths_benchmark = create_generic_benchmark_from_paths
tensors_benchmark = create_generic_benchmark_from_tensor_lists
lazy_benchmark = create_lazy_generic_benchmark


def _one_dataset_per_exp_class_order(
    class_list_per_exp: Sequence[Sequence[int]],
    shuffle: bool,
    seed: Union[int, None],
) -> (List[int], Dict[int, int]):
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
            torch.randperm(len(dataset_order))
        ].tolist()
    fixed_class_order = []
    classes_per_exp = {}
    for dataset_position, dataset_idx in enumerate(dataset_order):
        fixed_class_order.extend(class_list_per_exp[dataset_idx])
        classes_per_exp[dataset_position] = len(class_list_per_exp[dataset_idx])
    return fixed_class_order, classes_per_exp


def fixed_size_experience_split_strategy(
    experience_size: int,
    shuffle: bool,
    drop_last: bool,
    experience: ClassificationExperience,
):
    """
    The default splitting strategy used by :func:`data_incremental_benchmark`.

    This splitting strategy simply splits the experience in smaller experiences
    of size `experience_size`.

    When taking inspiration for your custom splitting strategy, please consider
    that all parameters preceding `experience` are filled by
    :func:`data_incremental_benchmark` by using `partial` from the `functools`
    standard library. A custom splitting strategy must have only a single
    parameter: the experience. Consider wrapping your custom splitting strategy
    with `partial` if more parameters are needed.

    Also consider that the stream name of the experience can be obtained by
    using `experience.origin_stream.name`.

    :param experience_size: The experience size (number of instances).
    :param shuffle: If True, instances will be shuffled before splitting.
    :param drop_last: If True, the last mini-experience will be dropped if
        not of size `experience_size`
    :param experience: The experience to split.
    :return: The list of datasets that will be used to create the
        mini-experiences.
    """

    exp_dataset = experience.dataset
    exp_indices = list(range(len(exp_dataset)))

    result_datasets = []

    if shuffle:
        exp_indices = torch.as_tensor(exp_indices)[
            torch.randperm(len(exp_indices))
        ].tolist()

    init_idx = 0
    while init_idx < len(exp_indices):
        final_idx = init_idx + experience_size  # Exclusive
        if final_idx > len(exp_indices):
            if drop_last:
                break

            final_idx = len(exp_indices)

        result_datasets.append(
            AvalancheSubset(
                exp_dataset, indices=exp_indices[init_idx:final_idx]
            )
        )
        init_idx = final_idx

    return result_datasets


def data_incremental_benchmark(
    benchmark_instance: GenericCLScenario,
    experience_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    split_streams: Sequence[str] = ("train",),
    custom_split_strategy: Callable[
        [ClassificationExperience], Sequence[AvalancheDataset]
    ] = None,
    experience_factory: Callable[
        [ClassificationStream, int], ClassificationExperience
    ] = None,
):
    """
    High-level benchmark generator for a Data Incremental setup.

    This generator accepts an existing benchmark instance and returns a version
    of it in which experiences have been split in order to produce a
    Data Incremental stream.

    In its base form this generator will split train experiences in experiences
    of a fixed, configurable, size. The split can be also performed on other
    streams (like the test one) if needed.

    The `custom_split_strategy` parameter can be used if a more specific
    splitting is required.

    Beware that experience splitting is NOT executed in a lazy way. This
    means that the splitting process takes place immediately. Consider
    optimizing the split process for speed when using a custom splitting
    strategy.

    Please note that each mini-experience will have a task labels field
    equal to the one of the originating experience.

    The `complete_test_set_only` field of the resulting benchmark instance
    will be `True` only if the same field of original benchmark instance is
    `True` and if the resulting test stream contains exactly one experience.

    :param benchmark_instance: The benchmark to split.
    :param experience_size: The size of the experience, as an int. Ignored
        if `custom_split_strategy` is used.
    :param shuffle: If True, experiences will be split by first shuffling
        instances in each experience. This will use the default PyTorch
        random number generator at its current state. Defaults to False.
        Ignored if `custom_split_strategy` is used.
    :param drop_last: If True, if the last experience doesn't contain
        `experience_size` instances, then the last experience will be dropped.
        Defaults to False. Ignored if `custom_split_strategy` is used.
    :param split_streams: The list of streams to split. By default only the
        "train" stream will be split.
    :param custom_split_strategy: A function that implements a custom splitting
        strategy. The function must accept an experience and return a list
        of datasets each describing an experience. Defaults to None, which means
        that the standard splitting strategy will be used (which creates
        experiences of size `experience_size`).
        A good starting to understand the mechanism is to look at the
        implementation of the standard splitting function
        :func:`fixed_size_experience_split_strategy`.

    :param experience_factory: The experience factory.
        Defaults to :class:`GenericExperience`.
    :return: The Data Incremental benchmark instance.
    """

    split_strategy = custom_split_strategy
    if split_strategy is None:
        split_strategy = partial(
            fixed_size_experience_split_strategy,
            experience_size,
            shuffle,
            drop_last,
        )

    stream_definitions: TStreamsUserDict = dict(
        benchmark_instance.stream_definitions
    )

    for stream_name in split_streams:
        if stream_name not in stream_definitions:
            raise ValueError(
                f"Stream {stream_name} could not be found in the "
                f"benchmark instance"
            )

        stream = getattr(benchmark_instance, f"{stream_name}_stream")

        split_datasets: List[AvalancheDataset] = []
        split_task_labels: List[Set[int]] = []

        exp: ClassificationExperience
        for exp in stream:
            experiences = split_strategy(exp)
            split_datasets += experiences
            for _ in range(len(experiences)):
                split_task_labels.append(set(exp.task_labels))

        stream_def = StreamUserDef(
            split_datasets,
            split_task_labels,
            stream_definitions[stream_name].origin_dataset,
            False,
        )

        stream_definitions[stream_name] = stream_def

    complete_test_set_only = (
        benchmark_instance.complete_test_set_only
        and len(stream_definitions["test"].exps_data) == 1
    )

    return GenericCLScenario(
        stream_definitions=stream_definitions,
        complete_test_set_only=complete_test_set_only,
        experience_factory=experience_factory,
    )


def random_validation_split_strategy(
    validation_size: Union[int, float],
    shuffle: bool,
    experience: ClassificationExperience,
):
    """
    The default splitting strategy used by
    :func:`benchmark_with_validation_stream`.

    This splitting strategy simply splits the experience in two experiences (
    train and validation) of size `validation_size`.

    When taking inspiration for your custom splitting strategy, please consider
    that all parameters preceding `experience` are filled by
    :func:`benchmark_with_validation_stream` by using `partial` from the
    `functools` standard library. A custom splitting strategy must have only
    a single parameter: the experience. Consider wrapping your custom
    splitting strategy with `partial` if more parameters are needed.

    Also consider that the stream name of the experience can be obtained by
    using `experience.origin_stream.name`.

    :param validation_size: The number of instances to allocate to the
    validation experience. Can be an int value or a float between 0 and 1.
    :param shuffle: If True, instances will be shuffled before splitting.
        Otherwise, the first instances will be allocated to the training
        dataset by leaving the last ones to the validation dataset.
    :param experience: The experience to split.
    :return: A tuple containing 2 elements: the new training and validation
        datasets.
    """

    exp_dataset = experience.dataset
    exp_indices = list(range(len(exp_dataset)))

    if shuffle:
        exp_indices = torch.as_tensor(exp_indices)[
            torch.randperm(len(exp_indices))
        ].tolist()

    if 0.0 <= validation_size <= 1.0:
        valid_n_instances = int(validation_size * len(exp_dataset))
    else:
        valid_n_instances = int(validation_size)
        if valid_n_instances > len(exp_dataset):
            raise ValueError(
                f"Can't create the validation experience: not enough "
                f"instances. Required {valid_n_instances}, got only"
                f"{len(exp_dataset)}"
            )

    train_n_instances = len(exp_dataset) - valid_n_instances

    result_train_dataset = AvalancheSubset(
        exp_dataset, indices=exp_indices[:train_n_instances]
    )
    result_valid_dataset = AvalancheSubset(
        exp_dataset, indices=exp_indices[train_n_instances:]
    )

    return result_train_dataset, result_valid_dataset


def class_balanced_split_strategy(
    validation_size: Union[int, float], experience: ClassificationExperience
):
    """Class-balanced train/validation splits.

    This splitting strategy splits `experience` into two experiences
    (train and validation) of size `validation_size` using a class-balanced
    split. Sample of each class are chosen randomly.

    :param validation_size: The percentage of samples to allocate to the
        validation experience as a float between 0 and 1.
    :param experience: The experience to split.
    :return: A tuple containing 2 elements: the new training and validation
        datasets.
    """
    if not isinstance(validation_size, float):
        raise ValueError("validation_size must be an integer")
    if not 0.0 <= validation_size <= 1.0:
        raise ValueError("validation_size must be a float in [0, 1].")

    exp_dataset = experience.dataset
    if validation_size > len(exp_dataset):
        raise ValueError(
            f"Can't create the validation experience: not enough "
            f"instances. Required {validation_size}, got only"
            f"{len(exp_dataset)}"
        )

    exp_indices = list(range(len(exp_dataset)))
    exp_classes = experience.classes_in_this_experience

    # shuffle exp_indices
    exp_indices = torch.as_tensor(exp_indices)[torch.randperm(len(exp_indices))]
    # shuffle the targets as well
    exp_targets = torch.as_tensor(experience.dataset.targets)[exp_indices]

    train_exp_indices = []
    valid_exp_indices = []
    for cid in exp_classes:  # split indices for each class separately.
        c_indices = exp_indices[exp_targets == cid]
        valid_n_instances = int(validation_size * len(c_indices))
        valid_exp_indices.extend(c_indices[:valid_n_instances])
        train_exp_indices.extend(c_indices[valid_n_instances:])

    result_train_dataset = AvalancheSubset(
        exp_dataset, indices=train_exp_indices
    )
    result_valid_dataset = AvalancheSubset(
        exp_dataset, indices=valid_exp_indices
    )
    return result_train_dataset, result_valid_dataset


def _gen_split(
    split_generator: Iterable[Tuple[AvalancheDataset, AvalancheDataset]]
) -> Tuple[
    Generator[AvalancheDataset, None, None],
    Generator[AvalancheDataset, None, None],
]:
    """
    Internal utility function to split the train-validation generator
    into two distinct generators (one for the train stream and another one
    for the valid stream).

    :param split_generator: The lazy stream generator returning tuples of train
        and valid datasets.
    :return: Two generators (one for the train, one for the valuid).
    """

    # For more info: https://stackoverflow.com/a/28030261
    gen_a, gen_b = tee(split_generator, 2)
    return (a for a, b in gen_a), (b for a, b in gen_b)


def _lazy_train_val_split(
    split_strategy: Callable[
        [ClassificationExperience], Tuple[AvalancheDataset, AvalancheDataset]
    ],
    experiences: Iterable[ClassificationExperience],
) -> Generator[Tuple[AvalancheDataset, AvalancheDataset], None, None]:
    """
    Creates a generator operating around the split strategy and the
    experiences stream.

    :param split_strategy: The strategy used to split each experience in train
        and validation datasets.
    :return: A generator returning a 2 elements tuple (the train and validation
        datasets).
    """

    for new_experience in experiences:
        yield split_strategy(new_experience)


def benchmark_with_validation_stream(
    benchmark_instance: GenericCLScenario,
    validation_size: Union[int, float] = 0.5,
    shuffle: bool = False,
    input_stream: str = "train",
    output_stream: str = "valid",
    custom_split_strategy: Callable[
        [ClassificationExperience], Tuple[AvalancheDataset, AvalancheDataset]
    ] = None,
    *,
    experience_factory: Callable[
        [ClassificationStream, int], ClassificationExperience
    ] = None,
    lazy_splitting: bool = None,
):
    """
    Helper that can be used to obtain a benchmark with a validation stream.

    This generator accepts an existing benchmark instance and returns a version
    of it in which a validation stream has been added.

    In its base form this generator will split train experiences to extract
    validation experiences of a fixed (by number of instances or relative
    size), configurable, size. The split can be also performed on other
    streams if needed and the name of the resulting validation stream can
    be configured too.

    Each validation experience will be extracted directly from a single training
    experience. Patterns selected for the validation experience will be removed
    from the training one.

    If shuffle is True, the validation stream will be created randomly.
    Beware that no kind of class balancing is done.

    The `custom_split_strategy` parameter can be used if a more specific
    splitting is required.

    Please note that the resulting experiences will have a task labels field
    equal to the one of the originating experience.

    Experience splitting can be executed in a lazy way. This behavior can be
    controlled using the `lazy_splitting` parameter. By default, experiences
    are split in a lazy way only when the input stream is lazily generated.

    :param benchmark_instance: The benchmark to split.
    :param validation_size: The size of the validation experience, as an int
        or a float between 0 and 1. Ignored if `custom_split_strategy` is used.
    :param shuffle: If True, patterns will be allocated to the validation
        stream randomly. This will use the default PyTorch random number
        generator at its current state. Defaults to False. Ignored if
        `custom_split_strategy` is used. If False, the first instances will be
        allocated to the training  dataset by leaving the last ones to the
        validation dataset.
    :param input_stream: The name of the input stream. Defaults to 'train'.
    :param output_stream: The name of the output stream. Defaults to 'valid'.
    :param custom_split_strategy: A function that implements a custom splitting
        strategy. The function must accept an experience and return a tuple
        containing the new train and validation dataset. Defaults to None,
        which means that the standard splitting strategy will be used (which
        creates experiences according to `validation_size` and `shuffle`).
        A good starting to understand the mechanism is to look at the
        implementation of the standard splitting function
        :func:`random_validation_split_strategy`.
    :param experience_factory: The experience factory. Defaults to
        :class:`GenericExperience`.
    :param lazy_splitting: If True, the stream will be split in a lazy way.
        If False, the stream will be split immediately. Defaults to None, which
        means that the stream will be split in a lazy or non-lazy way depending
        on the laziness of the `input_stream`.
    :return: A benchmark instance in which the validation stream has been added.
    """

    split_strategy = custom_split_strategy
    if split_strategy is None:
        split_strategy = partial(
            random_validation_split_strategy, validation_size, shuffle
        )

    stream_definitions: TStreamsUserDict = dict(
        benchmark_instance.stream_definitions
    )
    streams = benchmark_instance.streams

    if input_stream not in streams:
        raise ValueError(
            f"Stream {input_stream} could not be found in the "
            f"benchmark instance"
        )

    if output_stream in streams:
        raise ValueError(
            f"Stream {output_stream} already exists in the "
            f"benchmark instance"
        )

    stream = streams[input_stream]

    split_lazily = lazy_splitting
    if split_lazily is None:
        split_lazily = stream_definitions[input_stream].is_lazy

    exps_tasks_labels = list(stream_definitions[input_stream].exps_task_labels)

    if not split_lazily:
        # Classic static splitting
        train_exps_source = []
        valid_exps_source = []

        exp: ClassificationExperience
        for exp in stream:
            train_exp, valid_exp = split_strategy(exp)
            train_exps_source.append(train_exp)
            valid_exps_source.append(valid_exp)
    else:
        # Lazy splitting (based on a generator)
        split_generator = _lazy_train_val_split(split_strategy, stream)
        train_exps_gen, valid_exps_gen = _gen_split(split_generator)
        train_exps_source = (train_exps_gen, len(stream))
        valid_exps_source = (valid_exps_gen, len(stream))

    train_stream_def = StreamUserDef(
        train_exps_source,
        exps_tasks_labels,
        stream_definitions[input_stream].origin_dataset,
        split_lazily,
    )

    valid_stream_def = StreamUserDef(
        valid_exps_source,
        exps_tasks_labels,
        stream_definitions[input_stream].origin_dataset,
        split_lazily,
    )

    stream_definitions[input_stream] = train_stream_def
    stream_definitions[output_stream] = valid_stream_def

    complete_test_set_only = benchmark_instance.complete_test_set_only

    return GenericCLScenario(
        stream_definitions=stream_definitions,
        complete_test_set_only=complete_test_set_only,
        experience_factory=experience_factory,
    )


def gen_joint_training_benchmark(
    benchmark, 
    experience_factory: Callable[
        [ClassificationStream, int], ClassificationExperience
    ] = None,
):
    """
    Helper that can be used to obtain a benchmark with just one experience for 
    each stream.

    This generator accepts an existing benchmark instance and returns a version
    of it in which the data set off all experiences are concatenated in a 
    single experience for each stream. 
    
    It only works on not lazy experiences.

    :param benchmark:
    :param experience_factory: The experience factory. Defaults to
        :class:`GenericExperience`.
    :return: A benchmark instance with just one experience in all streams. 
    These streams contains the data of all data sets received in incoming 
    benchmark.
    """
    final_stream_definitions = dict()
    for stream_name in benchmark.streams.keys():
        stream = benchmark.streams[stream_name]
        stream_definitions = benchmark.stream_definitions[stream_name]
        complete_test_set_only = benchmark.complete_test_set_only
        is_lazy = stream_definitions.is_lazy

        exps_tasks_labels = list(stream_definitions.exps_task_labels)[0]

        if not is_lazy:
            exps_source = []

            adapted_dataset = stream[0].dataset
            for exp in stream[1:]:
                cat_data = AvalancheConcatDataset(
                    [adapted_dataset, exp.dataset]
                )
                adapted_dataset = cat_data

            data_exp = AvalancheDataset(
                adapted_dataset, 
                transform_groups=adapted_dataset.transform_groups, 
                task_labels=0,
                targets=adapted_dataset.targets, 
                dataset_type=adapted_dataset.dataset_type,
            )
            exps_source.append(data_exp)
        else:
            raise RuntimeError('Lazy experiences is not supported yet.')

        stream_def = \
            StreamUserDef(
                exps_source,
                exps_tasks_labels,
                stream_definitions.origin_dataset,
                is_lazy)

        final_stream_definitions[stream_name] = stream_def
    
    scenario = GenericCLScenario(
        stream_definitions=final_stream_definitions,
        complete_test_set_only=complete_test_set_only,
        experience_factory=experience_factory,
    )
    return scenario


__all__ = [
    "nc_benchmark",
    "ni_benchmark",
    "dataset_benchmark",
    "filelist_benchmark",
    "paths_benchmark",
    "tensors_benchmark",
    "data_incremental_benchmark",
    "benchmark_with_validation_stream",
    "gen_joint_training_benchmark",
]
