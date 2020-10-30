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

from typing import Sequence, Dict, List, Union, Any

import torch

from avalanche.training.utils.transform_dataset import IDatasetWithTargets, \
    concat_datasets_sequentially
from .nc_benchmark import NCScenario


def create_nc_single_dataset_scenario(
        train_dataset: IDatasetWithTargets,
        test_dataset: IDatasetWithTargets,
        n_steps: int,
        task_labels: bool,
        shuffle: bool = True,
        seed: int = None,
        fixed_class_order: Sequence[int] = None,
        per_step_classes: Dict[int, int] = None,
        class_ids_from_zero_from_first_step: bool = False,
        class_ids_from_zero_in_each_step: bool = False,
        reproducibility_data: Dict[str, Any] = None):
    """
    Creates a "New Classes" scenario given a train and test dataset pair.

    :param train_dataset: The training dataset.
    :param test_dataset: The test dataset.
    :param n_steps: The number of steps.
    :param task_labels: If True, each step will have an ascending task
            label. If False, the task label will be 0 for all the steps.
    :param shuffle: If True, the class order will be shuffled. Defaults to
        True.
    :param seed: If shuffle is True and seed is not None, the class order
        will be shuffled according to the seed. When None, the current
        PyTorch random number generator state will be used.
        Defaults to None.
    :param fixed_class_order: If not None, the class order to use (overrides
        the shuffle argument). Very useful for enhancing
        reproducibility. Defaults to None.
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
        Mutually exclusive with the class_ids_from_zero_in_each_step
        parameter.
    :param class_ids_from_zero_in_each_step: If True, original class IDs
        will be mapped to range [0, n_classes_in_step) for each step.
        Defaults to False. Mutually exclusive with the
        class_ids_from_zero_from_first_step parameter.
    :param reproducibility_data: If not None, overrides all the other
        scenario definition options. This is usually a dictionary containing
        data used to reproduce a specific experiment. One can use the
        ``get_reproducibility_data`` method to get (and even distribute)
        the experiment setup so that it can be loaded by passing it as this
        parameter. In this way one can be sure that the same specific
        experimental setup is being used (for reproducibility purposes).
        Beware that, in order to reproduce an experiment, the same train and
        test datasets must be used. Defaults to None.

    :returns: A :class:`NCScenario` instance.
    """
    if class_ids_from_zero_from_first_step and class_ids_from_zero_in_each_step:
        raise ValueError('Invalid mutually exclusive options '
                         'class_ids_from_zero_from_first_step and '
                         'classes_ids_from_zero_in_each_step set at the '
                         'same time')

    return NCScenario(train_dataset, test_dataset, n_steps, task_labels,
                      shuffle, seed, fixed_class_order, per_step_classes,
                      class_ids_from_zero_from_first_step,
                      class_ids_from_zero_in_each_step,
                      reproducibility_data)


def create_nc_multi_dataset_scenario(
        train_dataset_list: Sequence[IDatasetWithTargets],
        test_dataset_list: Sequence[IDatasetWithTargets],
        n_steps: int,
        task_labels: bool,
        shuffle: bool = True,
        seed: int = None,
        fixed_class_order: Sequence[int] = None,
        per_step_classes: Dict[int, int] = None,
        class_ids_from_zero_from_first_step: bool = False,
        class_ids_from_zero_in_each_step: bool = False,
        one_dataset_per_step: bool = False,
        reproducibility_data: Dict[str, Any] = None):
    """
    Creates a "New Classes" scenario given a list of train and test datasets.

    Please note that ``train_dataset_list`` and ``test_dataset_list`` must
    contain the same amount of datasets (train and test datasets are coupled
    together).

    This function can create a benchmark:
    - By mixing classes from all the datasets
    - By creating a step from each train/test dataset pair
    depending on the value of the ``one_dataset_per_step`` parameter.

    In the first case, datasets will be concatenated in order to obtain a single
    dataset by shifting class IDs of datasets (from the second one onwards).
    This means that ``fixed_class_order`` and ``per_step_classes`` can
    be used to control how class are assigned to each step.
    If ``shuffle`` is True, steps will be created by shuffling all classes.

    In the second case, datasets will be kept as is by creating a step out
    of each train/test dataset couple. This is mutually exclusive with the
    ``fixed_class_order`` and ``per_step_classes`` parameters. If ``shuffle`` is
    True, steps will be shuffled.

    :param train_dataset_list: A list of training datasets.
    :param test_dataset_list: A list of test datasets.
    :param n_steps: The number of steps.
    :param task_labels: If True, each step will have an ascending task
            label. If False, the task label will be 0 for all the steps.
    :param shuffle: If True, the class (or step) order will be shuffled.
        Defaults to True.
    :param seed: If ``shuffle`` is True and seed is not None, the class (or
        step) order will be shuffled according to the seed. When None, the
        current PyTorch random number generator state will be used. Defaults to
        None.
    :param fixed_class_order: If not None, the class order to use (overrides
        the shuffle argument). Very useful for enhancing
        reproducibility. Defaults to None.
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
    :param one_dataset_per_step: If True, each train/dataset dataset pair will
        be used to create a step. Overrides the n_steps parameter.
    :param reproducibility_data: If not None, overrides all the other
        scenario definition options. This is usually a dictionary containing
        data used to reproduce a specific experiment. One can use the
        ``get_reproducibility_data`` method to get (and even distribute)
        the experiment setup so that it can be loaded by passing it as this
        parameter. In this way one can be sure that the same specific
        experimental setup is being used (for reproducibility purposes).
        Beware that, in order to reproduce an experiment, the same train and
        test datasets must be used. Defaults to None.

    :returns: A :class:`NCScenario` instance.
    """
    if class_ids_from_zero_from_first_step and class_ids_from_zero_in_each_step:
        raise ValueError('Invalid mutually exclusive options '
                         'class_ids_from_zero_from_first_step and '
                         'classes_ids_from_zero_in_each_step set at the '
                         'same time')

    if len(train_dataset_list) != len(test_dataset_list):
        raise ValueError('Train/test dataset lists must contain the '
                         'exact same number of datasets')

    if per_step_classes and one_dataset_per_step:
        raise ValueError('Both per_step_classes and one_dataset_per_step are'
                         'used, but those options are mutually exclusive')

    if fixed_class_order and one_dataset_per_step:
        raise ValueError('Both fixed_class_order and one_dataset_per_step are'
                         'used, but those options are mutually exclusive')

    seq_train_dataset, seq_test_dataset, mapping = \
        concat_datasets_sequentially(train_dataset_list, test_dataset_list)

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
        n_steps = len(train_dataset_list)

    return NCScenario(seq_train_dataset, seq_test_dataset,
                      n_steps,
                      task_labels,
                      shuffle, seed, fixed_class_order,
                      per_step_classes,
                      class_ids_from_zero_from_first_step,
                      class_ids_from_zero_in_each_step,
                      reproducibility_data)


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


__all__ = ['create_nc_single_dataset_scenario',
           'create_nc_multi_dataset_scenario']
