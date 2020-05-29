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

from typing import Sequence, Optional

from avalanche.training.utils.transform_dataset import IDatasetWithTargets, \
    concat_datasets_sequentially
from .ni_scenario import NIScenario


def create_ni_single_dataset_sit_scenario(
        train_dataset: IDatasetWithTargets,
        test_dataset: IDatasetWithTargets,
        n_batches: int, shuffle: bool = True,
        seed: Optional[int] = None,
        balance_batches: bool = False,
        min_class_patterns_in_batch: int = 0,
        fixed_batch_assignment: Optional[Sequence[Sequence[int]]] = None) -> \
        NIScenario:
    """
    Creates a "New Instances - Single Incremental Task" scenario given a couple
    of train and test datasets.

    :param train_dataset: The training dataset.
    :param test_dataset: A list of test dataset.
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
    :param fixed_batch_assignment: If not None, the pattern assignment
        to use. It must be a list with an entry for each batch. Each entry
        is a list that contains the indexes of patterns belonging to that
        batch. Overrides the ``shuffle``, ``balance_batches`` and
        ``min_class_patterns_in_batch`` parameters.

    :returns: A :class:`NIScenario` instance.
    """

    return NIScenario(train_dataset, test_dataset, n_batches,
                      shuffle=shuffle, seed=seed,
                      balance_batches=balance_batches,
                      min_class_patterns_in_batch=min_class_patterns_in_batch,
                      fixed_batch_assignment=fixed_batch_assignment)


def create_ni_multi_dataset_sit_scenario(
        train_dataset_list: Sequence[IDatasetWithTargets],
        test_dataset_list: Sequence[IDatasetWithTargets],
        n_batches: int, shuffle: bool = True,
        seed: Optional[int] = None,
        balance_batches: bool = False,
        min_class_patterns_in_batch: int = 0) -> NIScenario:
    """
    Creates a "New Instances - Single Incremental Task" scenario given a list of
    datasets and the number of batches. The datasets will be merged together.

    Note: train_dataset_list and test_dataset_list must have the same number of
    datasets.

    :param train_dataset_list: A list of training datasets.
    :param test_dataset_list: A list of test datasets.
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

    :returns: A :class:`NIScenario` instance.
    """
    if len(train_dataset_list) != len(test_dataset_list):
        raise ValueError('Train/test dataset lists must contain the '
                         'exact same number of datasets')

    seq_train_dataset, seq_test_dataset, mapping = \
        concat_datasets_sequentially(train_dataset_list, test_dataset_list)

    return NIScenario(seq_train_dataset, seq_test_dataset, n_batches,
                      shuffle=shuffle, seed=seed,
                      balance_batches=balance_batches,
                      min_class_patterns_in_batch=min_class_patterns_in_batch)


__all__ = ['create_ni_single_dataset_sit_scenario',
           'create_ni_multi_dataset_sit_scenario']
