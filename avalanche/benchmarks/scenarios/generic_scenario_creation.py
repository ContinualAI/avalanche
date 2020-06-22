from typing import Sequence

from avalanche.training.utils import IDatasetWithTargets, \
    ConcatDatasetWithTargets
from .generic_cl_scenario import GenericCLScenario


def create_multi_dataset_generic_scenario(
        train_dataset_list: Sequence[IDatasetWithTargets],
        test_dataset_list: Sequence[IDatasetWithTargets],
        task_labels: Sequence[int],
        complete_test_set_only: bool = False) -> GenericCLScenario:
    """
    Creates a generic scenario given a list of datasets and the respective task
    labels. Each training dataset will be considered a separate training step.
    Contents of the datasets will not be changed, including the targets.

    This helper method may be useful when loading the datasets from a set of
    fixed filelist. In its base form, this function accepts a list of test
    datsets that must contain the same amount of datasets of the training list.
    Those pairs are then used to create the "past", "cumulative"
    (a.k.a. growing) and "future" test sets. However, in certain Continual
    Learning scenarios only the concept of "complete" test set makes sense. In
    that case, the ``complete_test_set_only`` should be set to True.
    See the parameter description for more info.

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
       ` `test_dataset_list`` must contain the same amount of datasets.

    :returns: A :class:`NCMultiTaskScenario` instance initialized for the
        the SIT scenario.
    """

    # GenericCLScenario accepts a single training+test sets couple along with
    # the respective list of patterns indexes to include in each step.
    # This means that we have to concat the list of train and test sets
    # and create, for each step, a of indexes.
    # Each dataset describes a different step so the lists of indexes will
    # just be ranges of ascending indexes.
    train_structure = []
    train_dataset = ConcatDatasetWithTargets(train_dataset_list)
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
        test_dataset = test_dataset_list[0]
    else:
        test_dataset = ConcatDatasetWithTargets(test_dataset_list)
        test_structure = []
        next_idx = 0
        for test_dataset in test_dataset_list:
            end_idx = next_idx + len(test_dataset)
            test_structure.append(range(next_idx, end_idx))
            next_idx = end_idx

    # GenericCLScenario constructor will also check that the same amount of
    # train/test sets + task_labels have been defined.
    return GenericCLScenario(
        train_dataset, test_dataset, train_structure, test_structure,
        task_labels, return_complete_test_set_only=complete_test_set_only)


__all__ = ['create_multi_dataset_generic_scenario']
