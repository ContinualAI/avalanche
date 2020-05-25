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

from typing import Tuple, Generic, List, Union, Sequence, Optional

from .nc_definitions import T_train_set_w_targets, T_test_set_w_targets, \
    DatasetPart, MTSingleSet, MTMultipleSet
from .nc_utils import make_transformation_subset
from .nc_generic_scenario import NCGenericScenario, NCGenericBatchInfo
from avalanche.training.utils.transform_dataset import TransformationSubset


class NCMultiTaskScenario(Generic[T_train_set_w_targets,
                                  T_test_set_w_targets]):
    """
    This class defines a "New Classes" multi task scenario based on a
    :class:`NCGenericScenario` instance. Once created, an instance of this
    class can be iterated in order to obtain the task sequence under
    the form of instances of :class:`NCTaskInfo`.

    Instances of this class can be creating using the constructor directly.
    However, we recommend using facilities like:
    :func:`benchmarks.scenarios.create_nc_single_dataset_sit_scenario`,
    :func:`benchmarks.scenarios.create_nc_single_dataset_multi_task_scenario`,
    :func:`benchmarks.scenarios.create_nc_multi_dataset_sit_scenario` and
    :func:`benchmarks.scenarios.create_nc_multi_dataset_multi_task_scenario`.

    This class acts as a wrapper for :class:`NCGenericScenario`, adding the
    task label as the output to training/test set related functions
    (see: :class:`NCTaskInfo`).
    """
    def __init__(self,
                 nc_generic_scenario: NCGenericScenario[T_train_set_w_targets,
                                                        T_test_set_w_targets],
                 classes_ids_from_zero_in_each_task: bool = True):
        """
        Creates a NC multi task scenario given a :class:`NCGenericScenario`
        instance. That instance will be used as the batches factory.

        :param nc_generic_scenario: The :class:`NCGenericScenario` instance
            used to populate this scenario.
        :param classes_ids_from_zero_in_each_task: If True, class ids will be
            mapped to range [0, n_classes) for each task. Defaults to True.
        """
        # nc_generic_scenario keeps a reference to the NCGenericScenario
        self.nc_generic_scenario: \
            NCGenericScenario[T_train_set_w_targets,
                              T_test_set_w_targets] = nc_generic_scenario

        # n_tasks is the number of tasks for this scenario
        self.n_tasks: int = self.nc_generic_scenario.n_batches

        # n_classes is the overall number of classes
        # (copied from the nc_generic_scenario instance for easier access)
        self.n_classes: int = self.nc_generic_scenario.n_classes

        self.classes_ids_from_zero_in_each_task: bool = \
            classes_ids_from_zero_in_each_task

        self.class_mapping = [-1 for _ in range(self.n_classes)]

        if classes_ids_from_zero_in_each_task:
            for task_id in range(self.n_tasks):
                original_classes_ids = \
                    nc_generic_scenario.classes_in_batch[task_id]
                for mapped_class_id, original_id in \
                        enumerate(original_classes_ids):
                    self.class_mapping[original_id] = mapped_class_id

            self.classes_in_task = []
            for task_id in range(self.n_tasks):
                n_classes_this_task = \
                    nc_generic_scenario.n_classes_per_batch[task_id]
                self.classes_in_task.append(list(range(0, n_classes_this_task)))
        else:
            self.class_mapping = list(range(self.n_classes))
            self.classes_in_task = nc_generic_scenario.classes_in_batch

        self.original_classes_in_task = nc_generic_scenario.classes_in_batch

    def __len__(self) -> int:
        return self.n_tasks

    def __getitem__(self, task_idx) -> 'NCTaskInfo[' \
                                       'T_train_set_w_targets,' \
                                       'T_test_set_w_targets]':
        return NCTaskInfo(self, self.nc_generic_scenario[task_idx],
                          current_task=task_idx)


class NCTaskInfo(Generic[T_train_set_w_targets,
                         T_test_set_w_targets]):
    """
    Defines a "New Classes" task. It defines methods to obtain the current,
    previous, cumulative and future training and test sets. It also defines
    fields that can be used to check which classes are in this, previous and
    future batches. Instances of this class are usually created when iterating
    over the :class:`NCMultiDatasetMultiTaskScenario`.

    It keeps a reference to that :class:`NCMultiDatasetMultiTaskScenario`
    instance, which can be used to retrieve additional info about the
    scenario.
    """
    def __init__(self,
                 nc_task_scenario:
                 NCMultiTaskScenario[T_train_set_w_targets,
                                     T_test_set_w_targets],
                 sit_batch_info: NCGenericBatchInfo[T_train_set_w_targets,
                                                    T_test_set_w_targets],
                 current_task: int = -1):
        """
        Creates a NCMultiDatasetTaskInfo instance given the root scenario.
        Instances of this class are usually created automatically while
        iterating over an instance of :class:`NCMultiDatasetMultiTaskScenario`.
        
        :param nc_task_scenario: The scenario
        :param sit_batch_info: The batch info
        :param current_task: The task ID
        """
        self.scenario: NCMultiTaskScenario[T_train_set_w_targets,
                                           T_test_set_w_targets] \
            = nc_task_scenario

        self.current_task: int = current_task

        # Just wrap NCGenericGenericBatchInfo
        self._sit_batch_info = sit_batch_info

        # List of classes (original IDs) of current and previous batches,
        # in their encounter order
        self.classes_seen_so_far: List[int] = []

        # The list of classes (original IDs) in this task
        self.classes_in_this_task: List[int] = []

        # The list of classes (original IDs) in previous batches,
        # in their encounter order
        self.previous_classes: List[int] = []

        # The list of classes (original IDs) of next batches,
        # in their encounter order
        self.future_classes: List[int] = []

        # _go_to_batch initializes the above lists
        self._go_to_task()

    def current_training_set(self, bucket_classes=False, sort_classes=False,
                             sort_indexes=False) -> MTSingleSet:
        """
        Gets the training set for the current task

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The current task training set, as a tuple containing the
            Dataset and the task label.
        """
        return self.task_specific_training_set(self.current_task,
                                               bucket_classes=bucket_classes,
                                               sort_classes=sort_classes,
                                               sort_indexes=sort_indexes)

    def cumulative_training_sets(self, include_current_task: bool = True,
                                 bucket_classes=False, sort_classes=False,
                                 sort_indexes=False) -> MTMultipleSet:
        """
        Gets the cumulative training set

        :param include_current_task: If True, include the current task training
            set. Defaults to True.
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The cumulative training sets, as a list. Each element of the
            list is a tuple containing the Dataset and the task label.
        """
        if include_current_task:
            tasks = range(0, self.current_task+1)
        else:
            tasks = range(0, self.current_task)
        return self.__make_train_subset(tasks, bucket_classes=bucket_classes,
                                        sort_classes=sort_classes,
                                        sort_indexes=sort_indexes)

    def complete_training_sets(self, bucket_classes=False, sort_classes=False,
                               sort_indexes=False) -> MTMultipleSet:
        """
        Gets the complete training set

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: All the training sets, as a list. Each element of
            the list is a tuple containing the Dataset and the task label.
        """
        return self.__make_train_subset(list(range(0, self.scenario.n_tasks)),
                                        bucket_classes=bucket_classes,
                                        sort_classes=sort_classes,
                                        sort_indexes=sort_indexes)

    def future_training_sets(self, bucket_classes=False, sort_classes=False,
                             sort_indexes=False) \
            -> MTMultipleSet:
        """
        Gets the "future" training set. That is, a dataset made of training
        patterns belonging to not-already-encountered classes.

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The future training sets, as a list. Each element of the
            list is a tuple containing the Dataset and the task label.
        """
        return self.__make_train_subset(range(self.current_task+1,
                                              self.scenario.n_tasks),
                                        bucket_classes=bucket_classes,
                                        sort_classes=sort_classes,
                                        sort_indexes=sort_indexes)

    def task_specific_training_set(self, task_id: int,
                                   bucket_classes=False,
                                   sort_classes=False, sort_indexes=False) \
            -> MTSingleSet:
        """
        Gets the training set of a specific task, given its ID.

        :param task_id: The ID of the task
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The required training set, as a tuple containing the Dataset
            and the task label.
        """
        return self.__make_train_subset(task_id,
                                        bucket_classes=bucket_classes,
                                        sort_classes=sort_classes,
                                        sort_indexes=sort_indexes)[0]

    def training_set_part(self, dataset_part: DatasetPart,
                          bucket_classes=False, sort_classes=False,
                          sort_indexes=False) \
            -> Union[MTSingleSet, MTMultipleSet]:
        """
        Gets the training subset of a specific part of the scenario.

        :param dataset_part: The part of the scenario
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The training sets of the desired part, as a list. Each element
            of the list is a tuple containing the Dataset and the task label.
        """
        if dataset_part == DatasetPart.CURRENT:
            return self.current_training_set(bucket_classes=bucket_classes,
                                             sort_classes=sort_classes,
                                             sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.CUMULATIVE:
            return self.cumulative_training_sets(include_current_task=True,
                                                 bucket_classes=bucket_classes,
                                                 sort_classes=sort_classes,
                                                 sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.OLD:
            return self.cumulative_training_sets(include_current_task=False,
                                                 bucket_classes=bucket_classes,
                                                 sort_classes=sort_classes,
                                                 sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.FUTURE:
            return self.future_training_sets(bucket_classes=bucket_classes,
                                             sort_classes=sort_classes,
                                             sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.COMPLETE:
            return self.complete_training_sets(bucket_classes=bucket_classes,
                                               sort_classes=sort_classes,
                                               sort_indexes=sort_indexes)
        else:
            raise ValueError('Unsupported dataset part')

    def current_test_set(self, bucket_classes=False, sort_classes=False,
                         sort_indexes=False) \
            -> MTSingleSet:
        """
        Gets the test set for the current batch

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The current test sets, as a tuple containing the Dataset and
            the task label.
        """
        return self.task_specific_test_set(self.current_task,
                                           bucket_classes=bucket_classes,
                                           sort_classes=sort_classes,
                                           sort_indexes=sort_indexes)

    def cumulative_test_sets(self, include_current_task: bool = True,
                             bucket_classes=False, sort_classes=False,
                             sort_indexes=False) -> MTMultipleSet:
        """
        Gets the cumulative test set

        :param include_current_task: If True, include the current task training
            set. Defaults to True.
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The cumulative test sets, as a list. Each element of the
            list is a tuple containing the Dataset and the task label.
        """
        if include_current_task:
            tasks = range(0, self.current_task + 1)
        else:
            tasks = range(0, self.current_task)
        return self.__make_test_subset(tasks, bucket_classes=bucket_classes,
                                       sort_classes=sort_classes,
                                       sort_indexes=sort_indexes)

    def complete_test_sets(self, bucket_classes=False, sort_classes=False,
                           sort_indexes=False) -> MTMultipleSet:
        """
        Gets the complete test set

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: All the test sets, as a list. Each element of
            the list is a tuple containing the Dataset and the task label.
        """
        return self.__make_test_subset(list(range(0, self.scenario.n_tasks)),
                                       bucket_classes=bucket_classes,
                                       sort_classes=sort_classes,
                                       sort_indexes=sort_indexes)

    def future_test_sets(self, bucket_classes=False, sort_classes=False,
                         sort_indexes=False) -> MTMultipleSet:
        """
        Gets the "future" test set. That is, a dataset made of training patterns
        belonging to not-already-encountered classes.

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The future test sets, as a list. Each element of the
            list is a tuple containing the Dataset and the task label.
        """
        return self.__make_test_subset(range(self.current_task + 1,
                                             self.scenario.n_tasks),
                                       bucket_classes=bucket_classes,
                                       sort_classes=sort_classes,
                                       sort_indexes=sort_indexes)

    def task_specific_test_set(self, task_id: int, bucket_classes=False,
                               sort_classes=False, sort_indexes=False) \
            -> MTSingleSet:
        """
        Gets the test set of a specific batch, given its ID.

        :param task_id: The ID of the batch
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The required test set, as a tuple containing the Dataset
            and the task label.
        """
        return self.__make_test_subset(task_id,
                                       bucket_classes=bucket_classes,
                                       sort_classes=sort_classes,
                                       sort_indexes=sort_indexes)[0]

    def test_set_part(self, dataset_part: DatasetPart, bucket_classes=False,
                      sort_classes=False, sort_indexes=False) \
            -> Union[MTSingleSet, MTMultipleSet]:
        """
        Gets the test subset of a specific part of the scenario.

        :param dataset_part: The part of the scenario
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The test sets of the desired part, as a list. Each element
            of the list is a tuple containing the Dataset and the task label.
        """
        if dataset_part == DatasetPart.CURRENT:
            return self.current_test_set(bucket_classes=bucket_classes,
                                         sort_classes=sort_classes,
                                         sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.CUMULATIVE:
            return self.cumulative_test_sets(include_current_task=True,
                                             bucket_classes=bucket_classes,
                                             sort_classes=sort_classes,
                                             sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.OLD:
            return self.cumulative_test_sets(include_current_task=False,
                                             bucket_classes=bucket_classes,
                                             sort_classes=sort_classes,
                                             sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.FUTURE:
            return self.future_test_sets(bucket_classes=bucket_classes,
                                         sort_classes=sort_classes,
                                         sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.COMPLETE:
            return self.complete_test_sets(bucket_classes=bucket_classes,
                                           sort_classes=sort_classes,
                                           sort_indexes=sort_indexes)
        else:
            raise ValueError('Unsupported dataset part')

    def disable_transformations(self) -> 'NCTaskInfo[' \
                                         'T_train_set_w_targets, ' \
                                         'T_test_set_w_targets]':
        """
        Returns a new batch info instance in which transformations are disabled.
        The current instance is not affected. This is useful when there is a
        need to access raw data. Can be used when picking and storing
        rehearsal/replay patterns.

        :returns: A new NCTaskInfo in which transformations are
            disabled.
        """

        return NCTaskInfo(
            self.scenario, self._sit_batch_info.disable_transformations(),
            current_task=self.current_task)

    def enable_transformations(self) -> 'NCTaskInfo[' \
                                        'T_train_set_w_targets, ' \
                                        'T_test_set_w_targets]':
        """
        Returns a new batch info instance in which transformations are enabled.
        The current instance is not affected. When created the
        NCGenericGenericBatchInfo instance already has transformations enabled.
        This method can be used to re-enable transformations after a previous
        call to disable_transformations().

        :returns: A new NCTaskInfo in which transformations are
            enabled.
        """
        return NCTaskInfo(
            self.scenario, self._sit_batch_info.enable_transformations(),
            current_task=self.current_task)

    def with_train_transformations(self) -> 'NCTaskInfo[' \
                                            'T_train_set_w_targets, ' \
                                            'T_test_set_w_targets]':
        """
        Returns a new batch info instance in which train transformations are
        applied to both training and test sets. The current instance is not
        affected.

        :returns: A new NCTaskInfo in which train transformations
            are applied to both training and test sets.
        """
        return NCTaskInfo(
            self.scenario, self._sit_batch_info.with_train_transformations(),
            current_task=self.current_task)

    def with_test_transformations(self) -> 'NCTaskInfo[' \
                                           'T_train_set_w_targets, ' \
                                           'T_test_set_w_targets]':
        """
        Returns a new batch info instance in which test transformations are
        applied to both training and test sets. The current instance is
        not affected. This is useful to get the accuracy on the training set
        without considering the usual training data augmentations.

        :returns: A new NCTaskInfo in which test transformations
            are applied to both training and test sets.
        """
        return NCTaskInfo(
            self.scenario, self._sit_batch_info.with_test_transformations(),
            current_task=self.current_task)
    
    def __make_subset(self, tasks: Union[int, Sequence[int]],
                      is_train: bool, **kwargs):
        """
        Given the batches IDs list and the dataset type (train or test),
        returns a list of tuples (dataset, task_label) in order specified by the
        batches parameter.

        When running a single incremental task scenario, the "batches" parameter
        will be considered as defining the "batch" IDs (not the batches ones).

        :param tasks: The list of batches IDs. Can be a single int.
        :param is_train: when True, training sets will be returned. When False,
            test sets will be returned.

        :returns: A list of tuples each containing 2 elements: the dataset and
            the corresponding task label. Note that, when running a single
            incremental task scenario, the second element of the tuple(s) will
            always be "0".
        """
        if isinstance(tasks, int):  # Required single task
            return self.__make_subset([tasks], is_train, **kwargs)

        classes_mapping = self.scenario.class_mapping
        result: List[Tuple[TransformationSubset, int]] = []

        for task_id in tasks:
            if is_train:
                dataset = self._sit_batch_info. \
                    batch_specific_training_set(task_id, **kwargs)
            else:
                dataset = self._sit_batch_info. \
                    batch_specific_test_set(task_id, **kwargs)

            dataset = TransformationSubset(
                dataset, None, class_mapping=classes_mapping)
            subset = make_transformation_subset(
                dataset, None, None, list(range(self.scenario.n_classes)),
                **kwargs)

            result.append((subset, task_id))

        return result
        
    def __make_train_subset(self, tasks: Union[int, Sequence[int]], **kwargs):
        return self.__make_subset(tasks, True, **kwargs)

    def __make_test_subset(self, tasks: Union[int, Sequence[int]], **kwargs):
        return self.__make_subset(tasks, False, **kwargs)

    def __get_tasks_classes(self, task_start: int,
                            task_end: Optional[int] = None) -> List[int]:
        """
        Gets a list of classes contained int the given batches. The batches are
        defined by range. This means that only the classes in range
        [batch_start, batch_end) will be included.

        :param task_start: The starting task ID
        :param task_end: The final task ID. Can be None.

        :returns: The classes contained in the required task range.
        """
        # Ref: https://stackoverflow.com/a/952952
        if task_end is None:
            return [
                item for sublist in
                self.scenario.original_classes_in_task[task_start:]
                for item in sublist]
        else:
            return [
                item for sublist in
                self.scenario.original_classes_in_task[task_start:task_end]
                for item in sublist]

    def _go_to_task(self):
        if self.current_task >= 0:
            self.classes_in_this_task = self.scenario.original_classes_in_task[
                self.current_task]
            self.previous_classes = self.__get_tasks_classes(
                0, self.current_task)
            self.classes_seen_so_far = \
                self.previous_classes + self.classes_in_this_task
            self.future_classes = self.__get_tasks_classes(self.current_task)
        else:
            self.classes_in_this_task = []
            self.previous_classes = []
            self.classes_seen_so_far = []
            self.future_classes = self.__get_tasks_classes(0)


class NCSingleTaskScenario(Generic[T_train_set_w_targets,
                                   T_test_set_w_targets]):
    """
    This class defines a "New Classes" Single Incremental Task scenario based
    on a :class:`NCGenericScenario` instance. Once created, an instance of this
    class can be iterated in order to obtain the batch sequence under
    the form of instances of :class:`NCBatchInfo`.

    Instances of this class can be creating using the constructor directly.
    However, we recommend using facilities like:
    :func:`benchmarks.scenarios.create_nc_single_dataset_sit_scenario`,
    :func:`benchmarks.scenarios.create_nc_single_dataset_multi_task_scenario`,
    :func:`benchmarks.scenarios.create_nc_multi_dataset_sit_scenario` and
    :func:`benchmarks.scenarios.create_nc_multi_dataset_multi_task_scenario`.

    This class acts as a wrapper for :class:`NCGenericScenario`, adding the
    task label (always "0") as the output to training/test set related functions
    (see: :class:`NCBatchInfo`).
    """
    def __init__(self,
                 nc_generic_scenario: NCGenericScenario[T_train_set_w_targets,
                                                        T_test_set_w_targets]):
        """
        Creates a NC Single Incremental Task scenario given a
        :class:`NCGenericScenario` instance. That instance will be used as the
        batches factory.

        :param nc_generic_scenario: The :class:`NCGenericScenario` instance
            used to populate this scenario.
        """
        # nc_generic_scenario keeps a reference to the NCGenericScenario
        self.nc_generic_scenario: \
            NCGenericScenario[T_train_set_w_targets,
                              T_test_set_w_targets] = nc_generic_scenario

        # n_batches is the number of batches in this scenario
        self.n_batches: int = self.nc_generic_scenario.n_batches

        # n_classes is the overall number of classes (copied from the
        # nc_generic_scenario instance for easier access, like classes_in_batch)
        self.n_classes: int = self.nc_generic_scenario.n_classes
        self.classes_in_batch = nc_generic_scenario.classes_in_batch

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, batch_idx) -> 'NCBatchInfo[' \
                                        'T_train_set_w_targets,' \
                                        'T_test_set_w_targets]':
        return NCBatchInfo(self, self.nc_generic_scenario[batch_idx],
                           current_batch=batch_idx)


class NCBatchInfo(Generic[T_train_set_w_targets, T_test_set_w_targets]):
    """
    Defines a "New Classes" batch. It defines methods to obtain the current,
    previous, cumulative and future training and test sets. It also defines
    fields that can be used to check which classes are in this, previous and
    future batches. Instances of this class are usually created when iterating
    over the :class:`NCSingleTaskScenario`.

    It keeps a reference to that :class:`NCSingleTaskScenario` instance, which can be
    used to retrieve additional info about the scenario.
    """

    def __init__(self,
                 sit_scenario:
                 NCSingleTaskScenario[T_train_set_w_targets,
                                      T_test_set_w_targets],
                 sit_batch_info: NCGenericBatchInfo[T_train_set_w_targets,
                                                    T_test_set_w_targets],
                 current_batch: int = -1):
        """
        Creates a NCBatchInfo instance given the root scenario. 
        Instances of this class are usually created automatically while 
        iterating over an instance of :class:`NCSingleTaskScenario`.

        :param sit_scenario: The scenario
        :param sit_batch_info: The batch info
        :param current_batch: The batch ID
        """
        self.scenario: NCSingleTaskScenario[T_train_set_w_targets,
                                            T_test_set_w_targets] = sit_scenario

        self.current_batch: int = current_batch

        # Just wrap NCGenericGenericBatchInfo
        self._sit_batch_info = sit_batch_info

        # The list of classes in this batch
        self.classes_in_this_batch = sit_batch_info.classes_in_this_batch

        # The list of classes in previous batches,
        # in their encounter order
        self.previous_classes = sit_batch_info.previous_classes

        # List of classes of current and previous batches,
        # in their encounter order
        self.classes_seen_so_far = sit_batch_info.classes_seen_so_far

        # The list of classes of next batches,
        # in their encounter order
        self.future_classes = sit_batch_info.future_classes

    def current_training_set(self, bucket_classes=False, sort_classes=False,
                             sort_indexes=False) -> MTSingleSet:
        """
        Gets the training set for the current batch

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The current batch training set, as a tuple containing the
            Dataset and the task label "0".
        """
        return self.batch_specific_training_set(self.current_batch,
                                                bucket_classes=bucket_classes,
                                                sort_classes=sort_classes,
                                                sort_indexes=sort_indexes)

    def cumulative_training_sets(self, include_current_batch: bool = True,
                                 bucket_classes=False, sort_classes=False,
                                 sort_indexes=False) -> MTMultipleSet:
        """
        Gets the cumulative training set

        :param include_current_batch: If True, include the current batch
            training set. Defaults to True.
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The cumulative training sets, as a list. Each element of the
            list is a tuple containing the Dataset and the task label "0".
        """
        if include_current_batch:
            batches = range(0, self.current_batch + 1)
        else:
            batches = range(0, self.current_batch)
        return self.__make_train_subset(batches, bucket_classes=bucket_classes,
                                        sort_classes=sort_classes,
                                        sort_indexes=sort_indexes)

    def complete_training_sets(self, bucket_classes=False, sort_classes=False,
                               sort_indexes=False) -> MTMultipleSet:
        """
        Gets the complete training set

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: All the training sets, as a list. Each element of
            the list is a tuple containing the Dataset and the task label "0".
        """
        return self.__make_train_subset(list(range(0, self.scenario.n_batches)),
                                        bucket_classes=bucket_classes,
                                        sort_classes=sort_classes,
                                        sort_indexes=sort_indexes)

    def future_training_sets(self, bucket_classes=False, sort_classes=False,
                             sort_indexes=False) \
            -> MTMultipleSet:
        """
        Gets the "future" training set. That is, a dataset made of training
        patterns belonging to not-already-encountered classes.

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The future training sets, as a list. Each element of the
            list is a tuple containing the Dataset and the task label "0".
        """
        return self.__make_train_subset(range(self.current_batch + 1,
                                              self.scenario.n_batches),
                                        bucket_classes=bucket_classes,
                                        sort_classes=sort_classes,
                                        sort_indexes=sort_indexes)

    def batch_specific_training_set(self, batch_id: int,
                                    bucket_classes=False,
                                    sort_classes=False, sort_indexes=False) \
            -> MTSingleSet:
        """
        Gets the training set of a specific batch, given its ID.

        :param batch_id: The ID of the batch
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The required training set, as a tuple containing the Dataset
            and the task label "0".
        """
        return self.__make_train_subset(batch_id,
                                        bucket_classes=bucket_classes,
                                        sort_classes=sort_classes,
                                        sort_indexes=sort_indexes)[0]

    def training_set_part(self, dataset_part: DatasetPart,
                          bucket_classes=False, sort_classes=False,
                          sort_indexes=False) \
            -> Union[MTSingleSet, MTMultipleSet]:
        """
        Gets the training subset of a specific part of the scenario.

        :param dataset_part: The part of the scenario
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The training sets of the desired part, as a list. Each element
            of the list is a tuple containing the Dataset and the task label
            "0".
        """
        if dataset_part == DatasetPart.CURRENT:
            return self.current_training_set(bucket_classes=bucket_classes,
                                             sort_classes=sort_classes,
                                             sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.CUMULATIVE:
            return self.cumulative_training_sets(include_current_batch=True,
                                                 bucket_classes=bucket_classes,
                                                 sort_classes=sort_classes,
                                                 sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.OLD:
            return self.cumulative_training_sets(include_current_batch=False,
                                                 bucket_classes=bucket_classes,
                                                 sort_classes=sort_classes,
                                                 sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.FUTURE:
            return self.future_training_sets(bucket_classes=bucket_classes,
                                             sort_classes=sort_classes,
                                             sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.COMPLETE:
            return self.complete_training_sets(bucket_classes=bucket_classes,
                                               sort_classes=sort_classes,
                                               sort_indexes=sort_indexes)
        else:
            raise ValueError('Unsupported dataset part')

    def current_test_set(self, bucket_classes=False, sort_classes=False,
                         sort_indexes=False) \
            -> MTSingleSet:
        """
        Gets the test set for the current batch

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The current test sets, as a tuple containing the Dataset and
            the task label "0".
        """
        return self.batch_specific_test_set(self.current_batch,
                                            bucket_classes=bucket_classes,
                                            sort_classes=sort_classes,
                                            sort_indexes=sort_indexes)

    def cumulative_test_sets(self, include_current_batch: bool = True,
                             bucket_classes=False, sort_classes=False,
                             sort_indexes=False) -> MTMultipleSet:
        """
        Gets the cumulative test set

        :param include_current_batch: If True, include the current batch
            training set. Defaults to True.
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The cumulative test sets, as a list. Each element of the
            list is a tuple containing the Dataset and the task label "0".
        """
        if include_current_batch:
            batches = range(0, self.current_batch + 1)
        else:
            batches = range(0, self.current_batch)
        return self.__make_test_subset(batches, bucket_classes=bucket_classes,
                                       sort_classes=sort_classes,
                                       sort_indexes=sort_indexes)

    def complete_test_sets(self, bucket_classes=False, sort_classes=False,
                           sort_indexes=False) -> MTMultipleSet:
        """
        Gets the complete test set

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: All the test sets, as a list. Each element of
            the list is a tuple containing the Dataset and the task label "0".
        """
        return self.__make_test_subset(list(range(0, self.scenario.n_batches)),
                                       bucket_classes=bucket_classes,
                                       sort_classes=sort_classes,
                                       sort_indexes=sort_indexes)

    def future_test_sets(self, bucket_classes=False, sort_classes=False,
                         sort_indexes=False) -> MTMultipleSet:
        """
        Gets the "future" test set. That is, a dataset made of training patterns
        belonging to not-already-encountered classes.

        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The future test sets, as a list. Each element of the
            list is a tuple containing the Dataset and the task label "0".
        """
        return self.__make_test_subset(range(self.current_batch + 1,
                                             self.scenario.n_batches),
                                       bucket_classes=bucket_classes,
                                       sort_classes=sort_classes,
                                       sort_indexes=sort_indexes)

    def batch_specific_test_set(self, batch_id: int, bucket_classes=False,
                                sort_classes=False, sort_indexes=False) \
            -> MTSingleSet:
        """
        Gets the test set of a specific batch, given its ID.

        :param batch_id: The ID of the batch
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The required test set, as a tuple containing the Dataset
            and the task label "0".
        """
        return self.__make_test_subset(batch_id,
                                       bucket_classes=bucket_classes,
                                       sort_classes=sort_classes,
                                       sort_indexes=sort_indexes)[0]

    def test_set_part(self, dataset_part: DatasetPart, bucket_classes=False,
                      sort_classes=False, sort_indexes=False) \
            -> Union[MTSingleSet, MTMultipleSet]:
        """
        Gets the test subset of a specific part of the scenario.

        :param dataset_part: The part of the scenario
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.

        :returns: The test sets of the desired part, as a list. Each element
            of the list is a tuple containing the Dataset and the task label
            "0".
        """
        if dataset_part == DatasetPart.CURRENT:
            return self.current_test_set(bucket_classes=bucket_classes,
                                         sort_classes=sort_classes,
                                         sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.CUMULATIVE:
            return self.cumulative_test_sets(include_current_batch=True,
                                             bucket_classes=bucket_classes,
                                             sort_classes=sort_classes,
                                             sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.OLD:
            return self.cumulative_test_sets(include_current_batch=False,
                                             bucket_classes=bucket_classes,
                                             sort_classes=sort_classes,
                                             sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.FUTURE:
            return self.future_test_sets(bucket_classes=bucket_classes,
                                         sort_classes=sort_classes,
                                         sort_indexes=sort_indexes)
        elif dataset_part == DatasetPart.COMPLETE:
            return self.complete_test_sets(bucket_classes=bucket_classes,
                                           sort_classes=sort_classes,
                                           sort_indexes=sort_indexes)
        else:
            raise ValueError('Unsupported dataset part')

    def disable_transformations(self) -> 'NCBatchInfo[' \
                                         'T_train_set_w_targets, ' \
                                         'T_test_set_w_targets]':
        """
        Returns a new batch info instance in which transformations are disabled.
        The current instance is not affected. This is useful when there is a
        need to access raw data. Can be used when picking and storing
        rehearsal/replay patterns.

        :returns: A new NCBatchInfo in which transformations are disabled.
        """

        return NCBatchInfo(
            self.scenario, self._sit_batch_info.disable_transformations(),
            current_batch=self.current_batch)

    def enable_transformations(self) -> 'NCBatchInfo[' \
                                        'T_train_set_w_targets, ' \
                                        'T_test_set_w_targets]':
        """
        Returns a new batch info instance in which transformations are enabled.
        The current instance is not affected. When created the
        NCGenericGenericBatchInfo instance already has transformations enabled.
        This method can be used to re-enable transformations after a previous
        call to disable_transformations().

        :returns: A new NCBatchInfo in which transformations are enabled.
        """
        return NCBatchInfo(
            self.scenario, self._sit_batch_info.enable_transformations(),
            current_batch=self.current_batch)

    def with_train_transformations(self) -> 'NCBatchInfo[' \
                                            'T_train_set_w_targets, ' \
                                            'T_test_set_w_targets]':
        """
        Returns a new batch info instance in which train transformations are
        applied to both training and test sets. The current instance is not
        affected.

        :returns: A new NCBatchInfo in which train transformations
            are applied to both training and test sets.
        """
        return NCBatchInfo(
            self.scenario, self._sit_batch_info.with_train_transformations(),
            current_batch=self.current_batch)

    def with_test_transformations(self) -> 'NCBatchInfo[' \
                                           'T_train_set_w_targets, ' \
                                           'T_test_set_w_targets]':
        """
        Returns a new batch info instance in which test transformations are
        applied to both training and test sets. The current instance is
        not affected. This is useful to get the accuracy on the training set
        without considering the usual training data augmentations.

        :returns: A new NCBatchInfo in which test transformations
            are applied to both training and test sets.
        """
        return NCBatchInfo(
            self.scenario, self._sit_batch_info.with_test_transformations(),
            current_batch=self.current_batch)

    def __make_subset(self, batches: Union[int, Sequence[int]],
                      is_train: bool, **kwargs):
        """
        Given the batches IDs list and the dataset type (train or test),
        returns a list of tuples (dataset, task_label "0") in the order
        specified by the batches parameter.

        :param batches: The list of batches IDs. Can be a single int.
        :param is_train: when True, training sets will be returned. When False,
            test sets will be returned.

        :returns: A list of tuples each containing 2 elements: the dataset and
            the corresponding task label (always "0", as this is a SIT
            scenario).
        """
        if isinstance(batches, int):  # Required single batch
            return self.__make_subset([batches], is_train, **kwargs)

        result: List[Tuple[TransformationSubset, int]] = []

        for batch_id in batches:
            if is_train:
                dataset = self._sit_batch_info. \
                    batch_specific_training_set(batch_id, **kwargs)
            else:
                dataset = self._sit_batch_info. \
                    batch_specific_test_set(batch_id, **kwargs)

            subset = make_transformation_subset(
                dataset, None, None, None, **kwargs)

            result.append((subset, 0))

        return result

    def __make_train_subset(self, batches: Union[int, Sequence[int]], **kwargs):
        return self.__make_subset(batches, True, **kwargs)

    def __make_test_subset(self, batches: Union[int, Sequence[int]], **kwargs):
        return self.__make_subset(batches, False, **kwargs)


__all__ = ['NCMultiTaskScenario', 'NCTaskInfo',
           'NCSingleTaskScenario', 'NCBatchInfo']
