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

from typing import Generic, List

from avalanche.benchmarks.scenarios.generic_definitions import \
    TrainSetWithTargets, TestSetWithTargets, MTSingleSet
from .nc_generic_scenario import NCGenericScenario
from avalanche.training.utils.transform_dataset import TransformationSubset
from avalanche.benchmarks.scenarios.general_cl_scenario import \
    GenericCLScenario, GenericStepInfo
from avalanche.benchmarks.utils import grouped_and_ordered_indexes


# TODO: implement reproducibility_data constructors parameter
# TODO: implement get_reproducibility_data methods


class NCMultiTaskScenario(GenericCLScenario[TrainSetWithTargets,
                                            TestSetWithTargets],
                          Generic[TrainSetWithTargets, TestSetWithTargets]):
    """
    This class defines a "New Classes" multi task scenario based on a
    :class:`NCGenericScenario` instance. Once created, an instance of this
    class can be iterated in order to obtain the task sequence under
    the form of instances of :class:`NCTaskInfo`.

    Instances of this class can be created using the constructor directly.
    However, we recommend using facilities like:
    :func:`.scenario_creation.create_nc_single_dataset_sit_scenario`,
    :func:`.scenario_creation.create_nc_single_dataset_multi_task_scenario`,
    :func:`.scenario_creation.create_nc_multi_dataset_sit_scenario` and
    :func:`.scenario_creation.create_nc_multi_dataset_multi_task_scenario`.

    This class acts as a wrapper for :class:`NCGenericScenario`, adding the
    task label as the output to training/test set related functions
    (see: :class:`NCTaskInfo`).
    """
    def __init__(self,
                 nc_generic_scenario: NCGenericScenario[TrainSetWithTargets,
                                                        TestSetWithTargets],
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
            NCGenericScenario[TrainSetWithTargets,
                              TestSetWithTargets] = nc_generic_scenario

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
                self.classes_in_task.append(
                    list(range(0, n_classes_this_task)))
        else:
            self.class_mapping = list(range(self.n_classes))
            self.classes_in_task = nc_generic_scenario.classes_in_batch

        self.original_classes_in_task = nc_generic_scenario.classes_in_batch

        super(NCMultiTaskScenario, self).__init__(
            self.nc_generic_scenario.train_dataset,
            self.nc_generic_scenario.test_dataset,
            self.nc_generic_scenario.train_steps_patterns_assignment,
            self.nc_generic_scenario.test_steps_patterns_assignment,
            list(range(self.n_tasks)))

    def __getitem__(self, task_id) -> 'NCTaskInfo[' \
                                      'TrainSetWithTargets, ' \
                                      'TestSetWithTargets]':
        return NCTaskInfo(self, task_id)


class NCTaskInfo(GenericStepInfo[NCMultiTaskScenario[TrainSetWithTargets,
                                                     TestSetWithTargets]],
                 Generic[TrainSetWithTargets, TestSetWithTargets]):
    """
    Defines a "New Classes" task. It defines methods to obtain the current,
    previous, cumulative and future training and test sets. It also defines
    fields that can be used to check which classes are in this, previous and
    future batches. Instances of this class are usually created when iterating
    over a :class:`NCMultiTaskScenario` instance.

    It keeps a reference to that :class:`NCMultiTaskScenario`
    instance, which can be used to retrieve additional info about the
    scenario.
    """
    def __init__(self,
                 scenario: NCMultiTaskScenario[TrainSetWithTargets,
                                               TestSetWithTargets],
                 current_task: int,
                 force_train_transformations: bool = False,
                 force_test_transformations: bool = False,
                 are_transformations_disabled: bool = False):
        """
        Creates a NCMultiDatasetTaskInfo instance given the root scenario.
        Instances of this class are usually created automatically while
        iterating over an instance of :class:`NCMultiTaskScenario`.
        
        :param scenario: A reference to the NC scenario
        :param current_task: The task ID
        :param force_train_transformations: If True, train transformations will
            be applied to the test set too. The ``force_test_transformations``
            parameter can't be True at the same time. Defaults to False.
        :param force_test_transformations: If True, test transformations will be
            applied to the training set too. The ``force_train_transformations``
            parameter can't be True at the same time. Defaults to False.
        :param are_transformations_disabled: If True, transformations are
            disabled. That is, patterns and targets will be returned as
            outputted by  the original training and test Datasets. Overrides
            ``force_train_transformations`` and ``force_test_transformations``.
            Defaults to False.
        """

        super(NCTaskInfo, self).__init__(
            scenario, current_task,
            force_train_transformations=force_train_transformations,
            force_test_transformations=force_test_transformations,
            are_transformations_disabled=are_transformations_disabled,
            transformation_step_factory=NCTaskInfo)

        self.current_task: int = current_task

        # The list of classes (original IDs) in this task
        self.classes_in_this_task: List[int] = []

        # The list of classes (original IDs) in previous batches,
        # in their encounter order
        self.previous_classes: List[int] = []

        # List of classes (original IDs) of current and previous batches,
        # in their encounter order
        self.classes_seen_so_far: List[int] = []

        # The list of classes (original IDs) of next batches,
        # in their encounter order
        self.future_classes: List[int] = []

        # _go_to_task initializes the above lists
        self._go_to_task()

    def _make_subset(self, is_train: bool, step: int, **kwargs) -> MTSingleSet:
        subset, t = super()._make_subset(is_train, step)

        subset = TransformationSubset(
            subset, None, class_mapping=self.scenario.class_mapping)

        return TransformationSubset(
            subset,
            grouped_and_ordered_indexes(
                subset.targets, None,
                **kwargs)), t

    def _go_to_task(self):
        class_split = self.scenario.nc_generic_scenario.get_class_split(
            self.current_task)
        self.classes_in_this_task, self.previous_classes, self.\
            classes_seen_so_far, self.future_classes = class_split


class NCSingleTaskScenario(GenericCLScenario[TrainSetWithTargets,
                                             TestSetWithTargets],
                           Generic[TrainSetWithTargets, TestSetWithTargets]):
    """
    This class defines a "New Classes" Single Incremental Task scenario based
    on a :class:`NCGenericScenario` instance. Once created, an instance of this
    class can be iterated in order to obtain the batch sequence under
    the form of instances of :class:`NCBatchInfo`.

    Instances of this class can be created using the constructor directly.
    However, we recommend using facilities like:
    :func:`.scenario_creation.create_nc_single_dataset_sit_scenario`,
    :func:`.scenario_creation.create_nc_single_dataset_multi_task_scenario`,
    :func:`.scenario_creation.create_nc_multi_dataset_sit_scenario` and
    :func:`.scenario_creation.create_nc_multi_dataset_multi_task_scenario`.

    This class acts as a wrapper for :class:`NCGenericScenario`, adding the
    task label (always "0") as the output to training/test set related functions
    (see: :class:`NCBatchInfo`).
    """
    def __init__(self,
                 nc_generic_scenario: NCGenericScenario[TrainSetWithTargets,
                                                        TestSetWithTargets]):
        """
        Creates a NC Single Incremental Task scenario given a
        :class:`NCGenericScenario` instance. That instance will be used as the
        batches factory.

        :param nc_generic_scenario: The :class:`NCGenericScenario` instance
            used to populate this scenario.
        """
        # nc_generic_scenario keeps a reference to the NCGenericScenario
        self.nc_generic_scenario: \
            NCGenericScenario[TrainSetWithTargets,
                              TestSetWithTargets] = nc_generic_scenario

        # n_batches is the number of batches in this scenario
        self.n_batches: int = self.nc_generic_scenario.n_batches

        # n_classes is the overall number of classes (copied from the
        # nc_generic_scenario instance for easier access, like classes_in_batch)
        self.n_classes: int = self.nc_generic_scenario.n_classes
        self.classes_in_batch = nc_generic_scenario.classes_in_batch

        super(NCSingleTaskScenario, self).__init__(
            self.nc_generic_scenario.train_dataset,
            self.nc_generic_scenario.test_dataset,
            self.nc_generic_scenario.train_steps_patterns_assignment,
            self.nc_generic_scenario.test_steps_patterns_assignment,
            [0] * self.n_batches)

    def __getitem__(self, batch_id) -> 'NCBatchInfo[' \
                                       'TrainSetWithTargets,' \
                                       'TestSetWithTargets]':
        return NCBatchInfo(self, batch_id)


class NCBatchInfo(GenericStepInfo[NCMultiTaskScenario[TrainSetWithTargets,
                                                      TestSetWithTargets]],
                  Generic[TrainSetWithTargets, TestSetWithTargets]):
    """
    Defines a "New Classes" batch. It defines methods to obtain the current,
    previous, cumulative and future training and test sets. It also defines
    fields that can be used to check which classes are in this, previous and
    future batches. Instances of this class are usually created when iterating
    over a :class:`NCSingleTaskScenario` instance.

    It keeps a reference to that :class:`NCSingleTaskScenario` instance,
    which can be used to retrieve additional info about the scenario.
    """

    def __init__(self,
                 scenario: NCSingleTaskScenario[TrainSetWithTargets,
                                                TestSetWithTargets],
                 current_batch: int,
                 force_train_transformations: bool = False,
                 force_test_transformations: bool = False,
                 are_transformations_disabled: bool = False):
        """
        Creates a NCBatchInfo instance given the root scenario. 
        Instances of this class are usually created automatically while 
        iterating over an instance of :class:`NCSingleTaskScenario`.

        :param scenario: A reference to the NC scenario
        :param current_batch: The batch ID
        :param force_train_transformations: If True, train transformations will
            be applied to the test set too. The ``force_test_transformations``
            parameter can't be True at the same time. Defaults to False.
        :param force_test_transformations: If True, test transformations will be
            applied to the training set too. The ``force_train_transformations``
            parameter can't be True at the same time. Defaults to False.
        :param are_transformations_disabled: If True, transformations are
            disabled. That is, patterns and targets will be returned as
            outputted by  the original training and test Datasets. Overrides
            ``force_train_transformations`` and ``force_test_transformations``.
            Defaults to False.
        """

        super(NCBatchInfo, self).__init__(
            scenario, current_batch,
            force_train_transformations=force_train_transformations,
            force_test_transformations=force_test_transformations,
            are_transformations_disabled=are_transformations_disabled,
            transformation_step_factory=NCBatchInfo)

        self.current_batch: int = current_batch

        # The list of classes in this batch
        self.classes_in_this_batch: List[int] = []

        # The list of classes in previous batches,
        # in their encounter order
        self.previous_classes: List[int] = []

        # List of classes of current and previous batches,
        # in their encounter order
        self.classes_seen_so_far: List[int] = []

        # The list of classes of next batches,
        # in their encounter order
        self.future_classes: List[int] = []

        # _go_to_batch initializes the above lists
        self._go_to_batch()

    def _go_to_batch(self):
        class_split = self.scenario.nc_generic_scenario.get_class_split(
            self.current_batch)
        self.classes_in_this_task, self.previous_classes, self.\
            classes_seen_so_far, self.future_classes = class_split


__all__ = ['NCMultiTaskScenario', 'NCTaskInfo',
           'NCSingleTaskScenario', 'NCBatchInfo']
