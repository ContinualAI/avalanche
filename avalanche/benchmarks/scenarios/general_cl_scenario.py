from typing import Generic, TypeVar, Union, Sequence, Callable, Optional, \
    Mapping, Dict, Any
from abc import ABC, abstractmethod

from avalanche.benchmarks.scenarios import MTSingleSet, MTMultipleSet, \
    DatasetPart, TrainSetWithTargets, TestSetWithTargets
from avalanche.training.utils import TransformationSubset
from avalanche.benchmarks.utils import grouped_and_ordered_indexes

TBaseScenario = TypeVar('TBaseScenario')


class GenericCLScenario(Generic[TrainSetWithTargets, TestSetWithTargets]):
    """
    Base implementation of a Continual Learning scenario. A Continual Learning
    scenario is defined by a sequence of steps (batches or tasks depending on
    the terminology), with each step containing the training (and test) data
    that becomes available at a certain time instant.

    From a practical point of view, this means that we usually have to define
    two datasets (training and test), and some way to assign the patterns
    contained in these datasets to each step.

    This assignment is usually made in children classes, with this class serving
    as the more general implementation. This class handles the most simple type
    of assignment: each step is defined by a list of patterns (identified by
    their indexes) contained in that step.
    """
    def __init__(self, train_dataset: TrainSetWithTargets,
                 test_dataset: TestSetWithTargets,
                 train_steps_patterns_assignment: Mapping[int, Sequence[int]],
                 test_steps_patterns_assignment: Mapping[int, Sequence[int]],
                 task_labels: Mapping[int, int],
                 reproducibility_data: Optional[Dict[str, Any]] = None):
        """
        Creates an instance of a Continual Learning scenario.

        The scenario is defined by the train and test datasets plus the
        assignment of patterns to steps (batches/tasks).

        :param train_dataset: The training dataset.
        :param test_dataset:  The test dataset.
        :param train_steps_patterns_assignment: A list of steps. Each step is
            in turn defined by a list of integers describing the pattern index
            inside the training dataset.
        :param test_steps_patterns_assignment: A list of steps. Each step is
            in turn defined by a list of integers describing the pattern index
            inside the test dataset.
        :param task_labels: The mapping from step IDs to task labels, usually
            as a list of integers.
        :param reproducibility_data: If not None, overrides the
            ``train/test_steps_patterns_assignment` and ``task_labels``
            parameters. This is usually a dictionary containing data used to
            reproduce a specific experiment. One can use the
            ``get_reproducibility_data`` method to get (and even distribute)
            the experiment setup so that it can be loaded by passing it as this
            parameter. In this way one can be sure that the same specific
            experimental setup is being used (for reproducibility purposes).
            Beware that, in order to reproduce an experiment, the same train and
            test datasets must be used.
        """
        self.train_dataset: TrainSetWithTargets = train_dataset
        self.test_dataset: TestSetWithTargets = test_dataset
        self.train_steps_patterns_assignment: Mapping[int, Sequence[int]]
        self.test_steps_patterns_assignment: Mapping[int, Sequence[int]]
        self.task_labels: Mapping[int, int] = task_labels

        if hasattr(train_dataset, 'transform') and \
                train_dataset.transform is not None:
            self.train_transform = train_dataset.transform
            train_dataset.transform = None
        if hasattr(train_dataset, 'target_transform') and \
                train_dataset.target_transform is not None:
            self.train_target_transform = train_dataset.target_transform
            train_dataset.target_transform = None

        if hasattr(test_dataset, 'transform') and \
                test_dataset.transform is not None:
            self.test_transform = test_dataset.transform
            test_dataset.transform = None
        if hasattr(test_dataset, 'target_transform') and \
                test_dataset.target_transform is not None:
            self.test_target_transform = test_dataset.target_transform
            test_dataset.target_transform = None

        self.train_steps_patterns_assignment = train_steps_patterns_assignment
        self.test_steps_patterns_assignment = test_steps_patterns_assignment
        self.task_labels = task_labels

        if reproducibility_data is not None:
            self.train_steps_patterns_assignment = reproducibility_data['train']
            self.test_steps_patterns_assignment = reproducibility_data['test']
            self.task_labels = reproducibility_data['task_labels']

    def __len__(self):
        """
        Gets the number of steps this scenario it's made of.

        :return: The number of steps in this scenario.
        """
        return len(self.train_steps_patterns_assignment)

    def __getitem__(self, step_id: int):
        """
        Gets a steps given its step ID.

        :param step_id: The step ID.

        :return: The step instance associated to the given step ID.
        """
        return GenericStepInfo(self, step_id)

    def get_reproducibility_data(self):
        """
        Gets the data needed to reproduce this experiment.

        This data can be stored using the pickle module or some other mechanism.
        It can then be loaded by passing it as the ``reproducibility_data``
        parameter in the constructor.

        Child classes should get the reproducibility dictionary from super class
        and then merge their custom data before returning it.

        :return: A dictionary containing the data needed to reproduce the
        experiment.
        """
        train_steps = []
        for train_step_id in range(len(self.train_steps_patterns_assignment)):
            train_step = self.train_steps_patterns_assignment[train_step_id]
            train_steps.append(list(train_step))
        test_steps = []
        for test_step_id in range(len(self.test_steps_patterns_assignment)):
            test_step = self.test_steps_patterns_assignment[test_step_id]
            test_steps.append(list(test_step))
        return {'train': train_steps, 'test': test_steps, 'task_labels':
                list(self.task_labels)}


class AbstractStepInfo(ABC, Generic[TBaseScenario]):
    """
    Definition of a learning step. A learning step contains a set of patterns
    which has become available at a particular time instant. The content and
    size of a Step is defined by the specific benchmark that creates the
    step instance.

    For instance, a step of a New Classes scenario will contain all patterns
    belonging to a subset of classes of the original training set. A step of a
    New Instance scenario will contain patterns from previously seen classes.

    Steps of  Single Incremental Task (a.k.a. task-free) scenarios are usually
    called "batches" while in Multi Task scenarios a Step is usually associated
    to a "task". Finally, in a Multi Incremental Task scenario the Step may be
    composed by patterns from different tasks.
    """

    def __init__(
            self, scenario: TBaseScenario, current_step: int, n_steps: int,
            train_transform: Optional[Callable] = None,
            train_target_transform: Optional[Callable] = None,
            test_transform: Optional[Callable] = None,
            test_target_transform: Optional[Callable] = None,
            force_train_transformations: bool = False,
            force_test_transformations: bool = False,
            are_transformations_disabled: bool = False):
        """
        Creates an instance of the abstract step info given the base scenario,
        the current step ID, the overall number of steps, transformations and
        transformations flags.

        :param scenario: The base scenario.
        :param current_step: The current step, as an integer.
        :param n_steps: The overall number of steps in the base scenario.
        :param train_transform: The train transformation. Can be None.
        :param train_target_transform: The train targets transformation.
            Can be None.
        :param test_transform: The test transformation. Can be None.
        :param test_target_transform: The test targets transformation.
            Can be None.
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

        # scenario keeps a reference to the base scenario
        self.scenario: TBaseScenario = scenario

        # current_step is usually an incremental, 0-indexed, value used to
        # keep track of the current batch/task.
        self.current_step: int = current_step

        # n_steps is the overall amount of steps in the scenario
        self.n_steps: int = n_steps

        # Transformations are kept here so that the "enable" / "disable"
        # and "force train"/"force test" transformations-related methods can be
        # implemented
        self.train_transform: Optional[Callable] = train_transform
        self.test_transform: Optional[Callable] = test_transform
        self.train_target_transform: Optional[Callable] = train_target_transform
        self.test_target_transform: Optional[Callable] = test_target_transform

        self.force_train_transformations: bool = force_train_transformations
        self.force_test_transformations: bool = force_test_transformations
        self.are_transformations_disabled: bool = are_transformations_disabled

    @abstractmethod
    def _make_subset(self, is_train: bool, step: int, **kwargs) -> MTSingleSet:
        """
        Returns the train/test dataset given the step ID.

        :param is_train: If True, the training subset is returned. If False,
            the test subset will be returned instead.
        :param step: The step ID.
        :param kwargs: Other scenario-specific arguments. Subclasses may define
            some utility options.
        :return: The required train/test dataset.
        """
        pass

    @abstractmethod
    def disable_transformations(self) -> 'AbstractStepInfo[TBaseScenario]':
        """
        Returns a new step info instance in which transformations are disabled.
        The current instance is not affected. This is useful when there is a
        need to access raw data. Can be used when picking and storing
        rehearsal/replay patterns.

        :returns: A new step info in which transformations are disabled.
        """
        pass

    @abstractmethod
    def enable_transformations(self) -> 'AbstractStepInfo[TBaseScenario]':
        """
        Returns a new step info instance in which transformations are enabled.
        The current instance is not affected. When created, the step instance
        already has transformations enabled. This method can be used to
        re-enable transformations after a previous call to
        disable_transformations().

        :returns: A new step info in which transformations are enabled.
        """
        pass

    @abstractmethod
    def with_train_transformations(self) -> 'AbstractStepInfo[TBaseScenario]':
        """
        Returns a new step info instance in which train transformations are
        applied to both training and test sets. The current instance is not
        affected.

        :returns: A new step info in which train transformations are applied to
            both training and test sets.
        """
        pass

    @abstractmethod
    def with_test_transformations(self) -> 'AbstractStepInfo[TBaseScenario]':
        """
        Returns a new step info instance in which test transformations are
        applied to both training and test sets. The current instance is
        not affected. This is useful to get the accuracy on the training set
        without considering the usual training data augmentations.

        :returns: A new step info in which test transformations are applied to
            both training and test sets.
        """
        pass

    def current_training_set(self, **kwargs) -> MTSingleSet:
        """
        Gets the training set for the current step.

        :returns: The current step training set, as a tuple containing the
            Dataset and the task label.
        """
        return self.step_specific_training_set(self.current_step, **kwargs)

    def cumulative_training_sets(self, include_current_step: bool = True,
                                 **kwargs) -> MTMultipleSet:
        """
        Gets the list of cumulative training sets.

        :param include_current_step: If True, include the current step training
            set. Defaults to True.

        :returns: The cumulative training sets, as a list. Each element of the
            list is a tuple containing the Dataset and the task label.
        """
        if include_current_step:
            steps = range(0, self.current_step + 1)
        else:
            steps = range(0, self.current_step)
        return self._make_train_subsets(steps, **kwargs)

    def complete_training_sets(self, **kwargs) -> MTMultipleSet:
        """
        Gets the complete list of training sets.

        :returns: All the training sets, as a list. Each element of
            the list is a tuple containing the Dataset and the task label.
        """
        return self._make_train_subsets(
            list(range(0, self.n_steps)), **kwargs)

    def future_training_sets(self, **kwargs) -> MTMultipleSet:
        """
        Gets the "future" training set. That is, a dataset made of training
        patterns belonging to not-already-encountered steps.

        :returns: The future training sets, as a list. Each element of the
            list is a tuple containing the Dataset and the task label.
        """
        return self._make_train_subsets(
            range(self.current_step + 1, self.n_steps), **kwargs)

    def step_specific_training_set(self, step_id: int, **kwargs) -> MTSingleSet:
        """
        Gets the training set of a specific step, given its ID.

        :param step_id: The ID of the step.

        :returns: The required training set, as a tuple containing the Dataset
            and the task label.
        """
        return self._make_train_subsets(step_id, **kwargs)[0]

    def training_set_part(self, dataset_part: DatasetPart, **kwargs) \
            -> MTMultipleSet:
        """
        Gets the training subset of a specific part of the scenario.

        :returns: The training sets of the desired part, as a list. Each element
            of the list is a tuple containing the Dataset and the task label.
        """
        if dataset_part == DatasetPart.CURRENT:
            return [self.current_training_set(**kwargs)]
        if dataset_part == DatasetPart.CUMULATIVE:
            return self.cumulative_training_sets(
                include_current_step=True, **kwargs)
        if dataset_part == DatasetPart.OLD:
            return self.cumulative_training_sets(
                include_current_step=False, **kwargs)
        if dataset_part == DatasetPart.FUTURE:
            return self.future_training_sets(**kwargs)
        if dataset_part == DatasetPart.COMPLETE:
            return self.complete_training_sets(**kwargs)
        raise ValueError('Unsupported dataset part')

    def current_test_set(self, **kwargs) \
            -> MTSingleSet:
        """
        Gets the test set for the current step.

        :returns: The current test sets, as a tuple containing the Dataset and
            the task label.
        """
        return self.step_specific_test_set(self.current_step, **kwargs)

    def cumulative_test_sets(self, include_current_step: bool = True,
                             **kwargs) -> MTMultipleSet:
        """
        Gets the list of cumulative test sets.

        :param include_current_step: If True, include the current step training
            set. Defaults to True.

        :returns: The cumulative test sets, as a list. Each element of the
            list is a tuple containing the Dataset and the task label.
        """

        if include_current_step:
            steps = range(0, self.current_step + 1)
        else:
            steps = range(0, self.current_step)
        return self._make_test_subsets(steps, **kwargs)

    def complete_test_sets(self, **kwargs) -> MTMultipleSet:
        """
        Gets the complete list of test sets.

        :returns: All the test sets, as a list. Each element of
            the list is a tuple containing the Dataset and the task label.
        """
        return self._make_test_subsets(list(range(0, self.n_steps)), **kwargs)

    def future_test_sets(self, **kwargs) -> MTMultipleSet:
        """
        Gets the "future" test set. That is, a dataset made of test patterns
        belonging to not-already-encountered steps.

        :returns: The future test sets, as a list. Each element of the
            list is a tuple containing the Dataset and the task label.
        """
        return self._make_test_subsets(
            range(self.current_step + 1, self.n_steps), **kwargs)

    def step_specific_test_set(self, step_id: int, **kwargs) -> MTSingleSet:
        """
        Gets the test set of a specific step, given its ID.

        :param step_id: The ID of the step.

        :returns: The required test set, as a tuple containing the Dataset
            and the task label.
        """
        return self._make_test_subsets(step_id, **kwargs)[0]

    def test_set_part(self, dataset_part: DatasetPart,
                      **kwargs) -> MTMultipleSet:
        """
        Gets the test subset of a specific part of the scenario.

        :param dataset_part: The part of the scenario.

        :returns: The test sets of the desired part, as a list. Each element
            of the list is a tuple containing the Dataset and the task label.
        """
        if dataset_part == DatasetPart.CURRENT:
            return [self.current_test_set(**kwargs)]
        if dataset_part == DatasetPart.CUMULATIVE:
            return self.cumulative_test_sets(
                include_current_step=True, **kwargs)
        if dataset_part == DatasetPart.OLD:
            return self.cumulative_test_sets(
                include_current_step=False, **kwargs)
        if dataset_part == DatasetPart.FUTURE:
            return self.future_test_sets(**kwargs)
        if dataset_part == DatasetPart.COMPLETE:
            return self.complete_test_sets(**kwargs)
        raise ValueError('Unsupported dataset part')

    def _make_train_subsets(self, steps: Union[int, Sequence[int]], **kwargs) \
            -> Union[MTMultipleSet]:
        """
        Internal utility used to aggregate results from ``_make_subset``.

        :param steps: A list of step IDs whose training dataset must be fetched.
            Can also be a single step ID (as an integer).
        :param kwargs: Other ``_make_subset`` specific options.
        :return: A list of tuples. Each tuples contains the dataset and task
            label for that step (relative to the order defined in the steps
            parameter).
        """
        if isinstance(steps, int):  # Required single step
            return [self._make_subset(True, steps, **kwargs)]
        return [self._make_subset(True, step, **kwargs) for step in steps]

    def _make_test_subsets(self, steps: Union[int, Sequence[int]], **kwargs) \
            -> Union[MTMultipleSet]:
        """
        Internal utility used to aggregate results from ``_make_subset``.

        :param steps: A list of step IDs whose test dataset must be fetched.
            Can also be a single step ID (as an integer).
        :param kwargs: Other ``_make_subset`` specific options.
        :return: A list of tuples. Each tuples contains the dataset and task
            label for that step (relative to the order defined in the steps
            parameter).
        """
        if isinstance(steps, int):  # Required single step
            return [self._make_subset(False, steps, **kwargs)]
        return [self._make_subset(False, step, **kwargs) for step in steps]


class GenericStepInfo(AbstractStepInfo[GenericCLScenario[TrainSetWithTargets,
                                                         TestSetWithTargets]],
                      ABC, Generic[TrainSetWithTargets, TestSetWithTargets]):
    """
    Definition of a learning step based on a :class:`GenericCLScenario`
    instance.

    This step implementation uses the generic step-patterns assignment defined
    in the :class:`GenericCLScenario` instance. Instances of this class are
    usually obtained as the output of the iteration of the base scenario
    instance.
    """

    def __init__(self, scenario: GenericCLScenario[TrainSetWithTargets,
                                                   TestSetWithTargets],
                 current_step: int,
                 force_train_transformations: bool = False,
                 force_test_transformations: bool = False,
                 are_transformations_disabled: bool = False):
        """
        Creates an instance of a generic step info given the base generic
        scenario and the current step ID and transformations
        flags.

        :param scenario: The base generic scenario.
        :param current_step: The current step, as an integer.
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
        super(GenericStepInfo, self).__init__(
            scenario, current_step, len(scenario), scenario.train_transform,
            scenario.train_target_transform, scenario.test_transform,
            scenario.test_target_transform, force_train_transformations,
            force_test_transformations,
            are_transformations_disabled)

    def _get_task_label(self, step: int):
        """
        Returns the task label given the step ID.

        :param step: The step ID.

        :return: The task label of the step.
        """
        return self.scenario.task_labels[step]

    def _make_subset(self, is_train: bool, step: int,
                     bucket_classes=False, sort_classes=False,
                     sort_indexes=False, **kwargs) -> MTSingleSet:
        """
        Returns the train/test dataset given the step ID.

        :param is_train: If True, the training subset is returned. If False,
            the test subset will be returned instead.
        :param step: The step ID.
        :param bucket_classes: If True, dataset patterns will be grouped by
            class. Defaults to False.
        :param sort_classes: If True (and ``bucket_classes`` is True), class
            groups will be sorted by class ID (ascending). Defaults to False.
        :param sort_indexes: If True patterns will be ordered by their ID
            (ascending). If ``sort_classes`` and ``bucket_classes`` are both
            True, patterns will be sorted inside their groups.
            Defaults to False.
        :param kwargs: Other scenario-specific arguments. Subclasses may define
            some utility options.
        :return: The required train/test dataset.
        """
        if self.are_transformations_disabled:
            patterns_transformation = None
            targets_transformation = None
        elif self.force_test_transformations:
            patterns_transformation = \
                self.test_transform
            targets_transformation = \
                self.test_target_transform
        else:
            patterns_transformation = self.train_transform if is_train \
                else self.test_transform
            targets_transformation = self.train_target_transform if is_train \
                else self.test_target_transform

        if is_train:
            patterns_indexes = \
                self.scenario.train_steps_patterns_assignment[step]
            dataset = self.scenario.train_dataset
        else:
            patterns_indexes = \
                self.scenario.test_steps_patterns_assignment[step]
            dataset = self.scenario.test_dataset

        return TransformationSubset(
            dataset,
            grouped_and_ordered_indexes(
                dataset.targets, patterns_indexes,
                bucket_classes=bucket_classes,
                sort_classes=sort_classes,
                sort_indexes=sort_indexes),
            transform=patterns_transformation,
            target_transform=targets_transformation), self._get_task_label(step)

    def disable_transformations(self) -> \
            'GenericStepInfo[GenericCLScenario[TrainSetWithTargets, ' \
            'TestSetWithTargets]]':
        return GenericStepInfo(
            self.scenario, self.current_step,
            self.force_train_transformations,
            self.force_test_transformations, True
        )

    def enable_transformations(self) -> \
            'GenericStepInfo[GenericCLScenario[TrainSetWithTargets, ' \
            'TestSetWithTargets]]':
        return GenericStepInfo(
            self.scenario, self.current_step,
            self.force_train_transformations,
            self.force_test_transformations, False
        )

    def with_train_transformations(self) -> \
            'GenericStepInfo[GenericCLScenario[TrainSetWithTargets, ' \
            'TestSetWithTargets]]':
        return GenericStepInfo(
            self.scenario, self.current_step, True,
            False, self.are_transformations_disabled
        )

    def with_test_transformations(self) -> \
            'GenericStepInfo[GenericCLScenario[TrainSetWithTargets, ' \
            'TestSetWithTargets]]':
        return GenericStepInfo(
            self.scenario, self.current_step, False,
            True, self.are_transformations_disabled
        )
