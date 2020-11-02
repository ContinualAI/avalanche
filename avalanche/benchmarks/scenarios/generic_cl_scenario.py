from typing import Generic, TypeVar, Union, Sequence, Callable, Optional, \
    Dict, Any, Iterable, List, Set
from abc import ABC, abstractmethod
import copy

from avalanche.benchmarks.scenarios.generic_definitions import MTSingleSet, \
    MTMultipleSet, DatasetPart, TrainSetWithTargets, TestSetWithTargets
from avalanche.training.utils import TransformationSubset
from avalanche.benchmarks.utils import grouped_and_ordered_indexes

TBaseScenario = TypeVar('TBaseScenario')
TStepInfo = TypeVar('TStepInfo')
TGenericCLScenario = TypeVar('TGenericCLScenario', bound='GenericCLScenario')


class GenericCLScenario(Generic[TrainSetWithTargets, TestSetWithTargets,
                        TStepInfo]):
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
    def __init__(self: TGenericCLScenario,
                 original_train_dataset: TrainSetWithTargets,
                 original_test_dataset: TestSetWithTargets,
                 train_dataset: TrainSetWithTargets,
                 test_dataset: TestSetWithTargets,
                 train_steps_patterns_assignment: Sequence[Sequence[int]],
                 test_steps_patterns_assignment: Sequence[Sequence[int]],
                 task_labels: Sequence[int],
                 complete_test_set_only: bool = False,
                 reproducibility_data: Optional[Dict[str, Any]] = None,
                 step_factory: Callable[[TGenericCLScenario, int],
                                        TStepInfo] = None):
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
        :param complete_test_set_only: If True, only the complete test
            set will be returned from test set related methods of the linked
            :class:`GenericStepInfo` instances. This also means that the
            ``test_steps_patterns_assignment`` parameter can be a single element
            or even an empty list (in which case, the full set defined by
            the ``test_dataset`` parameter will be returned). The returned
            task label for the complete test set will be the first element
            of the ``task_labels`` parameter. Defaults to False, which means
            that ```train_steps_patterns_assignment`` and
            ``test_steps_patterns_assignment`` parameters must describe an equal
            amount of steps.
        :param reproducibility_data: If not None, overrides the
            ``train/test_steps_patterns_assignment`` and ``task_labels``
            parameters. This is usually a dictionary containing data used to
            reproduce a specific experiment. One can use the
            ``get_reproducibility_data`` method to get (and even distribute)
            the experiment setup so that it can be loaded by passing it as this
            parameter. In this way one can be sure that the same specific
            experimental setup is being used (for reproducibility purposes).
            Beware that, in order to reproduce an experiment, the same train and
            test datasets must be used. Defaults to None.
        :param step_factory: If not None, a callable that, given the scenario
            instance and the step ID, returns a step info instance. This
            parameter is usually used in subclasses (when invoking the super
            constructor) to specialize the step info class. Defaults to None,
            which means that the :class:`GenericStepInfo` constructor will be
            used.
        """

        self.original_train_dataset: TrainSetWithTargets = \
            original_train_dataset
        """ The original training set. """

        self.original_test_dataset: TestSetWithTargets = original_test_dataset
        """ The original test set. """

        self.train_dataset: TrainSetWithTargets = train_dataset
        """ The training set used to generate the incremental steps. """

        self.test_dataset: TestSetWithTargets = test_dataset
        """ The test set used to generate the incremental steps. """

        self.train_steps_patterns_assignment: Sequence[Sequence[int]]
        """ A list containing which training patterns are assigned to each step.
        Patterns are identified by their id w.r.t. the dataset found in the 
        train_dataset field. """

        self.test_steps_patterns_assignment: Sequence[Sequence[int]]
        """ A list containing which test patterns are assigned to each step.
        Patterns are identified by their id w.r.t. the dataset found in the 
        test_dataset field """

        self.task_labels: Sequence[int] = task_labels
        """ The task label of each step. """

        self.slice_ids: Optional[List[int]] = None
        """ Describes which steps are contained in the current scenario slice. 
        Can be None, which means that this object is the original scenario. """

        # Steal transforms from the datasets, that is, copy the reference to the
        # transformation functions, and set to None the fields in the
        # respective Dataset instances. This will allow us to disable
        # transformations (useful while managing rehearsal) or even apply test
        # transforms to train patterns (useful when if testing on the training
        # sets, as test transforms usually don't contain data augmentation
        # transforms)
        self.train_transform = None
        self.train_target_transform = None
        self.test_transform = None
        self.test_target_transform = None

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

        self.train_steps_patterns_assignment: Sequence[Sequence[int]] = \
            train_steps_patterns_assignment
        self.test_steps_patterns_assignment: Sequence[Sequence[int]] = \
            test_steps_patterns_assignment
        self.task_labels: Sequence[int] = task_labels

        self.complete_test_set_only: bool = bool(complete_test_set_only)
        """ If True, only the complete test set will be returned from step info
        instances. """

        if reproducibility_data is not None:
            self.train_steps_patterns_assignment = reproducibility_data['train']
            self.test_steps_patterns_assignment = reproducibility_data['test']
            self.task_labels = reproducibility_data['task_labels']
            self.complete_test_set_only = \
                reproducibility_data['complete_test_only']

        self.n_steps: int = len(self.train_steps_patterns_assignment)
        """  The number of incremental steps this scenario is made of. """

        if step_factory is None:
            step_factory = GenericStepInfo

        self._step_factory: Callable[[TGenericCLScenario, int], TStepInfo] = \
            step_factory

        if self.complete_test_set_only:
            if len(self.test_steps_patterns_assignment) > 1:
                raise ValueError(
                    'complete_test_set_only is True, but '
                    'test_steps_patterns_assignment contains more than one '
                    'element')
        elif len(self.train_steps_patterns_assignment) != \
                len(self.test_steps_patterns_assignment):
            raise ValueError('There must be the same amount of train and '
                             'test steps')

        if len(self.train_steps_patterns_assignment) != len(self.task_labels):
            raise ValueError('There must be the same number of train steps '
                             'and task labels')

    def __len__(self) -> int:
        """
        Gets the number of steps this scenario it's made of.

        :return: The number of steps in this scenario.
        """
        if self.slice_ids is None:
            return len(self.train_steps_patterns_assignment)
        else:
            return len(self.slice_ids)

    def __getitem__(self: TGenericCLScenario,
                    step_idx: Union[int, slice, Iterable[int]]) -> \
            Union[TGenericCLScenario, TStepInfo]:
        """
        Gets a step given its step index (or a scenario slice given the step
        order).

        :param step_idx: An int describing the step index or an iterable/slice
            object describing a slice of this scenario.

        :return: The step instance associated to the given step index or
            a sliced scenario instance (see parameter step_idx).
        """
        if isinstance(step_idx, int):
            if step_idx < len(self):
                if self.slice_ids is None:
                    return self._step_factory(self, step_idx)
                else:
                    return self._step_factory(self, self.slice_ids[step_idx])
            raise IndexError('Step index out of bounds' + str(int(step_idx)))
        else:
            return self._create_slice(step_idx)

    def get_reproducibility_data(self) -> Dict[str, Any]:
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
        return {'train': train_steps, 'test': test_steps,
                'task_labels': list(self.task_labels),
                'complete_test_only': bool(self.complete_test_set_only)}

    def get_classes_timeline(self, current_step: int):
        """
        Returns the classes timeline for a this scenario.

        Given a step ID, this method returns the classes in this step,
        previously seen classes, the cumulative class list and a list
        of classes that will be encountered in next steps.

        :param current_step: The reference step ID.
        :return: A tuple composed of four lists: the first list contains the
            IDs of classes in this step, the second contains IDs of classes seen
            in previous steps, the third returns a cumulative list of classes
            (that is, the union of the first two list) while the last one
            returns a list of classes that will be encountered in next steps.
        """
        train_dataset: TrainSetWithTargets
        train_steps_patterns_assignment: Sequence[Sequence[int]]

        class_set_current_step = self.classes_in_step[current_step]

        classes_in_this_step = list(class_set_current_step)

        class_set_prev_steps = set()
        for step_id in range(0, current_step):
            class_set_prev_steps.update(self.classes_in_step[step_id])
        previous_classes = list(class_set_prev_steps)

        classes_seen_so_far = \
            list(class_set_current_step.union(class_set_prev_steps))

        class_set_future_steps = set()
        for step_id in range(current_step, self.n_steps):
            class_set_prev_steps.update(self.classes_in_step[step_id])
        future_classes = list(class_set_future_steps)

        return (classes_in_this_step, previous_classes, classes_seen_so_far,
                future_classes)

    @property
    def classes_in_step(self) -> Sequence[Set[int]]:
        """ A list that, for each step (identified by its index/ID),
        stores a set of the (optionally remapped) IDs of classes of patterns
        assigned to that step. """
        return LazyClassesInSteps(self)

    def _create_slice(self: TGenericCLScenario,
                      steps_slice: Union[int, slice, Iterable[int]]) \
            -> TGenericCLScenario:
        """
        Creates a sliced version of this scenario.

        In its base version, a shallow copy of this scenario is created and
        then its ``slice_ids`` field is adapted.

        :param steps_slice: The slice to use.
        :return: A sliced version of this scenario.
        """
        scenario_copy = copy.copy(self)
        slice_steps = _get_slice_ids(steps_slice, len(self))

        if self.slice_ids is None:
            scenario_copy.slice_ids = slice_steps
        else:
            scenario_copy.slice_ids = [self.slice_ids[x] for x in slice_steps]
        return scenario_copy


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
            classes_in_this_step: Sequence[int],
            previous_classes: Sequence[int],
            classes_seen_so_far: Sequence[int],
            future_classes: Optional[Sequence[int]],
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

        if force_test_transformations and force_train_transformations:
            raise ValueError(
                'Error in force_train/test_transformations arguments.'
                'Can\'t be both True.')

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

        self.classes_in_this_step: Sequence[int] = classes_in_this_step
        """ The list of classes in this step """

        self.previous_classes: Sequence[int] = previous_classes
        """ The list of classes in previous steps """

        self.classes_seen_so_far: Sequence[int] = classes_seen_so_far
        """ List of classes of current and previous steps """

        self.future_classes: Optional[Sequence[int]] = future_classes
        """ The list of classes of next steps """

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


class GenericStepInfo(AbstractStepInfo[TGenericCLScenario]):
    """
    Definition of a learning step based on a :class:`GenericCLScenario`
    instance.

    This step implementation uses the generic step-patterns assignment defined
    in the :class:`GenericCLScenario` instance. Instances of this class are
    usually obtained as the output of the iteration of the base scenario
    instance.
    """

    def __init__(self, scenario: TGenericCLScenario,
                 current_step: int,
                 force_train_transformations: bool = False,
                 force_test_transformations: bool = False,
                 are_transformations_disabled: bool = False,
                 transformation_step_factory: Optional[Callable] = None):
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

        (classes_in_this_step, previous_classes, classes_seen_so_far,
         future_classes) = scenario.get_classes_timeline(current_step)

        super(GenericStepInfo, self).__init__(
            scenario, current_step, len(scenario), classes_in_this_step,
            previous_classes, classes_seen_so_far, future_classes,
            scenario.train_transform, scenario.train_target_transform,
            scenario.test_transform, scenario.test_target_transform,
            force_train_transformations, force_test_transformations,
            are_transformations_disabled)

        self.transformation_step_factory = transformation_step_factory
        if transformation_step_factory is None:
            self.transformation_step_factory = GenericStepInfo

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
            dataset = self.scenario.train_dataset
            patterns_indexes = \
                self.scenario.train_steps_patterns_assignment[step]
        else:
            dataset = self.scenario.test_dataset
            if len(self.scenario.test_steps_patterns_assignment) == 0:
                # self.scenario.complete_test_set_only is True (otherwise
                # test_steps_patterns_assignment couldn't be empty)
                # This means we have to return the entire test_dataset as-is.
                patterns_indexes = None
            else:
                patterns_indexes = \
                    self.scenario.test_steps_patterns_assignment[step]

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
        return self.transformation_step_factory(
            self.scenario, self.current_step,
            self.force_train_transformations,
            self.force_test_transformations, True
        )

    def enable_transformations(self) -> \
            'GenericStepInfo[GenericCLScenario[TrainSetWithTargets, ' \
            'TestSetWithTargets]]':
        return self.transformation_step_factory(
            self.scenario, self.current_step,
            self.force_train_transformations,
            self.force_test_transformations, False
        )

    def with_train_transformations(self) -> \
            'GenericStepInfo[GenericCLScenario[TrainSetWithTargets, ' \
            'TestSetWithTargets]]':
        return self.transformation_step_factory(
            self.scenario, self.current_step, True,
            False, self.are_transformations_disabled
        )

    def with_test_transformations(self) -> \
            'GenericStepInfo[GenericCLScenario[TrainSetWithTargets, ' \
            'TestSetWithTargets]]':
        return self.transformation_step_factory(
            self.scenario, self.current_step, False,
            True, self.are_transformations_disabled
        )

    def _make_test_subsets(self, steps: Union[int, Sequence[int]], **kwargs) \
            -> Union[MTMultipleSet]:

        if self.scenario.complete_test_set_only:
            return [self._make_subset(False, 0, **kwargs)]

        return super()._make_test_subsets(steps, **kwargs)


def _get_slice_ids(slice_definition: Union[int, slice, Iterable[int]],
                   sliceable_len: int) -> List[int]:
    # Obtain steps list from slice object (or any iterable)
    steps_list: List[int]
    if isinstance(slice_definition, slice):
        steps_list = list(
            range(*slice_definition.indices(sliceable_len)))
    elif isinstance(slice_definition, int):
        steps_list = [slice_definition]
    elif hasattr(slice_definition, 'shape') and \
            len(getattr(slice_definition, 'shape')) == 0:
        steps_list = [int(slice_definition)]
    else:
        steps_list = list(slice_definition)

    # Check step id(s) boundaries
    if max(steps_list) >= sliceable_len:
        raise ValueError('Step index out of range: ' + str(max(steps_list)))

    if min(steps_list) < 0:
        raise ValueError('Step index out of range: ' + str(min(steps_list)))

    return steps_list


class LazyClassesInSteps(Sequence[Set[int]]):
    def __init__(self, scenario: GenericCLScenario):
        self._scenario = scenario

    def __len__(self):
        return len(self._scenario)

    def __getitem__(self, step_id) -> Set[int]:
        return set(
            [self._scenario.train_dataset.targets[pattern_idx] for pattern_idx
             in self._scenario.train_steps_patterns_assignment[step_id]])

    def __str__(self):
        return '[' + \
               ', '.join([str(self[idx]) for idx in range(len(self))]) + \
               ']'



__all__ = ['GenericStepInfo', 'GenericCLScenario', 'AbstractStepInfo',
           'TBaseScenario', 'TStepInfo', 'TGenericCLScenario']
