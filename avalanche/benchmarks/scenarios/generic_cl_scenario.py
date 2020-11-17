import copy
from abc import ABC
from typing import Generic, TypeVar, Union, Sequence, Callable, Optional, \
    Dict, Any, Iterable, List, Set

from avalanche.benchmarks.scenarios.generic_definitions import \
    TrainSetWithTargets, TestSetWithTargets, \
    TStepInfo, IScenarioStream, TScenarioStream, IStepInfo, TScenario
from avalanche.training.utils import TransformationDataset, TransformationSubset

TGenericCLScenario = TypeVar('TGenericCLScenario', bound='GenericCLScenario')
TGenericStepInfo = TypeVar('TGenericStepInfo', bound='GenericStepInfo')
TGenericScenarioStream = TypeVar('TGenericScenarioStream',
                                 bound='GenericScenarioStream')


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
                 step_factory: Callable[['GenericScenarioStream', int],
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
        """
        If True, only the complete test set will be returned from step info
        instances.
        
        This flag is usually set to true in scenarios where having one separate
        test set aligned to each training step is impossible or doesn't make
        sense from a semantic point of view.
        """

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

        self.step_factory: Callable[[TGenericScenarioStream, int],
                                    TStepInfo] = step_factory

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

        self.train_stream: GenericScenarioStream[
            TStepInfo, TGenericCLScenario] = GenericScenarioStream('train',
                                                                   self)
        """
        The stream used to obtain the training steps. This stream can be sliced
        in order to obtain a subset of this stream.
        """

        self.test_stream: GenericScenarioStream[
            TStepInfo, TGenericCLScenario] = GenericScenarioStream('test',
                                                                   self)
        """
        The stream used to obtain the test steps. This stream can be sliced
        in order to obtain a subset of this stream.
        
        Beware that, in certain scenarios, this stream may contain a single
        element. Check the ``complete_test_set_only`` field for more details.
        """

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


class GenericScenarioStream(Generic[TStepInfo, TGenericCLScenario],
                            IScenarioStream[TGenericCLScenario, TStepInfo],
                            Sequence[TStepInfo]):

    def __init__(self: TGenericScenarioStream,
                 name: str,
                 scenario: TGenericCLScenario,
                 *,
                 slice_ids: List[int] = None):
        self.slice_ids: Optional[List[int]] = slice_ids
        """
        Describes which steps are contained in the current stream slice. 
        Can be None, which means that this object is the original stream. """

        self.name: str = name
        self.scenario = scenario

    def __len__(self) -> int:
        """
        Gets the number of steps this scenario it's made of.

        :return: The number of steps in this scenario.
        """
        if self.slice_ids is None:
            if self.name == 'train':
                return len(self.scenario.train_steps_patterns_assignment)
            elif self.scenario.complete_test_set_only:
                return 1
            else:
                return len(self.scenario.test_steps_patterns_assignment)
        else:
            return len(self.slice_ids)

    def __getitem__(self, step_idx: Union[int, slice, Iterable[int]]) -> \
            Union[TStepInfo, TScenarioStream]:
        """
        Gets a step given its step index (or a stream slice given the step
        order).

        :param step_idx: An int describing the step index or an iterable/slice
            object describing a slice of this stream.

        :return: The step instance associated to the given step index or
            a sliced stream instance.
        """
        if isinstance(step_idx, int):
            if step_idx < len(self):
                if self.slice_ids is None:
                    return self.scenario.step_factory(self, step_idx)
                else:
                    return self.scenario.step_factory(self,
                                                      self.slice_ids[step_idx])
            raise IndexError('Step index out of bounds' + str(int(step_idx)))
        else:
            return self._create_slice(step_idx)

    def _create_slice(self: TGenericScenarioStream,
                      steps_slice: Union[int, slice, Iterable[int]]) \
            -> TScenarioStream:
        """
        Creates a sliced version of this stream.

        In its base version, a shallow copy of this stream is created and
        then its ``slice_ids`` field is adapted.

        :param steps_slice: The slice to use.
        :return: A sliced version of this stream.
        """
        stream_copy = copy.copy(self)
        slice_steps = _get_slice_ids(steps_slice, len(self))

        if self.slice_ids is None:
            stream_copy.slice_ids = slice_steps
        else:
            stream_copy.slice_ids = [self.slice_ids[x] for x in slice_steps]
        return stream_copy


class LazyClassesInSteps(Sequence[Set[int]]):
    def __init__(self, scenario: GenericCLScenario):
        self._scenario = scenario

    def __len__(self):
        return len(self._scenario.train_stream)

    def __getitem__(self, step_id) -> Set[int]:
        return set(
            [self._scenario.train_dataset.targets[pattern_idx] for pattern_idx
             in self._scenario.train_steps_patterns_assignment[step_id]])

    def __str__(self):
        return '[' + \
               ', '.join([str(self[idx]) for idx in range(len(self))]) + \
               ']'


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
        raise IndexError('Step index out of range: ' + str(max(steps_list)))

    if min(steps_list) < 0:
        raise IndexError('Step index out of range: ' + str(min(steps_list)))

    return steps_list


class AbstractStepInfo(IStepInfo[TScenario, TScenarioStream], ABC):
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
            self: TStepInfo,
            origin_stream: TScenarioStream,
            current_step: int,
            classes_in_this_step: Sequence[int],
            previous_classes: Sequence[int],
            classes_seen_so_far: Sequence[int],
            future_classes: Optional[Sequence[int]]):
        """
        Creates an instance of the abstract step info given the scenario stream,
        the current step ID and data about the classes timeline.

        :param origin_stream: The stream from which this step was obtained.
        :param current_step: The current step ID, as an integer.
        :param classes_in_this_step: The list of classes in this step.
        :param previous_classes: The list of classes in previous steps.
        :param classes_seen_so_far: List of classes of current and previous
            steps.
        :param future_classes: The list of classes of next steps.
        """

        self.origin_stream: TScenarioStream = origin_stream

        # scenario keeps a reference to the base scenario
        self.scenario: TScenario = origin_stream.scenario

        # current_step is usually an incremental, 0-indexed, value used to
        # keep track of the current batch/task.
        self.current_step: int = current_step

        self.classes_in_this_step: Sequence[int] = classes_in_this_step
        """ The list of classes in this step """

        self.previous_classes: Sequence[int] = previous_classes
        """ The list of classes in previous steps """

        self.classes_seen_so_far: Sequence[int] = classes_seen_so_far
        """ List of classes of current and previous steps """

        self.future_classes: Optional[Sequence[int]] = future_classes
        """ The list of classes of next steps """


class GenericStepInfo(AbstractStepInfo[TGenericCLScenario,
                                       GenericScenarioStream[
                                           TGenericStepInfo,
                                           TGenericCLScenario]]):
    """
    Definition of a learning step based on a :class:`GenericCLScenario`
    instance.

    This step implementation uses the generic step-patterns assignment defined
    in the :class:`GenericCLScenario` instance. Instances of this class are
    usually obtained from a scenario stream.
    """

    def __init__(self: TGenericStepInfo,
                 origin_stream: GenericScenarioStream[TGenericStepInfo,
                                                      TGenericCLScenario],
                 current_step: int):
        """
        Creates an instance of a generic step info given the stream from this
        step was taken and and the current step ID.

        :param origin_stream: The stream from which this step was obtained.
        :param current_step: The current step ID, as an integer.
        """

        (classes_in_this_step, previous_classes, classes_seen_so_far,
         future_classes) = origin_stream.scenario.get_classes_timeline(
            current_step)

        super(GenericStepInfo, self).__init__(
            origin_stream, current_step, classes_in_this_step,
            previous_classes, classes_seen_so_far, future_classes)

    def _get_task_label(self, step: int):
        """
        Returns the task label given the step ID.

        :param step: The step ID.

        :return: The task label of the step.
        """
        return self.scenario.task_labels[step]

    @property
    def dataset(self) -> TransformationDataset:
        if self._is_train():
            dataset = self.scenario.train_dataset
            patterns_indexes = \
                self.scenario.train_steps_patterns_assignment[self.current_step]
        else:
            dataset = self.scenario.test_dataset
            if self.scenario.complete_test_set_only:
                patterns_indexes = None
            else:
                patterns_indexes = self.scenario.test_steps_patterns_assignment[
                    self.current_step]

        # TODO: solve transformation issue
        return TransformationSubset(dataset, patterns_indexes)

    @property
    def task_label(self) -> int:
        if self._is_train():
            return self.scenario.task_labels[self.current_step]
        else:
            if self.scenario.complete_test_set_only:
                return self.scenario.task_labels[0]
            else:
                return self.scenario.task_labels[self.current_step]

    def _is_train(self):
        return self.origin_stream.name == 'train'


__all__ = ['TGenericCLScenario', 'GenericCLScenario', 'GenericScenarioStream',
           'GenericStepInfo', 'AbstractStepInfo']
