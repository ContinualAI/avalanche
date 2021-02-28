import copy
from abc import ABC
from typing import Generic, TypeVar, Union, Sequence, Callable, Optional, \
    Dict, Any, Iterable, List, Set

from avalanche.benchmarks.scenarios.generic_definitions import \
    TExperience, IScenarioStream, TScenarioStream, IExperience, TScenario, \
    TrainSet, TestSet
from avalanche.benchmarks.utils import AvalancheDataset, \
    AvalancheSubset

TGenericCLScenario = TypeVar('TGenericCLScenario', bound='GenericCLScenario')
TGenericExperience = TypeVar('TGenericExperience', bound='GenericExperience')
TGenericScenarioStream = TypeVar('TGenericScenarioStream',
                                 bound='GenericScenarioStream')


class GenericCLScenario(Generic[TrainSet, TestSet, TExperience]):
    """
    Base implementation of a Continual Learning scenario. A Continual Learning
    scenario is defined by a sequence of experiences (batches or tasks depending
    on the terminology), with each experience containing the training (or test)
    data that becomes available at a certain time instant.

    From a practical point of view, this means that we usually have to define
    two datasets (training and test), and some way to assign the patterns
    contained in these datasets to each experience.

    This assignment is usually made in children classes, with this class serving
    as the more general implementation. This class handles the most simple type
    of assignment: each experience is defined by a list of patterns (identified
    by their indexes) contained in that experience.
    """
    def __init__(self: TGenericCLScenario,
                 original_train_dataset: TrainSet,
                 original_test_dataset: TestSet,
                 train_dataset: AvalancheDataset,
                 test_dataset: AvalancheDataset,
                 train_exps_patterns_assignment: Sequence[Sequence[int]],
                 test_exps_patterns_assignment: Sequence[Sequence[int]],
                 task_labels: Sequence[List[int]],
                 pattern_train_task_labels: Sequence[int],
                 pattern_test_task_labels: Sequence[int],
                 complete_test_set_only: bool = False,
                 reproducibility_data: Optional[Dict[str, Any]] = None,
                 experience_factory: Callable[['GenericScenarioStream', int],
                                              TExperience] = None):
        """
        Creates an instance of a Continual Learning scenario.

        The scenario is defined by the train and test datasets plus the
        assignment of patterns to experiences (batches/tasks).

        :param train_dataset: The training dataset. The dataset must be a
            subclass of :class:`AvalancheDataset`. For instance, one can
            use the datasets from the torchvision package like that:
            ``train_dataset=AvalancheDataset(torchvision_dataset)``.
        :param test_dataset: The test dataset. The dataset must be a
            subclass of :class:`AvalancheDataset`. For instance, one can
            use the datasets from the torchvision package like that:
            ``test_dataset=AvalancheDataset(torchvision_dataset)``.
        :param train_exps_patterns_assignment: A list of experiences. Each
            experience is in turn defined by a list of integers describing the
            pattern index inside the training dataset.
        :param test_exps_patterns_assignment: A list of experiences. Each
            experience is in turn defined by a list of integers describing the
            pattern index inside the test dataset.
        :param task_labels: The mapping from experience IDs to task labels,
            usually as a list of integers.
        :param pattern_train_task_labels: The list of task labels of each
            pattern in the `train_dataset`.
        :param pattern_test_task_labels: The list of task labels of each
            pattern in the `test_dataset`.
        :param complete_test_set_only: If True, only the complete test
            set will be returned from test set related methods of the linked
            :class:`GenericExperience` instances. This also means that the
            ``test_exps_patterns_assignment`` parameter can be a single element
            or even an empty list (in which case, the full set defined by
            the ``test_dataset`` parameter will be returned). The returned
            task label for the complete test set will be the first element
            of the ``task_labels`` parameter. Defaults to False, which means
            that ```train_exps_patterns_assignment`` and
            ``test_exps_patterns_assignment`` parameters must describe an equal
            amount of experiences.
        :param reproducibility_data: If not None, overrides the
            ``train/test_exps_patterns_assignment`` and ``task_labels``
            parameters. This is usually a dictionary containing data used to
            reproduce a specific experiment. One can use the
            ``get_reproducibility_data`` method to get (and even distribute)
            the experiment setup so that it can be loaded by passing it as this
            parameter. In this way one can be sure that the same specific
            experimental setup is being used (for reproducibility purposes).
            Beware that, in order to reproduce an experiment, the same train and
            test datasets must be used. Defaults to None.
        :param experience_factory: If not None, a callable that, given the
            scenario instance and the experience ID, returns a experience
            instance. This parameter is usually used in subclasses (when
            invoking the super constructor) to specialize the experience class.
            Defaults to None, which means that the :class:`GenericExperience`
            constructor will be used.
        """

        self.original_train_dataset: TrainSet = original_train_dataset
        """ The original training set. """

        self.original_test_dataset: TestSet = original_test_dataset
        """ The original test set. """

        self.train_exps_patterns_assignment: Sequence[Sequence[int]]
        """ A list containing which training patterns are assigned to each 
        experience. Patterns are identified by their id w.r.t. the dataset found
        in the train_dataset field. """

        self.test_exps_patterns_assignment: Sequence[Sequence[int]]
        """ A list containing which test patterns are assigned to each
        experience. Patterns are identified by their id w.r.t. the dataset found
        in the test_dataset field """

        self.task_labels: Sequence[List[int]] = task_labels
        """ The task label of each experience. """

        self.pattern_train_task_labels: Sequence[int] = \
            pattern_train_task_labels
        """ The task label of each pattern in the training dataset. """

        self.pattern_test_task_labels: Sequence[int] = pattern_test_task_labels
        """ The task label of each pattern in the test dataset. """

        self.train_exps_patterns_assignment: Sequence[Sequence[int]] = \
            train_exps_patterns_assignment
        self.test_exps_patterns_assignment: Sequence[Sequence[int]] = \
            test_exps_patterns_assignment

        self.complete_test_set_only: bool = bool(complete_test_set_only)
        """
        If True, only the complete test set will be returned from experience
        instances.
        
        This flag is usually set to True in scenarios where having one separate
        test set aligned to each training experience is impossible or doesn't
        make sense from a semantic point of view.
        """

        if reproducibility_data is not None:
            self.train_exps_patterns_assignment = reproducibility_data['train']
            self.test_exps_patterns_assignment = reproducibility_data['test']
            self.task_labels = reproducibility_data['task_labels']
            self.pattern_train_task_labels = reproducibility_data[
                'pattern_train_task_labels']
            self.pattern_test_task_labels = reproducibility_data[
                'pattern_test_task_labels']
            self.complete_test_set_only = \
                reproducibility_data['complete_test_only']

        self.n_experiences: int = len(self.train_exps_patterns_assignment)
        """  The number of incremental experiences this scenario is made of. """

        if experience_factory is None:
            experience_factory = GenericExperience

        self.experience_factory: Callable[[TGenericScenarioStream, int],
                                          TExperience] = experience_factory

        if self.complete_test_set_only:
            if len(self.test_exps_patterns_assignment) > 1:
                raise ValueError(
                    'complete_test_set_only is True, but '
                    'test_exps_patterns_assignment contains more than one '
                    'element')
        elif len(self.train_exps_patterns_assignment) != \
                len(self.test_exps_patterns_assignment):
            raise ValueError('There must be the same amount of train and '
                             'test experiences')

        if len(self.train_exps_patterns_assignment) != len(self.task_labels):
            raise ValueError('There must be the same number of train '
                             'experiences and task labels')

        self.train_dataset: AvalancheDataset = AvalancheDataset(
            train_dataset, task_labels=self.pattern_train_task_labels)
        """ The training set used to generate the incremental experiences. """

        self.test_dataset: AvalancheDataset = AvalancheDataset(
            test_dataset, task_labels=self.pattern_test_task_labels)
        """ The test set used to generate the incremental experiences. """

        self.train_stream: GenericScenarioStream[
            TExperience, TGenericCLScenario] = GenericScenarioStream('train',
                                                                     self)
        """
        The stream used to obtain the training experiences. 
        This stream can be sliced in order to obtain a subset of this stream.
        """

        self.test_stream: GenericScenarioStream[
            TExperience, TGenericCLScenario] = GenericScenarioStream('test',
                                                                     self)
        """
        The stream used to obtain the test experiences. This stream can be 
        sliced in order to obtain a subset of this stream.
        
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
        train_exps = []
        for train_exp_id in range(len(self.train_exps_patterns_assignment)):
            train_exp = self.train_exps_patterns_assignment[train_exp_id]
            train_exps.append(list(train_exp))
        test_exps = []
        for test_exp_id in range(len(self.test_exps_patterns_assignment)):
            test_exp = self.test_exps_patterns_assignment[test_exp_id]
            test_exps.append(list(test_exp))
        return {'train': train_exps, 'test': test_exps,
                'task_labels': list(self.task_labels),
                'complete_test_only': bool(self.complete_test_set_only),
                'pattern_train_task_labels': list(
                    self.pattern_train_task_labels),
                'pattern_test_task_labels': list(self.pattern_test_task_labels)}

    def get_classes_timeline(self, current_experience: int):
        """
        Returns the classes timeline for a this scenario.

        Given a experience ID, this method returns the classes in this
        experience, previously seen classes, the cumulative class list and a
        list of classes that will be encountered in next experiences.

        :param current_experience: The reference experience ID.
        :return: A tuple composed of four lists: the first list contains the
            IDs of classes in this experience, the second contains IDs of
            classes seen in previous experiences, the third returns a cumulative
            list of classes (that is, the union of the first two list) while the
            last one returns a list of classes that will be encountered in next
            experiences.
        """
        train_exps_patterns_assignment: Sequence[Sequence[int]]

        class_set_current_exp = self.classes_in_experience[current_experience]

        classes_in_this_exp = list(class_set_current_exp)

        class_set_prev_exps = set()
        for exp_id in range(0, current_experience):
            class_set_prev_exps.update(self.classes_in_experience[exp_id])
        previous_classes = list(class_set_prev_exps)

        classes_seen_so_far = \
            list(class_set_current_exp.union(class_set_prev_exps))

        class_set_future_exps = set()
        for exp_id in range(current_experience, self.n_experiences):
            class_set_prev_exps.update(self.classes_in_experience[exp_id])
        future_classes = list(class_set_future_exps)

        return (classes_in_this_exp, previous_classes, classes_seen_so_far,
                future_classes)

    @property
    def classes_in_experience(self) -> Sequence[Set[int]]:
        """ A list that, for each experience (identified by its index/ID),
        stores a set of the (optionally remapped) IDs of classes of patterns
        assigned to that experience. """
        return LazyClassesInExps(self)


class GenericScenarioStream(Generic[TExperience, TGenericCLScenario],
                            IScenarioStream[TGenericCLScenario, TExperience],
                            Sequence[TExperience]):

    def __init__(self: TGenericScenarioStream,
                 name: str,
                 scenario: TGenericCLScenario,
                 *,
                 slice_ids: List[int] = None):
        self.slice_ids: Optional[List[int]] = slice_ids
        """
        Describes which experiences are contained in the current stream slice. 
        Can be None, which means that this object is the original stream. """

        self.name: str = name
        self.scenario = scenario

    def __len__(self) -> int:
        """
        Gets the number of experiences this scenario it's made of.

        :return: The number of experiences in this scenario.
        """
        if self.slice_ids is None:
            if self.name == 'train':
                return len(self.scenario.train_exps_patterns_assignment)
            elif self.scenario.complete_test_set_only:
                return 1
            else:
                return len(self.scenario.test_exps_patterns_assignment)
        else:
            return len(self.slice_ids)

    def __getitem__(self, exp_idx: Union[int, slice, Iterable[int]]) -> \
            Union[TExperience, TScenarioStream]:
        """
        Gets a experience given its experience index (or a stream slice given
        the experience order).

        :param exp_idx: An int describing the experience index or an
            iterable/slice object describing a slice of this stream.

        :return: The experience instance associated to the given experience
            index or a sliced stream instance.
        """
        if isinstance(exp_idx, int):
            if exp_idx < len(self):
                if self.slice_ids is None:
                    return self.scenario.experience_factory(self, exp_idx)
                else:
                    return self.scenario.experience_factory(
                        self, self.slice_ids[exp_idx])
            raise IndexError('Experience index out of bounds' +
                             str(int(exp_idx)))
        else:
            return self._create_slice(exp_idx)

    def _create_slice(self: TGenericScenarioStream,
                      exps_slice: Union[int, slice, Iterable[int]]) \
            -> TScenarioStream:
        """
        Creates a sliced version of this stream.

        In its base version, a shallow copy of this stream is created and
        then its ``slice_ids`` field is adapted.

        :param exps_slice: The slice to use.
        :return: A sliced version of this stream.
        """
        stream_copy = copy.copy(self)
        slice_exps = _get_slice_ids(exps_slice, len(self))

        if self.slice_ids is None:
            stream_copy.slice_ids = slice_exps
        else:
            stream_copy.slice_ids = [self.slice_ids[x] for x in slice_exps]
        return stream_copy


class LazyClassesInExps(Sequence[Set[int]]):
    def __init__(self, scenario: GenericCLScenario):
        self._scenario = scenario

    def __len__(self):
        return len(self._scenario.train_stream)

    def __getitem__(self, exp_id) -> Set[int]:
        return set(
            [self._scenario.train_dataset.targets[pattern_idx]
             for pattern_idx
             in self._scenario.train_exps_patterns_assignment[exp_id]])

    def __str__(self):
        return '[' + \
               ', '.join([str(self[idx]) for idx in range(len(self))]) + \
               ']'


def _get_slice_ids(slice_definition: Union[int, slice, Iterable[int]],
                   sliceable_len: int) -> List[int]:
    # Obtain experiences list from slice object (or any iterable)
    exps_list: List[int]
    if isinstance(slice_definition, slice):
        exps_list = list(
            range(*slice_definition.indices(sliceable_len)))
    elif isinstance(slice_definition, int):
        exps_list = [slice_definition]
    elif hasattr(slice_definition, 'shape') and \
            len(getattr(slice_definition, 'shape')) == 0:
        exps_list = [int(slice_definition)]
    else:
        exps_list = list(slice_definition)

    # Check experience id(s) boundaries
    if max(exps_list) >= sliceable_len:
        raise IndexError(
            'Experience index out of range: ' + str(max(exps_list)))

    if min(exps_list) < 0:
        raise IndexError(
            'Experience index out of range: ' + str(min(exps_list)))

    return exps_list


class AbstractExperience(IExperience[TScenario, TScenarioStream], ABC):
    """
    Definition of a learning experience. A learning experience contains a set of
    patterns which has become available at a particular time instant. The
    content and size of an Experience is defined by the specific benchmark that
    creates the experience.

    For instance, an experience of a New Classes scenario will contain all
    patterns belonging to a subset of classes of the original training set. An
    experience of a New Instance scenario will contain patterns from previously
    seen classes.

    Experiences of Single Incremental Task (a.k.a. task-free) scenarios are
    usually called "batches" while in Multi Task scenarios an Experience is
    usually associated to a "task". Finally, in a Multi Incremental Task
    scenario the Experience may be composed by patterns from different tasks.
    """

    def __init__(
            self: TExperience,
            origin_stream: TScenarioStream,
            current_experience: int,
            classes_in_this_exp: Sequence[int],
            previous_classes: Sequence[int],
            classes_seen_so_far: Sequence[int],
            future_classes: Optional[Sequence[int]]):
        """
        Creates an instance of the abstract experience given the scenario
        stream, the current experience ID and data about the classes timeline.

        :param origin_stream: The stream from which this experience was
            obtained.
        :param current_experience: The current experience ID, as an integer.
        :param classes_in_this_exp: The list of classes in this experience.
        :param previous_classes: The list of classes in previous experiences.
        :param classes_seen_so_far: List of classes of current and previous
            experiences.
        :param future_classes: The list of classes of next experiences.
        """

        self.origin_stream: TScenarioStream = origin_stream

        # scenario keeps a reference to the base scenario
        self.scenario: TScenario = origin_stream.scenario

        # current_experience is usually an incremental, 0-indexed, value used to
        # keep track of the current batch/task.
        self.current_experience: int = current_experience

        self.classes_in_this_experience: Sequence[int] = classes_in_this_exp
        """ The list of classes in this experience """

        self.previous_classes: Sequence[int] = previous_classes
        """ The list of classes in previous experiences """

        self.classes_seen_so_far: Sequence[int] = classes_seen_so_far
        """ List of classes of current and previous experiences """

        self.future_classes: Optional[Sequence[int]] = future_classes
        """ The list of classes of next experiences """

    @property
    def task_label(self) -> int:
        """
        The task label. This value will never have value "None". However,
        for scenarios that don't produce task labels a placeholder value like 0
        is usually set. Beware that this field is meant as a shortcut to obtain
        a unique task label: it assumes that only patterns labeled with a
        single task label are present. If this experience contains patterns from
        multiple tasks, accessing this property will result in an exception.
        """
        if len(self.task_labels) != 1:
            raise ValueError('The task_label property can only be accessed '
                             'when the experience contains a single task label')

        return self.task_labels[0]


class GenericExperience(AbstractExperience[TGenericCLScenario,
                                           GenericScenarioStream[
                                               TGenericExperience,
                                               TGenericCLScenario]]):
    """
    Definition of a learning experience based on a :class:`GenericCLScenario`
    instance.

    This experience implementation uses the generic experience-patterns
    assignment defined in the :class:`GenericCLScenario` instance. Instances of
    this class are usually obtained from a scenario stream.
    """

    def __init__(self: TGenericExperience,
                 origin_stream: GenericScenarioStream[TGenericExperience,
                                                      TGenericCLScenario],
                 current_experience: int):
        """
        Creates an instance of a generic experience given the stream from this
        experience was taken and and the current experience ID.

        :param origin_stream: The stream from which this experience was
            obtained.
        :param current_experience: The current experience ID, as an integer.
        """

        (classes_in_this_exp, previous_classes, classes_seen_so_far,
         future_classes) = origin_stream.scenario.get_classes_timeline(
            current_experience)

        super(GenericExperience, self).__init__(
            origin_stream, current_experience, classes_in_this_exp,
            previous_classes, classes_seen_so_far, future_classes)

    @property
    def dataset(self) -> AvalancheDataset:
        if self._is_train():
            dataset = self.scenario.train_dataset
            patterns_indexes = \
                self.scenario.train_exps_patterns_assignment[
                    self.current_experience]
        else:
            dataset = self.scenario.test_dataset
            if self.scenario.complete_test_set_only:
                patterns_indexes = None
            else:
                patterns_indexes = self.scenario.test_exps_patterns_assignment[
                    self.current_experience]

        return AvalancheSubset(dataset, indices=patterns_indexes)

    @property
    def task_labels(self) -> List[int]:
        if self._is_train():
            return self.scenario.task_labels[self.current_experience]
        else:
            if self.scenario.complete_test_set_only:
                return self.scenario.task_labels[0]
            else:
                return self.scenario.task_labels[self.current_experience]

    def _is_train(self):
        return self.origin_stream.name == 'train'


__all__ = [
    'TGenericCLScenario',
    'GenericCLScenario',
    'GenericScenarioStream',
    'AbstractExperience',
    'GenericExperience',
]
