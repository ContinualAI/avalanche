import copy
import re
from abc import ABC
from typing import Generic, TypeVar, Union, Sequence, Callable, Optional, \
    Dict, Any, Iterable, List, Set, Iterator, Tuple, NamedTuple

from torch.utils.data.dataset import Dataset

from avalanche.benchmarks.scenarios.generic_definitions import \
    TExperience, ScenarioStream, TScenarioStream, Experience, TScenario
from avalanche.benchmarks.utils import AvalancheDataset

TGenericCLScenario = TypeVar('TGenericCLScenario', bound='GenericCLScenario')
TGenericExperience = TypeVar('TGenericExperience', bound='GenericExperience')
TGenericScenarioStream = TypeVar('TGenericScenarioStream',
                                 bound='GenericScenarioStream')

TStreamDataOrigin = Union[AvalancheDataset, Sequence[AvalancheDataset],
                          Tuple[Iterator[AvalancheDataset], int]]
TStreamTaskLabels = Optional[Sequence[Union[int, Set[int]]]]
TOriginDataset = Optional[Dataset]


# The definitions used to accept user stream definition
# Those definitions allow for a more simpler usage as they don't
# mandate setting task labels and the origin dataset
class StreamUserDef(NamedTuple):
    exps_data: TStreamDataOrigin
    exps_task_labels: TStreamTaskLabels = None
    origin_dataset: TOriginDataset = None


TStreamUserDef = \
    Union[Tuple[TStreamDataOrigin, TStreamTaskLabels, TOriginDataset],
          Tuple[TStreamDataOrigin, TStreamTaskLabels],
          Tuple[TStreamDataOrigin]]


TStreamsUserDict = Dict[str, TStreamUserDef]


# The definitions used to store stream definitions
class StreamDef(NamedTuple):
    exps_data: Sequence[AvalancheDataset]
    exps_task_labels: Sequence[Set[int]]
    origin_dataset: TOriginDataset


TStreamsDict = Dict[str, StreamDef]


STREAM_NAME_REGEX = re.compile('^[A-Za-z][A-Za-z_\\d]*$')


class GenericCLScenario(Generic[TExperience]):
    """
    Base implementation of a Continual Learning benchmark instance.
    A Continual Learning benchmark instance is defined by a set of streams of
    experiences (batches or tasks depending on the terminology). Each experience
    contains the training (or test, or validation, ...) data that becomes
    available at a certain time instant.

    Experiences are usually defined in children classes, with this class serving
    as the more general implementation. This class handles the most simple type
    of assignment: each stream is defined as a list of experiences, each
    experience is defined by a dataset.

    Defining the "train" and "test" streams is mandatory. This class supports
    custom streams as well. Custom streams can be accessed by using the
    `streamname_stream` field of the created instance.

    The name of custom streams can only contain letters, numbers or the "_"
    character and must not start with a number.
    """
    def __init__(self: TGenericCLScenario,
                 *,
                 stream_definitions: TStreamsUserDict,
                 complete_test_set_only: bool = False,
                 experience_factory: Callable[['GenericScenarioStream', int],
                                              TExperience] = None):
        """
        Creates an instance of a Continual Learning benchmark instance.

        The benchmark instance is defined by a stream definition dictionary,
        which describes the content of each stream. The "train" and "test"
        stream are mandatory. Any other custom stream can be added.

        There is no constraint on the amount of experiences in each stream
        (excluding the case in which `complete_test_set_only` is set).

        :param stream_definitions: The stream definitions dictionary. Must
            be a dictionary where the key is the stream name and the value
            is the definition of that stream. "train" and "test" streams are
            mandatory. This class supports custom streams as well. The name of
            custom streams can only contain letters, numbers and the "_"
            character and must not start with a number. The definition of each
            stream must be a tuple containing 1, 2 or 3 elements:
            - The first element must be a list containing the datasets
            describing each experience. Datasets must be instances of
            :class:`AvalancheDataset`.
            - The second element is optional and must be a list containing the
            task labels of each experience (as an int or a set of ints).
            If the stream definition tuple contains only one element (the list
            of datasets), then the task labels for each experience will be
            obtained by inspecting the content of the datasets.
            - The third element is optional and must be a reference to the
            originating dataset (if applicable). For instance, for SplitMNIST
            this may be a reference to the whole MNIST dataset. If the stream
            definition tuple contains less than 3 elements, then the reference
            to the original dataset will be set to None.
        :param complete_test_set_only: If True, the test stream will contain
            a single experience containing the complete test set. This also
            means that the definition for the test stream must contain the
            definition for a single experience.
        :param experience_factory: If not None, a callable that, given the
            scenario instance and the experience ID, returns a experience
            instance. This parameter is usually used in subclasses (when
            invoking the super constructor) to specialize the experience class.
            Defaults to None, which means that the :class:`GenericExperience`
            constructor will be used.
        """

        self.stream_definitions = \
            self._check_stream_definitions(stream_definitions)

        self.original_train_dataset: Optional[Dataset] = \
            self.stream_definitions['train'].origin_dataset
        """ The original training set. May be None. """

        self.original_test_dataset: Optional[Dataset] = \
            self.stream_definitions['test'].origin_dataset
        """ The original test set. May be None. """

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

        self.complete_test_set_only: bool = bool(complete_test_set_only)
        """
        If True, only the complete test set will be returned from experience
        instances.
        
        This flag is usually set to True in scenarios where having one separate
        test set aligned to each training experience is impossible or doesn't
        make sense from a semantic point of view.
        """

        if self.complete_test_set_only:
            if len(self.stream_definitions['test'].exps_data) > 1:
                raise ValueError(
                    'complete_test_set_only is True, but the test stream'
                    ' contains more than one experience')

        if experience_factory is None:
            experience_factory = GenericExperience

        self.experience_factory: Callable[[TGenericScenarioStream, int],
                                          TExperience] = experience_factory

        self._make_original_dataset_fields()
        self._make_stream_fields()

    @property
    def n_experiences(self) -> int:
        """  The number of incremental training experiences contained
        in the train stream. """
        return len(self.stream_definitions['train'].exps_data)

    @property
    def task_labels(self) -> Sequence[List[int]]:
        """ The task label of each training experience. """
        t_labels = []

        for exp_t_labels in self.stream_definitions['train'].exps_task_labels:
            t_labels.append(list(exp_t_labels))

        return t_labels

    def get_reproducibility_data(self) -> Dict[str, Any]:
        """
        Gets the data needed to reproduce this experiment.

        This data can be stored using the pickle module or some other mechanism.
        It can then be loaded by passing it as the ``reproducibility_data``
        parameter in the constructor.

        Child classes should create their own reproducibility dictionary.
        This means that the implementation found in :class:`GenericCLScenario`
        will return an empty dictionary, which is meaningless.

        In order to obtain the same benchmark instance, the reproducibility
        data must be passed to the constructor along with the exact same
        input datasets.

        :return: A dictionary containing the data needed to reproduce the
            experiment.
        """

        return dict()

    @property
    def classes_in_experience(self) -> Sequence[Set[int]]:
        """ A list that, for each experience (identified by its index/ID),
        stores a set of the (optionally remapped) IDs of classes of patterns
        assigned to that experience. """
        return LazyClassesInExps(self)

    def get_classes_timeline(self, current_experience: int):
        """
        Returns the classes timeline given the ID of a training experience.

        Given a experience ID, this method returns the classes in that training
        experience, previously seen classes, the cumulative class list and a
        list of classes that will be encountered in next training experiences.

        Beware that this will obtain the timeline of an experience of the
        **training** stream.

        :param current_experience: The reference training experience ID.
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

    def _make_original_dataset_fields(self):
        for stream_name, stream_def in self.stream_definitions.items():
            if stream_name in ['train', 'test']:
                continue

            orig_dataset = stream_def.origin_dataset

            setattr(self, f'original_{stream_name}_dataset', orig_dataset)

    def _make_stream_fields(self):
        for stream_name, stream_def in self.stream_definitions.items():
            if stream_name in ['train', 'test']:
                continue

            stream_obj = GenericScenarioStream(stream_name, self)
            setattr(self, f'{stream_name}_stream', stream_obj)

    def _check_stream_definitions(
            self, stream_definitions: TStreamsUserDict) -> TStreamsDict:
        """
        A function used to check the input stream definitions.

        This function should returns the adapted definition in which the
        missing optional fields are filled. If the input definition doesn't
        follow the expected structure, a `ValueError` will be raised.

        :param stream_definitions: The input stream definitions.
        :return: The checked and adapted stream definitions.
        """
        streams_defs = dict()

        if 'train' not in stream_definitions:
            raise ValueError('No train stream found!')

        if 'test' not in stream_definitions:
            raise ValueError('No test stream found!')

        for stream_name, stream_def in stream_definitions.items():
            self._check_stream_name(stream_name)
            stream_def = self._check_single_stream_def(stream_def)
            streams_defs[stream_name] = stream_def

        return streams_defs

    def _check_stream_name(self, stream_name: Any):
        if not isinstance(stream_name, str):
            raise ValueError('Invalid type for stream name. Must be a "str"')

        if STREAM_NAME_REGEX.fullmatch(stream_name) is None:
            raise ValueError(f'Invalid name for stream {stream_name}')

    def _check_single_stream_def(self, stream_def: TStreamUserDef) -> StreamDef:
        exp_data = stream_def[0]
        task_labels = None
        origin_dataset = None
        if len(stream_def) > 1:
            task_labels = stream_def[1]

        if len(stream_def) > 2:
            origin_dataset = stream_def[2]

        if isinstance(exp_data, Dataset):
            # Single element
            exp_data = [exp_data]
        elif isinstance(exp_data, tuple):
            # Generator
            # We currently don't support lazily created experiences...
            exp_data_lst = []
            n_exps = exp_data[1]
            exp_idx = 0
            for exp in exp_data[0]:
                if exp_idx >= n_exps:
                    break
                exp_data_lst.append(exp)
                exp_idx += 1
            exp_data = exp_data_lst
        else:
            # Standard def
            exp_data = list(exp_data)

        for i, dataset in enumerate(exp_data):
            if not isinstance(dataset, AvalancheDataset):
                raise ValueError('All experience datasets must be subclasses of'
                                 ' AvalancheDataset')

        if task_labels is None:
            # Extract task labels from the dataset
            task_labels = []
            for i in range(len(exp_data)):
                exp_dataset: AvalancheDataset = exp_data[i]
                task_labels.append(set(exp_dataset.targets_task_labels))
        else:
            # Standardize task labels structure
            task_labels = list(task_labels)
            for i in range(len(task_labels)):
                if isinstance(task_labels[i], int):
                    task_labels[i] = {task_labels[i]}
                elif not isinstance(task_labels[i], set):
                    task_labels[i] = set(task_labels[i])

        if len(exp_data) != len(task_labels):
            raise ValueError(
                f'{len(exp_data)} experiences have been defined, but task '
                f'labels for {len(task_labels)} experiences are given.')

        return StreamDef(exp_data, task_labels, origin_dataset)


class GenericScenarioStream(Generic[TExperience, TGenericCLScenario],
                            ScenarioStream[TGenericCLScenario, TExperience],
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
        Gets the number of experiences this stream it's made of.

        :return: The number of experiences in this stream.
        """
        if self.slice_ids is None:
            return len(self.scenario.stream_definitions[self.name].exps_data)
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
        return set(self._scenario.stream_definitions['train']
                   .exps_data[exp_id].targets)

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


class AbstractExperience(Experience[TScenario, TScenarioStream], ABC):
    """
    Definition of a learning experience. A learning experience contains a set of
    patterns which has become available at a particular time instant. The
    content and size of an Experience is defined by the specific benchmark that
    creates the experience.

    For instance, an experience of a New Classes scenario will contain all
    patterns belonging to a subset of classes of the original training set. An
    experience of a New Instance scenario will contain patterns from previously
    seen classes.
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

    def _get_stream_def(self):
        return self.scenario.stream_definitions[self.origin_stream.name]

    @property
    def dataset(self) -> AvalancheDataset:
        return self._get_stream_def().exps_data[self.current_experience]

    @property
    def task_labels(self) -> List[int]:
        stream_def = self._get_stream_def()
        return list(stream_def.exps_task_labels[self.current_experience])


__all__ = [
    'StreamDef',
    'TStreamsDict',
    'TGenericCLScenario',
    'GenericCLScenario',
    'GenericScenarioStream',
    'AbstractExperience',
    'GenericExperience',
]
