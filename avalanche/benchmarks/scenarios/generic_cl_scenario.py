import copy
import re
from abc import ABC
from typing import (
    Generic,
    TypeVar,
    Union,
    Sequence,
    Callable,
    Optional,
    Dict,
    Any,
    Iterable,
    List,
    Set,
    Tuple,
    NamedTuple,
    Mapping,
)

import warnings
from torch.utils.data.dataset import Dataset

try:
    from gym import Env
except ImportError:
    # empty class to make sure everything below works without changes
    class Env:
        pass


from avalanche.benchmarks.scenarios.generic_definitions import (
    TExperience,
    ScenarioStream,
    TScenarioStream,
    Experience,
    TScenario,
)
from avalanche.benchmarks.scenarios.lazy_dataset_sequence import (
    LazyDatasetSequence,
)
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_utils import manage_advanced_indexing

TGenericCLScenario = TypeVar("TGenericCLScenario", bound="GenericCLScenario")
TGenericExperience = TypeVar("TGenericExperience", bound="GenericExperience")
TGenericScenarioStream = TypeVar(
    "TGenericScenarioStream", bound="GenericScenarioStream"
)

RLStreamDataOrigin = Union[Env, Sequence[Env]]
TStreamDataOrigin = Union[
    AvalancheDataset,
    Sequence[AvalancheDataset],
    Tuple[Iterable[AvalancheDataset], int],
    RLStreamDataOrigin,
]
TStreamTaskLabels = Optional[Sequence[Union[int, Set[int]]]]
TOriginDataset = Optional[Dataset]


# The definitions used to accept user stream definition
# Those definitions allow for a more simpler usage as they don't
# mandate setting task labels and the origin dataset
class StreamUserDef(NamedTuple):
    exps_data: TStreamDataOrigin
    exps_task_labels: TStreamTaskLabels = None
    origin_dataset: TOriginDataset = None
    is_lazy: Optional[bool] = None


TStreamUserDef = Union[
    Tuple[TStreamDataOrigin, TStreamTaskLabels, TOriginDataset, bool],
    Tuple[TStreamDataOrigin, TStreamTaskLabels, TOriginDataset],
    Tuple[TStreamDataOrigin, TStreamTaskLabels],
    Tuple[TStreamDataOrigin],
]

TStreamsUserDict = Dict[str, StreamUserDef]


# The definitions used to store stream definitions
class StreamDef(NamedTuple):
    exps_data: LazyDatasetSequence
    exps_task_labels: Sequence[Set[int]]
    origin_dataset: TOriginDataset
    is_lazy: bool


TStreamsDict = Dict[str, StreamDef]

STREAM_NAME_REGEX = re.compile("^[A-Za-z][A-Za-z_\\d]*$")


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

    def __init__(
        self: TGenericCLScenario,
        *,
        stream_definitions: TStreamsUserDict,
        complete_test_set_only: bool = False,
        experience_factory: Callable[
            ["GenericScenarioStream", int], TExperience
        ] = None,
    ):
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
            character and must not start with a number. Streams can be defined
            is two ways: static and lazy. In the static case, the
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
            In the lazy case, the stream must be defined as a tuple with 2
            elements:
            - The first element must be a tuple containing the dataset generator
            (one for each experience) and the number of experiences in that
            stream.
            - The second element must be a list containing the task labels of
            each experience (as an int or a set of ints).
        :param complete_test_set_only: If True, the test stream will contain
            a single experience containing the complete test set. This also
            means that the definition for the test stream must contain the
            definition for a single experience.
        :param experience_factory: If not None, a callable that, given the
            benchmark instance and the experience ID, returns an experience
            instance. This parameter is usually used in subclasses (when
            invoking the super constructor) to specialize the experience class.
            Defaults to None, which means that the :class:`GenericExperience`
            constructor will be used.
        """

        self.stream_definitions = GenericCLScenario._check_stream_definitions(
            stream_definitions
        )
        """
        A structure containing the definition of the streams.
        """

        self.original_train_dataset: Optional[
            Dataset
        ] = self.stream_definitions["train"].origin_dataset
        """ The original training set. May be None. """

        self.original_test_dataset: Optional[Dataset] = self.stream_definitions[
            "test"
        ].origin_dataset
        """ The original test set. May be None. """

        self.train_stream: GenericScenarioStream[
            TExperience, TGenericCLScenario
        ] = GenericScenarioStream("train", self)
        """
        The stream used to obtain the training experiences. 
        This stream can be sliced in order to obtain a subset of this stream.
        """

        self.test_stream: GenericScenarioStream[
            TExperience, TGenericCLScenario
        ] = GenericScenarioStream("test", self)
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
            if len(self.stream_definitions["test"].exps_data) > 1:
                raise ValueError(
                    "complete_test_set_only is True, but the test stream"
                    " contains more than one experience"
                )

        if experience_factory is None:
            experience_factory = GenericExperience

        self.experience_factory: Callable[
            [TGenericScenarioStream, int], TExperience
        ] = experience_factory

        # Create the original_<stream_name>_dataset fields for other streams
        self._make_original_dataset_fields()

        # Create the <stream_name>_stream fields for other streams
        self._make_stream_fields()

    @property
    def streams(
        self,
    ) -> Dict[str, "GenericScenarioStream[" "TExperience, TGenericCLScenario]"]:
        streams_dict = dict()
        for stream_name in self.stream_definitions.keys():
            streams_dict[stream_name] = getattr(self, f"{stream_name}_stream")

        return streams_dict

    @property
    def n_experiences(self) -> int:
        """The number of incremental training experiences contained
        in the train stream."""
        return len(self.stream_definitions["train"].exps_data)

    @property
    def task_labels(self) -> Sequence[List[int]]:
        """The task label of each training experience."""
        t_labels = []

        for exp_t_labels in self.stream_definitions["train"].exps_task_labels:
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
    def classes_in_experience(
        self,
    ) -> Mapping[str, Sequence[Optional[Set[int]]]]:
        """
        A dictionary mapping each stream (by name) to a list.

        Each element of the list is a set describing the classes included in
        that experience (identified by its index).

        In previous releases this field contained the list of sets for the
        training stream (that is, there was no way to obtain the list for other
        streams). That behavior is deprecated and support for that usage way
        will be removed in the future.
        """

        return LazyStreamClassesInExps(self)

    def get_classes_timeline(
        self, current_experience: int, stream: str = "train"
    ):
        """
        Returns the classes timeline given the ID of a experience.

        Given a experience ID, this method returns the classes in that
        experience, previously seen classes, the cumulative class list and a
        list of classes that will be encountered in next experiences of the
        same stream.

        Beware that by default this will obtain the timeline of an experience
        of the **training** stream. Use the stream parameter to select another
        stream.

        :param current_experience: The reference experience ID.
        :param stream: The stream name.
        :return: A tuple composed of four lists: the first list contains the
            IDs of classes in this experience, the second contains IDs of
            classes seen in previous experiences, the third returns a cumulative
            list of classes (that is, the union of the first two list) while the
            last one returns a list of classes that will be encountered in next
            experiences. Beware that each of these elements can be None when
            the benchmark is initialized by using a lazy generator.
        """

        class_set_current_exp = self.classes_in_experience[stream][
            current_experience
        ]

        if class_set_current_exp is not None:
            # May be None in lazy benchmarks
            classes_in_this_exp = list(class_set_current_exp)
        else:
            classes_in_this_exp = None

        class_set_prev_exps = set()
        for exp_id in range(0, current_experience):
            prev_exp_classes = self.classes_in_experience[stream][exp_id]
            if prev_exp_classes is None:
                # May be None in lazy benchmarks
                class_set_prev_exps = None
                break
            class_set_prev_exps.update(prev_exp_classes)

        if class_set_prev_exps is not None:
            previous_classes = list(class_set_prev_exps)
        else:
            previous_classes = None

        if (
            class_set_current_exp is not None
            and class_set_prev_exps is not None
        ):
            classes_seen_so_far = list(
                class_set_current_exp.union(class_set_prev_exps)
            )
        else:
            classes_seen_so_far = None

        class_set_future_exps = set()
        stream_n_exps = len(self.classes_in_experience[stream])
        for exp_id in range(current_experience + 1, stream_n_exps):
            future_exp_classes = self.classes_in_experience[stream][exp_id]
            if future_exp_classes is None:
                class_set_future_exps = None
                break
            class_set_future_exps.update(future_exp_classes)

        if class_set_future_exps is not None:
            future_classes = list(class_set_future_exps)
        else:
            future_classes = None

        return (
            classes_in_this_exp,
            previous_classes,
            classes_seen_so_far,
            future_classes,
        )

    def _make_original_dataset_fields(self):
        for stream_name, stream_def in self.stream_definitions.items():
            if stream_name in ["train", "test"]:
                continue

            orig_dataset = stream_def.origin_dataset
            setattr(self, f"original_{stream_name}_dataset", orig_dataset)

    def _make_stream_fields(self):
        for stream_name, stream_def in self.stream_definitions.items():
            if stream_name in ["train", "test"]:
                continue

            stream_obj = GenericScenarioStream(stream_name, self)
            setattr(self, f"{stream_name}_stream", stream_obj)

    @staticmethod
    def _check_stream_definitions(
        stream_definitions: TStreamsUserDict,
    ) -> TStreamsDict:
        """
        A function used to check the input stream definitions.

        This function should returns the adapted definition in which the
        missing optional fields are filled. If the input definition doesn't
        follow the expected structure, a `ValueError` will be raised.

        :param stream_definitions: The input stream definitions.
        :return: The checked and adapted stream definitions.
        """
        streams_defs = dict()

        if "train" not in stream_definitions:
            raise ValueError("No train stream found!")

        if "test" not in stream_definitions:
            raise ValueError("No test stream found!")

        for stream_name, stream_def in stream_definitions.items():
            GenericCLScenario._check_stream_name(stream_name)
            stream_def = GenericCLScenario._check_and_adapt_user_stream_def(
                stream_def, stream_name
            )
            streams_defs[stream_name] = stream_def

        return streams_defs

    @staticmethod
    def _check_stream_name(stream_name: Any):
        if not isinstance(stream_name, str):
            raise ValueError('Invalid type for stream name. Must be a "str"')

        if STREAM_NAME_REGEX.fullmatch(stream_name) is None:
            raise ValueError(f"Invalid name for stream {stream_name}")

    @staticmethod
    def _check_and_adapt_user_stream_def(
        stream_def: TStreamUserDef, stream_name: str
    ) -> StreamDef:
        exp_data = stream_def[0]
        task_labels = None
        origin_dataset = None
        is_lazy = None

        if len(stream_def) > 1:
            task_labels = stream_def[1]

        if len(stream_def) > 2:
            origin_dataset = stream_def[2]

        if len(stream_def) > 3:
            is_lazy = stream_def[3]

        if is_lazy or (isinstance(exp_data, tuple) and (is_lazy is None)):
            # Creation based on a generator
            if is_lazy:
                # We also check for LazyDatasetSequence, which is sufficient
                # per se (only if is_lazy==True, otherwise is treated as a
                # standard Sequence)
                if not isinstance(exp_data, LazyDatasetSequence):
                    if (not isinstance(exp_data, tuple)) or (
                        not len(exp_data) == 2
                    ):
                        raise ValueError(
                            f"The stream {stream_name} was flagged as "
                            f"lazy-generated but its definition is not a "
                            f"2-elements tuple (generator and stream length)."
                        )
            else:
                if (not len(exp_data) == 2) or (
                    not isinstance(exp_data[1], int)
                ):
                    raise ValueError(
                        f"The stream {stream_name} was detected "
                        f"as lazy-generated but its definition is not a "
                        f"2-elements tuple. If you're trying to define a "
                        f"non-lazily generated stream, don't use a tuple "
                        f"when passing the list of datasets, use a list "
                        f"instead."
                    )

            if isinstance(exp_data, LazyDatasetSequence):
                stream_length = len(exp_data)
            else:
                # exp_data[0] must contain the generator
                stream_length = exp_data[1]
            is_lazy = True
        elif isinstance(exp_data, AvalancheDataset):
            # Single element
            exp_data = [exp_data]
            is_lazy = False
            stream_length = 1
        elif isinstance(exp_data, Env) or all(
            [isinstance(e, Env) for e in exp_data]
        ):
            return StreamDef(exp_data, None, None, False)
        else:
            # Standard def
            stream_length = len(exp_data)
            is_lazy = False

        if not is_lazy:
            for i, dataset in enumerate(exp_data):
                if not isinstance(dataset, AvalancheDataset):
                    raise ValueError(
                        "All experience datasets must be subclasses of"
                        " AvalancheDataset"
                    )

        if task_labels is None:
            if is_lazy:
                raise ValueError(
                    "Task labels must be defined for each experience when "
                    "creating the stream using a generator."
                )

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

        if stream_length != len(task_labels):
            raise ValueError(
                f"{len(exp_data)} experiences have been defined, but task "
                f"labels for {len(task_labels)} experiences are given."
            )

        if is_lazy:
            if isinstance(exp_data, LazyDatasetSequence):
                lazy_sequence = exp_data
            else:
                lazy_sequence = LazyDatasetSequence(exp_data[0], stream_length)
        else:
            lazy_sequence = LazyDatasetSequence(exp_data, stream_length)
            lazy_sequence.load_all_experiences()

        return StreamDef(lazy_sequence, task_labels, origin_dataset, is_lazy)


class GenericScenarioStream(
    Generic[TExperience, TGenericCLScenario],
    ScenarioStream[TGenericCLScenario, TExperience],
    Sequence[TExperience],
):
    def __init__(
        self: TGenericScenarioStream,
        name: str,
        benchmark: TGenericCLScenario,
        *,
        slice_ids: List[int] = None,
    ):
        self.slice_ids: Optional[List[int]] = slice_ids
        """
        Describes which experiences are contained in the current stream slice. 
        Can be None, which means that this object is the original stream. """

        self.name: str = name
        """
        The name of the stream (for instance: "train", "test", "valid", ...).
        """

        self.benchmark = benchmark
        """
        A reference to the benchmark.
        """

    def __len__(self) -> int:
        """
        Gets the number of experiences this stream it's made of.

        :return: The number of experiences in this stream.
        """
        if self.slice_ids is None:
            return len(self.benchmark.stream_definitions[self.name].exps_data)
        else:
            return len(self.slice_ids)

    def __getitem__(
        self, exp_idx: Union[int, slice, Iterable[int]]
    ) -> Union[TExperience, TScenarioStream]:
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
                    return self.benchmark.experience_factory(self, exp_idx)
                else:
                    return self.benchmark.experience_factory(
                        self, self.slice_ids[exp_idx]
                    )
            raise IndexError(
                "Experience index out of bounds" + str(int(exp_idx))
            )
        else:
            return self._create_slice(exp_idx)

    def _create_slice(
        self: TGenericScenarioStream,
        exps_slice: Union[int, slice, Iterable[int]],
    ) -> TScenarioStream:
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

    def drop_previous_experiences(self, to_exp: int) -> None:
        """
        Drop the reference to experiences up to a certain experience ID
        (inclusive).

        This means that any reference to experiences with ID [0, from_exp] will
        be released. By dropping the reference to previous experiences, the
        memory associated with them can be freed, especially the one occupied by
        the dataset. However, if external references to the experience or the
        dataset still exist, dropping previous experiences at the stream level
        will have little to no impact on the memory usage.

        To make sure that the underlying dataset can be freed, make sure that:
        - No reference to previous datasets or experiences are kept in you code;
        - The replay implementation doesn't keep a reference to previous
            datasets (in which case, is better to store a copy of the raw
            tensors instead);
        - The benchmark is being generated using a lazy initializer.

        By dropping previous experiences, those experiences will no longer be
        available in the stream. Trying to access them will result in an
        exception.

        :param to_exp: The ID of the last exp to drop (inclusive). Can be a
            negative number, in which case this method doesn't have any effect.
            Can be greater or equal to the stream length, in which case all
            currently loaded experiences will be dropped.
        :return: None
        """
        self.benchmark.stream_definitions[
            self.name
        ].exps_data.drop_previous_experiences(to_exp)


class LazyStreamClassesInExps(Mapping[str, Sequence[Optional[Set[int]]]]):
    def __init__(self, benchmark: GenericCLScenario):
        self._benchmark = benchmark
        self._default_lcie = LazyClassesInExps(benchmark, stream="train")

    def __len__(self):
        return len(self._benchmark.stream_definitions)

    def __getitem__(self, stream_name_or_exp_id):
        if isinstance(stream_name_or_exp_id, str):
            return LazyClassesInExps(
                self._benchmark, stream=stream_name_or_exp_id
            )

        warnings.warn(
            "Using classes_in_experience[exp_id] is deprecated. "
            "Consider using classes_in_experience[stream_name][exp_id]"
            "instead.",
            stacklevel=2,
        )
        return self._default_lcie[stream_name_or_exp_id]

    def __iter__(self):
        yield from self._benchmark.stream_definitions.keys()


class LazyClassesInExps(Sequence[Optional[Set[int]]]):
    def __init__(self, benchmark: GenericCLScenario, stream: str = "train"):
        self._benchmark = benchmark
        self._stream = stream

    def __len__(self):
        return len(self._benchmark.streams[self._stream])

    def __getitem__(self, exp_id) -> Set[int]:
        return manage_advanced_indexing(
            exp_id,
            self._get_single_exp_classes,
            len(self),
            LazyClassesInExps._slice_collate,
        )

    def __str__(self):
        return (
            "[" + ", ".join([str(self[idx]) for idx in range(len(self))]) + "]"
        )

    def _get_single_exp_classes(self, exp_id):
        b = self._benchmark.stream_definitions[self._stream]
        if not b.is_lazy and exp_id not in b.exps_data.targets_field_sequence:
            raise IndexError
        targets = b.exps_data.targets_field_sequence[exp_id]
        if targets is None:
            return None
        return set(targets)

    @staticmethod
    def _slice_collate(*classes_in_exps: Optional[Set[int]]):
        if any(x is None for x in classes_in_exps):
            return None

        return [list(x) for x in classes_in_exps]


def _get_slice_ids(
    slice_definition: Union[int, slice, Iterable[int]], sliceable_len: int
) -> List[int]:
    # Obtain experiences list from slice object (or any iterable)
    exps_list: List[int]
    if isinstance(slice_definition, slice):
        exps_list = list(range(*slice_definition.indices(sliceable_len)))
    elif isinstance(slice_definition, int):
        exps_list = [slice_definition]
    elif (
        hasattr(slice_definition, "shape")
        and len(getattr(slice_definition, "shape")) == 0
    ):
        exps_list = [int(slice_definition)]
    else:
        exps_list = list(slice_definition)

    # Check experience id(s) boundaries
    if max(exps_list) >= sliceable_len:
        raise IndexError(
            "Experience index out of range: " + str(max(exps_list))
        )

    if min(exps_list) < 0:
        raise IndexError(
            "Experience index out of range: " + str(min(exps_list))
        )

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
        future_classes: Optional[Sequence[int]],
    ):
        """
        Creates an instance of the abstract experience given the benchmark
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

        # benchmark keeps a reference to the base benchmark
        self.benchmark: TScenario = origin_stream.benchmark

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
            raise ValueError(
                "The task_label property can only be accessed "
                "when the experience contains a single task label"
            )

        return self.task_labels[0]


class GenericExperience(
    AbstractExperience[
        TGenericCLScenario,
        GenericScenarioStream[TGenericExperience, TGenericCLScenario],
    ]
):
    """
    Definition of a learning experience based on a :class:`GenericCLScenario`
    instance.

    This experience implementation uses the generic experience-patterns
    assignment defined in the :class:`GenericCLScenario` instance. Instances of
    this class are usually obtained from a benchmark stream.
    """

    def __init__(
        self: TGenericExperience,
        origin_stream: GenericScenarioStream[
            TGenericExperience, TGenericCLScenario
        ],
        current_experience: int,
    ):
        """
        Creates an instance of a generic experience given the stream from this
        experience was taken and the current experience ID.

        :param origin_stream: The stream from which this experience was
            obtained.
        :param current_experience: The current experience ID, as an integer.
        """
        self.dataset: AvalancheDataset = (
            origin_stream.benchmark.stream_definitions[
                origin_stream.name
            ].exps_data[current_experience]
        )

        (
            classes_in_this_exp,
            previous_classes,
            classes_seen_so_far,
            future_classes,
        ) = origin_stream.benchmark.get_classes_timeline(
            current_experience, stream=origin_stream.name
        )

        super(GenericExperience, self).__init__(
            origin_stream,
            current_experience,
            classes_in_this_exp,
            previous_classes,
            classes_seen_so_far,
            future_classes,
        )

    def _get_stream_def(self):
        return self.benchmark.stream_definitions[self.origin_stream.name]

    @property
    def task_labels(self) -> List[int]:
        stream_def = self._get_stream_def()
        return list(stream_def.exps_task_labels[self.current_experience])


__all__ = [
    "StreamUserDef",
    "TStreamUserDef",
    "TStreamsUserDict",
    "StreamDef",
    "TStreamsDict",
    "TGenericCLScenario",
    "GenericCLScenario",
    "GenericScenarioStream",
    "AbstractExperience",
    "GenericExperience",
]
