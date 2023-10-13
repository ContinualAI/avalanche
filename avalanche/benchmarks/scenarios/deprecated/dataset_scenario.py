from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
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
    Mapping,
    TYPE_CHECKING,
)

from . import DatasetExperience


from ..generic_scenario import (
    CLScenario,
    CLExperience,
    CLStream,
    SequenceCLStream,
)


from .lazy_dataset_sequence import LazyDatasetSequence

from avalanche.benchmarks.utils import AvalancheDataset

from torch.utils.data.dataset import Dataset


# --- Dataset ---
# From utils:
TCLDataset = TypeVar("TCLDataset", bound="AvalancheDataset", covariant=True)

# --- Scenario ---
# From generic_scenario:
TCLScenario = TypeVar("TCLScenario", bound="CLScenario", covariant=True)
# Defined here:
TDatasetScenario = TypeVar("TDatasetScenario", bound="DatasetScenario")
TClassesTimelineCLScenario = TypeVar(
    "TClassesTimelineCLScenario", bound="ClassesTimelineCLScenario"
)

# --- Stream ---
# From generic_scenario:
TCLStream = TypeVar("TCLStream", bound="CLStream")

# --- Experience ---
TCLExperience = TypeVar("TCLExperience", bound="CLExperience")
TDatasetExperience = TypeVar("TDatasetExperience", bound="DatasetExperience")


# Definitions (stream)
TStreamDataOrigin = Union[
    AvalancheDataset,
    Iterable[AvalancheDataset],
    Tuple[Iterable[AvalancheDataset], int],
]
TStreamTaskLabels = Optional[Iterable[Union[int, Iterable[int]]]]
TOriginDataset = Optional[Union[Dataset, AvalancheDataset]]


# The definitions used to accept user stream definition
# Those definitions allow for a more simpler usage as they don't
# mandate setting task labels and the origin dataset
@dataclass
class StreamUserDef(Generic[TCLDataset]):
    exps_data: Union[TCLDataset, Iterable[TCLDataset], Tuple[Iterable[TCLDataset], int]]
    exps_task_labels: TStreamTaskLabels = None
    origin_dataset: TOriginDataset = None
    is_lazy: Optional[bool] = None


# Element used to store stream definitions
@dataclass
class StreamDef(Generic[TCLDataset]):
    exps_data: LazyDatasetSequence[TCLDataset]
    exps_task_labels: Sequence[Set[int]]
    origin_dataset: TOriginDataset
    is_lazy: bool


TStreamUserDef = Union[
    Tuple[TStreamDataOrigin, TStreamTaskLabels, TOriginDataset, Optional[bool]],
    Tuple[TStreamDataOrigin, TStreamTaskLabels, TOriginDataset],
    Tuple[TStreamDataOrigin, TStreamTaskLabels],
    Tuple[TStreamDataOrigin],
]

TStreamsUserDict = Mapping[str, Union[StreamUserDef, TStreamUserDef, StreamDef]]


STREAM_NAME_REGEX = re.compile("^[A-Za-z][A-Za-z_\\d]*$")


class DatasetScenario(
    CLScenario[TCLStream], Generic[TCLStream, TDatasetExperience, TCLDataset]
):
    """
    Base implementation of a Continual Learning benchmark instance.
    A Continual Learning benchmark instance is defined by a set of streams of
    experiences (batches or tasks depending on the terminology). Each
    experience contains the training (or test, or validation, ...) data that
    becomes available at a certain time instant.

    Experiences are usually defined in children classes, with this class
    serving as the more general implementation. This class handles the most
    simple type of assignment: each stream is defined as a list of experiences,
    each experience is defined by a dataset.

    Defining the "train" and "test" streams is mandatory. This class supports
    custom streams as well. Custom streams can be accessed by using the
    `streamname_stream` field of the created instance.

    The name of custom streams can only contain letters, numbers or the "_"
    character and must not start with a number.
    """

    def __init__(
        self: TDatasetScenario,
        *,
        stream_definitions: TStreamsUserDict,
        stream_factory: Callable[[str, TDatasetScenario], TCLStream],
        experience_factory: Callable[[TCLStream, int], TDatasetExperience],
        complete_test_set_only: bool = False,
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
            - The first element must be a tuple containing the dataset
                generator (one for each experience) and the number of
                experiences in that stream.
            - The second element must be a list containing the task labels of
                each experience (as an int or a set of ints).
        :param complete_test_set_only: If True, the test stream will contain
            a single experience containing the complete test set. This also
            means that the definition for the test stream must contain the
            definition for a single experience.
        :param stream_factory: A callable that, given the name of the
            stream and the benchmark instance, returns a stream instance.
            This parameter is usually used in subclasses (when
            invoking the super constructor) to specialize the experience class.
        :param experience_factory: A callable that, given the
            stream instance and the experience ID, returns an experience
            instance. This parameter is usually used in subclasses (when
            invoking the super constructor) to specialize the experience class.
        """

        self.experience_factory: Callable[
            [TCLStream, int], TDatasetExperience
        ] = experience_factory

        self.stream_factory: Callable[
            [str, TDatasetScenario], TCLStream
        ] = stream_factory

        self.stream_definitions: Dict[
            str, StreamDef[TCLDataset]
        ] = DatasetScenario._check_stream_definitions(stream_definitions)
        """
        A structure containing the definition of the streams.
        """

        self.original_train_dataset: Optional[TOriginDataset] = self.stream_definitions[
            "train"
        ].origin_dataset
        """ The original training set. May be None. """

        self.original_test_dataset: Optional[TOriginDataset] = self.stream_definitions[
            "test"
        ].origin_dataset
        """ The original test set. May be None. """

        self.train_stream: TCLStream = self.stream_factory("train", self)
        """
        The stream used to obtain the training experiences. 
        This stream can be sliced in order to obtain a subset of this stream.
        """

        self.test_stream: TCLStream = self.stream_factory("test", self)
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

        # Create the original_<stream_name>_dataset fields for other streams
        self._make_original_dataset_fields()

        # Create the <stream_name>_stream fields for other streams
        self._make_stream_fields()

        super().__init__(
            [
                getattr(self, f"{stream_name}_stream")
                for stream_name in self.stream_definitions.keys()
            ]
        )

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

    def _make_original_dataset_fields(self):
        for stream_name, stream_def in self.stream_definitions.items():
            if stream_name in ["train", "test"]:
                continue

            orig_dataset = stream_def.origin_dataset
            setattr(self, f"original_{stream_name}_dataset", orig_dataset)

    def _make_stream_fields(self: TDatasetScenario):
        for stream_name, stream_def in self.stream_definitions.items():
            if stream_name in ["train", "test"]:
                continue

            stream_obj = self.stream_factory(stream_name, self)
            setattr(self, f"{stream_name}_stream", stream_obj)

    @staticmethod
    def _check_stream_definitions(
        stream_definitions: TStreamsUserDict,
    ) -> Dict[str, StreamDef]:
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

        for stream_name, stream_user_def in stream_definitions.items():
            DatasetScenario._check_stream_name(stream_name)
            stream_def = DatasetScenario._check_and_adapt_user_stream_def(
                stream_user_def, stream_name
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
        stream_def: Union[
            StreamDef[TCLDataset], StreamUserDef[TCLDataset], TStreamUserDef
        ],
        stream_name: str,
    ) -> StreamDef[TCLDataset]:
        if isinstance(stream_def, StreamDef):
            return stream_def

        if isinstance(stream_def, StreamUserDef):
            stream_def = (
                stream_def.exps_data,
                stream_def.exps_task_labels,
                stream_def.origin_dataset,
                stream_def.is_lazy,
            )

        exp_data: TStreamDataOrigin = stream_def[0]
        task_labels: TStreamTaskLabels = None
        origin_dataset: TOriginDataset = None
        is_lazy: Optional[bool] = None

        if exp_data is None:
            raise ValueError("Experience data can't be None")

        if len(stream_def) > 1:
            task_labels = stream_def[1]  # type: ignore

        if len(stream_def) > 2:
            origin_dataset = stream_def[2]  # type: ignore

        if len(stream_def) > 3:
            is_lazy = stream_def[3]  # type: ignore

        if is_lazy or (isinstance(exp_data, tuple) and (is_lazy is None)):
            # Creation based on a generator
            if is_lazy:
                # We also check for LazyDatasetSequence, which is sufficient
                # per se (only if is_lazy==True, otherwise is treated as a
                # standard Sequence)
                if not isinstance(exp_data, LazyDatasetSequence):
                    if (not isinstance(exp_data, tuple)) or (not len(exp_data) == 2):
                        raise ValueError(
                            f"The stream {stream_name} was flagged as "
                            f"lazy-generated but its definition is not a "
                            f"2-elements tuple (generator and stream length)."
                        )
            else:
                if (
                    (not isinstance(exp_data, Sequence))
                    or (not len(exp_data) == 2)
                    or (not isinstance(exp_data[1], int))
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
        else:
            # Standard def
            stream_length = len(exp_data)  # type: ignore
            is_lazy = False

        if not is_lazy:
            for i, dataset in enumerate(exp_data):  # type: ignore
                if not isinstance(dataset, AvalancheDataset):
                    raise ValueError(
                        "All experience datasets must be subclasses of"
                        " AvalancheDataset"
                    )

        task_labels_list: List[Set[int]] = []
        if task_labels is None:
            if is_lazy:
                raise ValueError(
                    "Task labels must be defined for each experience when "
                    "creating the stream using a generator."
                )

            # Extract task labels from the dataset
            exp_dataset: AvalancheDataset
            for i, exp_dataset in enumerate(exp_data):  # type: ignore
                task_labels_list.append(
                    set(exp_dataset.targets_task_labels)
                )  # type: ignore
        else:
            # Standardize task labels structure
            for t_l in task_labels:
                if isinstance(t_l, int):
                    task_labels_list.append({t_l})
                elif not isinstance(t_l, set):
                    task_labels_list.append(set(t_l))
                else:
                    task_labels_list.append(t_l)

        if stream_length != len(task_labels_list):
            raise ValueError(
                f"{stream_length} experiences have been defined, but task "
                f"labels for {len(task_labels_list)} experiences are given."
            )

        lazy_sequence: LazyDatasetSequence[TCLDataset]
        if is_lazy:
            if isinstance(exp_data, LazyDatasetSequence):
                lazy_sequence = exp_data  # type: ignore
            else:
                lazy_sequence = LazyDatasetSequence(
                    exp_data[0], stream_length  # type: ignore
                )  # type: ignore
        else:
            lazy_sequence = LazyDatasetSequence(
                exp_data, stream_length  # type: ignore
            )  # type: ignore
            lazy_sequence.load_all_experiences()

        return StreamDef(lazy_sequence, task_labels_list, origin_dataset, is_lazy)


class ClassesTimelineCLScenario(
    DatasetScenario[TCLStream, TDatasetExperience, TCLDataset], ABC
):
    @property
    @abstractmethod
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
        pass

    def get_classes_timeline(
        self, current_experience: int, stream: str = "train"
    ) -> Tuple[
        Optional[List[int]],
        Optional[List[int]],
        Optional[List[int]],
        Optional[List[int]],
    ]:
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
        class_set_current_exp = self.classes_in_experience[stream][current_experience]

        if class_set_current_exp is not None:
            # May be None in lazy benchmarks
            classes_in_this_exp = list(class_set_current_exp)
        else:
            classes_in_this_exp = None

        prev_exps_not_none = True
        class_set_prev_exps: Set[int] = set()
        for exp_id in range(0, current_experience):
            prev_exp_classes = self.classes_in_experience[stream][exp_id]
            if prev_exp_classes is None:
                # May be None in lazy benchmarks
                prev_exps_not_none = False
                break
            class_set_prev_exps.update(prev_exp_classes)

        if prev_exps_not_none:
            previous_classes = list(class_set_prev_exps)
        else:
            previous_classes = None

        if class_set_current_exp is not None and prev_exps_not_none:
            classes_seen_so_far = list(class_set_current_exp.union(class_set_prev_exps))
        else:
            classes_seen_so_far = None

        future_exps_not_none = True
        class_set_future_exps: Set[int] = set()
        stream_n_exps = len(self.classes_in_experience[stream])
        for exp_id in range(current_experience + 1, stream_n_exps):
            future_exp_classes = self.classes_in_experience[stream][exp_id]
            if future_exp_classes is None:
                future_exps_not_none = False
                break
            class_set_future_exps.update(future_exp_classes)

        if future_exps_not_none:
            future_classes = list(class_set_future_exps)
        else:
            future_classes = None

        return (
            classes_in_this_exp,
            previous_classes,
            classes_seen_so_far,
            future_classes,
        )


class DatasetStream(SequenceCLStream[TDatasetExperience]):
    """
    Base class for all streams connected to a :class:`DatasetScenario`.

    This class includes proper typing for the `benchmark` field and
    the `drop_previous_experiences` method, which can be used to drop
    references to already processed datasets.
    """

    def __init__(
        self,
        name: str,
        benchmark: DatasetScenario,
        *,
        slice_ids: Optional[List[int]] = None,
        set_stream_info: bool = True,
    ):
        self.benchmark: DatasetScenario = benchmark

        super().__init__(
            name=name,
            benchmark=benchmark,
            slice_ids=slice_ids,
            set_stream_info=set_stream_info,
        )

    def drop_previous_experiences(self, to_exp: int) -> None:
        """
        Drop the reference to experiences up to a certain experience ID
        (inclusive).

        This means that any reference to experiences with ID [0, from_exp] will
        be released. By dropping the reference to previous experiences, the
        memory associated with them can be freed, especially the one occupied
        by the dataset. However, if external references to the experience or
        the dataset still exist, dropping previous experiences at the stream
        level will have little to no impact on the memory usage.

        To make sure that the underlying dataset can be freed, make sure that:
        - No reference to previous datasets or experiences are kept in your
            code;
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


class FactoryBasedStream(DatasetStream[TDatasetExperience]):
    def __init__(
        self,
        name: str,
        benchmark: DatasetScenario,
        *,
        slice_ids: Optional[List[int]] = None,
        set_stream_info: bool = True,
    ):
        super().__init__(
            name=name,
            benchmark=benchmark,
            slice_ids=slice_ids,
            set_stream_info=set_stream_info,
        )

    def _full_length(self) -> int:
        return len(self.benchmark.stream_definitions[self.name].exps_data)

    def _make_experience(self, experience_idx: int) -> TDatasetExperience:
        a = self.benchmark.experience_factory(self, experience_idx)  # type: ignore
        return a


__all__ = [
    "StreamUserDef",
    "TStreamUserDef",
    "TStreamsUserDict",
    "StreamDef",
    "DatasetScenario",
    "ClassesTimelineCLScenario",
    "FactoryBasedStream",
]
