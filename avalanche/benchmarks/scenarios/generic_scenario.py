################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 11-04-2022                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
from abc import abstractmethod, ABC
from contextlib import contextmanager
from copy import copy
from enum import Enum
from typing import Dict, Iterator, List, Iterable, Optional, Sequence, Tuple, TypeVar, Union, Generic, final, overload
from typing_extensions import Protocol, runtime_checkable
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.classification_dataset import \
    ClassificationDataset
import warnings

from avalanche.benchmarks.utils.dataset_utils import slice_alike_object_to_indices

T = TypeVar("T")
TCov = TypeVar("TCov", covariant=True)
E = TypeVar("E")

# Dataset
TCLDataset = TypeVar("TCLDataset", bound="AvalancheDataset")  # Implementation, defined in utils
TCLDatasetCov = TypeVar("TCLDatasetCov", bound="AvalancheDataset", covariant=True)  # Implementation, defined in utils

# Scenario
TCLScenario = TypeVar("TCLScenario", bound="CLScenario")  # Implementation, defined here
TCLScenarioCov = TypeVar("TCLScenarioCov", bound="CLScenario", covariant=True)  # Implementation, defined here

# Stream
TBaseCLStream = TypeVar('TBaseCLStream', bound='BaseCLStream')  # Implementation, defined here
TBaseCLStreamCov = TypeVar('TBaseCLStreamCov', bound='BaseCLStream', covariant=True)  # Implementation, defined here
TCLStream = TypeVar('TCLStream', bound='CLStream')  # Implementation, defined here
TCLStreamCov = TypeVar('TCLStreamCov', bound='CLStream', covariant=True)  # Implementation, defined here
TSequenceCLStream = TypeVar('TSequenceCLStream', bound='SequenceCLStream')

# Experience
TGenericExperience = TypeVar('TGenericExperience', bound='GenericExperienceProtocol')  # Protocol, defined here
TSettableGenericExperience = TypeVar('TSettableGenericExperience', bound='SettableGenericExperienceProtocol')  # Protocol, defined here
TCLExperience = TypeVar('TCLExperience', bound='CLExperience')  # Implementation, defined here


@runtime_checkable
class GenericExperienceProtocol(Protocol[TCLStreamCov]):
    @property
    def current_experience(self) -> int:
        """
        This is an incremental, 0-indexed, value used to keep track of the position 
        of current experience in the original stream.
        
        Beware that this value only describes the experience position in the 
        original stream and may be unrelated to the order in which the strategy will
        encounter experiences.
        """
        ...

    @property
    def origin_stream(self) -> TCLStreamCov:
        """
        A reference to the original stream from which this experience was obtained.
        """
        ...


@runtime_checkable
class DatasetExperienceProtocol(GenericExperienceProtocol[TCLStreamCov], Protocol[TCLScenarioCov, TCLStreamCov, TCLDatasetCov]):

    @property
    def benchmark(self) -> TCLScenarioCov:
        """
        A reference to the benchmark.
        """
        ...


    @property
    def dataset(self) -> TCLDatasetCov:
        """
        The dataset containing the patterns available in this experience.
        """
        ...

    @property
    def task_labels(self) -> List[int]:
        """
        This list will contain the unique task labels of the patterns contained
        in this experience. In the most common scenarios this will be a list
        with a single value. Note: for scenarios that don't produce task labels,
        a placeholder task label value like 0 is usually set to each pattern
        (see the description of the originating scenario for details).
        """
        ...

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
        ...

    @property
    def scenario(self) -> TCLScenarioCov:  # TODO: remove implementation from protocol
        """This property is DEPRECATED, use self.benchmark instead."""
        warnings.warn(
            "Using self.scenario is deprecated in Experience. "
            "Consider using self.benchmark instead.",
            stacklevel=2,
        )
        return self.benchmark


class SettableGenericExperienceProtocol(GenericExperienceProtocol[TCLStream], Protocol):
    @property
    def origin_stream(self) -> TCLStream:
        ...

    @origin_stream.setter
    @abstractmethod
    def origin_stream(self, stream: TCLStream):
        ...

    @property
    def current_experience(self) -> int:
        ...

    @current_experience.setter
    @abstractmethod
    def current_experience(self, id: int):
        ...


class SettableDatasetExperienceProtocol(DatasetExperienceProtocol[TCLScenario, TCLStream, TCLDataset], SettableGenericExperienceProtocol[TCLStream], Protocol):
    @property
    def benchmark(self) -> TCLScenario:
        ...

    @benchmark.setter
    @abstractmethod
    def benchmark(self, bench: TCLScenario):
        ...

    @property
    def dataset(self) -> TCLDataset:
        ...

    @dataset.setter
    @abstractmethod
    def dataset(self, d: TCLDataset):
        ...
    
    # task_label and task_labels are kept as read-only


@runtime_checkable
class ClassificationExperienceProtocol(DatasetExperienceProtocol[TCLScenarioCov, TCLStreamCov, ClassificationDataset], Protocol):
    """Definition of a classification experience.

    A classification experience contains a set of patterns
    which has become available at a particular time instant. The content and
    size of an Experience is defined by the specific benchmark that creates the
    experience instance.

    For instance, an experience of a New Classes scenario will contain all
    patterns belonging to a subset of classes of the original training set. An
    experience of a New Instance scenario will contain patterns from previously
    seen classes.

    Experiences of Single Incremental Task (a.k.a. task-free) scenarios are
    usually called "batches" while in Multi Task scenarios an Experience is
    usually associated to a "task". Finally, in a Multi Incremental Task
    scenario the Experience may be composed by patterns from different tasks.
    """
    ...


@runtime_checkable
class DetectionExperienceProtocol(DatasetExperienceProtocol[TCLScenarioCov, TCLStreamCov, ClassificationDataset], Protocol):
    """Definition of a detection experience.

    A detection experience contains a set of patterns
    which has become available at a particular time instant. The content and
    size of an Experience is defined by the specific benchmark that creates the
    experience instance.
    """
    ...


# Define alias to maintain compatibility with other parts of Avalanche
ClassificationExperience = ClassificationExperienceProtocol


class MaskedAttributeError(ValueError):
    """An error that is thrown when the user tries to access experience
    attributes which are private in the current experience's mode"""

    pass


class ExperienceMode(Enum):
    """ExperienceMode is an enum used to change visibility of experience's
    attributes.

    Example: task labels may be available during training but not evaluation.

    Current modes:
    - TRAIN: training time (e.g. train method in strategies).
    - INFERENCE: evaluation time (e.g. eval method in strategies).
    - LOGGING: maximum visibility. Useful when computing metrics.
    """

    TRAIN = 1
    EVAL = 2
    LOGGING = 3


class ExperienceAttribute(Generic[TCov]):
    """Experience attributes are used to define data belonging to an
    experience which may only be available at train or eval time.

    For example, experiences often keep a reference to the entire stream,
    which should be accessible only by the loggers and evaluation system,
    but should never be used by the strategy in the train/eval loops.
    """

    def __init__(
        self, value: TCov, use_in_train: bool = False, use_in_eval: bool = False
    ):
        """Init.

        :param value: attribute value.
        :param use_in_train: if True the attribute is available at training
            time.
        :param use_in_eval: if True the attribute is available at evaluation
            time.
        """
        self.value: TCov = value
        self.use_in_train: bool = use_in_train
        self.use_in_eval: bool = use_in_eval


# experience attributes can be values of a generic type T
# or they can be wrapped with an ExperienceAttribute object to handle
# scope visibility
TExperienceAttribute = Union[T, ExperienceAttribute[T]]


class CLExperience(SettableGenericExperienceProtocol[TCLStream]):
    """Base Experience.

    Experiences have an index which track the experience's position
    inside the stream for evaluation purposes.
    """

    def __init__(self, current_experience: int, origin_stream: TCLStream):
        super().__init__()
        self._current_experience: int = current_experience
        """Experience identifier (the position in the origin_stream)."""

        self._origin_stream: TCLStream = origin_stream
        """Stream containing the experience."""

        self._exp_mode: ExperienceMode = ExperienceMode.LOGGING
        # used to block access to private info (e.g. task labels,
        # past experiences).

        self._unmask_context_depth = 0

        self._as_attributes('_current_experience')

    @property
    def current_experience(self) -> int:
        return self._current_experience

    @current_experience.setter
    def current_experience(self, id: int):
        self._current_experience = id

    @property
    def origin_stream(self) -> TCLStream:
        return self._origin_stream

    @origin_stream.setter
    def origin_stream(self, stream: TCLStream):
        self._origin_stream = stream

    @contextmanager
    def no_attribute_masking(self):
        try:
            self._unmask_context_depth += 1
            assert self._unmask_context_depth > 0
            yield
        finally:
            self._unmask_context_depth -= 1
            assert self._unmask_context_depth >= 0

    @property
    def are_attributes_masked(self) -> bool:
        return self._unmask_context_depth == 0

    def __getattribute__(self, item):
        """Custom getattribute.

        Check that ExperienceAttribute are available in train/eval mode.
        """
        v = super().__getattribute__(item)
        if isinstance(v, ExperienceAttribute):
            if not self.are_attributes_masked:
                return v.value
            elif self._exp_mode == ExperienceMode.TRAIN and v.use_in_train:
                return v.value
            elif self._exp_mode == ExperienceMode.EVAL and v.use_in_eval:
                return v.value
            elif self._exp_mode == ExperienceMode.LOGGING:
                return v.value
            else:
                mode = (
                    "train"
                    if self._exp_mode == ExperienceMode.TRAIN
                    else "eval"
                )
                se = (
                    f"Attribute {item} is not available for the experience "
                    f"in {mode} mode."
                )
                raise MaskedAttributeError(se)
        else:
            return v
        
    def __setattr__(self, name, value):
        try:
            v = self.__dict__[name]
        except KeyError:
            return super().__setattr__(name, value)

        if isinstance(v, ExperienceAttribute):
            if isinstance(value, ExperienceAttribute):
                super().__setattr__(name, value)
            else:
                v.value = value
        else:
            return super().__setattr__(name, value)

    def _as_attributes(self, *fields: str, use_in_train=False, use_in_eval=False):
        """
        Internal method used to transform plain object fields to ExperienceAttribute(s).

        This is needed to ensure that static type checkers will not consider those fields
        as being of type "ExperienceAttribute", as this may be detrimental on the user
        experience.
        """
        for field in fields:
            v = super().__getattribute__(field)
            if isinstance(v, ExperienceAttribute):
                if v.use_in_train != use_in_train:
                    raise RuntimeError(
                        f'Experience attribute {field} redefined with inconguent use_in_train field. Was {v.use_in_train}, overridden with {use_in_train}.'
                    )
                
                if v.use_in_eval != use_in_eval:
                    raise RuntimeError(
                        f'Experience attribute {field} redefined with inconguent use_in_eval field. Was {v.use_in_eval}, overridden with {use_in_train}.'
                    )
            else:
                setattr(self, field, ExperienceAttribute(
                    value=v,
                    use_in_train=use_in_train,
                    use_in_eval=use_in_eval
                ))

    def train(self: TCLExperience) -> TCLExperience:
        """Return training experience.

        This is a copy of the experience itself where the private data (e.g.
        experience IDs) is removed to avoid its use during training.
        """
        exp = copy(self)
        exp._exp_mode = ExperienceMode.TRAIN
        return exp

    def eval(self: TCLExperience) -> TCLExperience:
        """Return inference experience.

        This is a copy of the experience itself where the inference data (e.g.
        experience IDs) is available.
        """
        exp = copy(self)
        exp._exp_mode = ExperienceMode.EVAL
        return exp

    def logging(self: TCLExperience) -> TCLExperience:
        """Return logging experience.

        This is a copy of the experience itself where all the attributes are
        available. Useful for logging and metric computations.
        """
        exp = copy(self)
        exp._exp_mode = ExperienceMode.LOGGING
        return exp


class DatasetExperience(CLExperience[TCLStream], Generic[TCLScenario, TCLStream, TCLDataset], SettableDatasetExperienceProtocol[TCLScenario, TCLStream, TCLDataset], ABC):
    """Base Experience.

    Experiences have an index which track the experience's position
    inside the stream for evaluation purposes.
    """

    def __init__(self, current_experience: int, origin_stream: TCLStream,
                 benchmark: TCLScenario, dataset: TCLDataset):
        super().__init__(current_experience=current_experience, origin_stream=origin_stream)

        self._benchmark: TCLScenario = benchmark
        self._dataset: TCLDataset = dataset
    
    @property
    def benchmark(self) -> TCLScenario:
        return self._benchmark

    @benchmark.setter
    def benchmark(self, bench: TCLScenario):
        self._benchmark = bench

    @property
    def dataset(self) -> TCLDataset:
        return self._dataset

    @dataset.setter
    def dataset(self, d: TCLDataset):
        self._dataset = d

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

    @property
    @abstractmethod
    def task_labels(self) -> List[int]:
        pass


class AbstractClassTimelineExperience(
    DatasetExperience[TCLScenario, TCLStream, TCLDataset],
    ABC
):
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
        self,
        origin_stream: TCLStream,
        dataset: TCLDataset,
        current_experience: int,
        classes_in_this_exp: Optional[Sequence[int]],
        previous_classes: Optional[Sequence[int]],
        classes_seen_so_far: Optional[Sequence[int]],
        future_classes: Optional[Sequence[int]],
    ):
        """
        Creates an instance of an experience given the benchmark
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

        self.classes_in_this_experience: Optional[Sequence[int]] = classes_in_this_exp
        """ The list of classes in this experience """

        self.previous_classes: Optional[Sequence[int]] = previous_classes
        """ The list of classes in previous experiences """

        self.classes_seen_so_far: Optional[Sequence[int]] = classes_seen_so_far
        """ List of classes of current and previous experiences """

        self.future_classes: Optional[Sequence[int]] = future_classes
        """ The list of classes of next experiences """

        super().__init__(
            current_experience=current_experience,
            origin_stream=origin_stream,
            benchmark=origin_stream.benchmark, # type: ignore
            dataset=dataset
        )

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
    
    @property
    @abstractmethod
    def task_labels(self) -> List[int]:
        pass


class ExperienceWrapper(SettableGenericExperienceProtocol[TCLStream], Generic[TCLStream, TGenericExperience]):
    def  __init__(
            self,
            base_exp: TGenericExperience,
            current_experience: int,
            origin_stream: TCLStream):
        self.wrapped_exp: TGenericExperience = base_exp
        self._current_experience = current_experience
        self._origin_stream = origin_stream

    @property
    def current_experience(self) -> int:
        return self._current_experience

    @current_experience.setter
    def current_experience(self, id: int):
        self._current_experience = id

    @property
    def origin_stream(self) -> TCLStream:
        return self._origin_stream

    @origin_stream.setter
    def origin_stream(self, stream: TCLStream):
        self._origin_stream = stream

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.wrapped_exp, attr)
    
    @property
    def task_labels(self) -> List[int]:
        return getattr(self.wrapped_exp, 'task_labels')


class BaseCLStream(Generic[TCLScenario, TGenericExperience], Iterable[TGenericExperience], ABC):
    """A CL stream is a named iterator of experiences.

    In general, many streams may be generator and not explicit lists to avoid
    keeping many objects in memory.

    NOTE: streams should not be used by training strategies since they
    provide access to past, current, and future data.
    """
    def __init__(
        self,
        name: str,
        exps_iter: Iterable[TGenericExperience],
        benchmark: TCLScenario
    ):
        self.name: str = name
        """
        The name of the stream (for instance: "train", "test", "valid", ...).
        """
        self.exps_iter: Iterable[TGenericExperience] = exps_iter

        self.benchmark: TCLScenario = benchmark
        """
        A reference to the benchmark.
        """

    def __iter__(self) -> Iterator[TGenericExperience]:
        yield from self.exps_iter


class CLStream(BaseCLStream[TCLScenario, TSettableGenericExperience]):
    """A CL stream is a named iterator of experiences.

    In general, many streams may be generator and not explicit lists to avoid
    keeping many objects in memory.

    NOTE: streams should not be used by training strategies since they
    provide access to past, current, and future data.
    """

    def __init__(
        self,
        name: str,
        exps_iter: Iterable[TSettableGenericExperience],
        benchmark: TCLScenario,
        set_stream_info: bool = True,
    ):
        self.set_stream_info: bool = set_stream_info
        super().__init__(name=name, exps_iter=exps_iter, benchmark=benchmark)
        
    def __iter__(self) -> Iterator[TSettableGenericExperience]:
        exp: TSettableGenericExperience
        for i, exp in enumerate(self.exps_iter):
            if self.set_stream_info:
                exp.current_experience = i
                exp.origin_stream = self
            yield exp


class SizedCLStream(CLStream[TCLScenario, TSettableGenericExperience], ABC):
    """
    Abstract class for defining CLStreams whose size
    (number of experiences) is known.
    """

    def __init__(
        self,
        name: str,
        exps_iter: Iterable[TSettableGenericExperience],
        benchmark: TCLScenario,
        set_stream_info: bool = True,
    ):
        super().__init__(
            name=name,
            exps_iter=exps_iter,
            benchmark=benchmark,
            set_stream_info=set_stream_info)

    @abstractmethod
    def __len__(self) -> int:
        """
        Gets the number of experiences this stream it's made of.

        :return: The number of experiences in this stream.
        """
        pass


# Internal TypeVar
TCLScenarioAny = TypeVar('TCLScenarioAny', bound='CLScenario')


TCLStreamWrapper = TypeVar('TCLStreamWrapper', bound='CLStreamWrapper')
TSizedCLStreamWrapper = TypeVar('TSizedCLStreamWrapper', bound='SizedCLStreamWrapper')


class CLStreamWrapper(CLStream[TCLScenario, ExperienceWrapper[TCLStreamWrapper, TSettableGenericExperience]]):
    def __init__(
            self,
            name: str,
            benchmark: TCLScenario,
            wrapped_stream: BaseCLStream[TCLScenarioAny, TSettableGenericExperience]):
        self.wrapped_stream = wrapped_stream
        super().__init__(
            name=name,
            exps_iter=None, # type: ignore
            benchmark=benchmark,
            set_stream_info=True)

    def __iter__(self: TCLStreamWrapper) -> Iterator[ExperienceWrapper[TCLStreamWrapper, TSettableGenericExperience]]:
        exp: TSettableGenericExperience
        for i, exp in enumerate(self.wrapped_stream):
            exp_wrapped = ExperienceWrapper(exp, i, self)
            yield exp_wrapped


class SizedCLStreamWrapper(SizedCLStream[TCLScenario, ExperienceWrapper[TSizedCLStreamWrapper, TSettableGenericExperience]]):
    def __init__(
            self,
            name: str,
            benchmark: TCLScenario,
            wrapped_stream: SizedCLStream[TCLScenarioAny, TSettableGenericExperience]):
        self.wrapped_stream = wrapped_stream
        super().__init__(
            name=name,
            exps_iter=None, # type: ignore
            benchmark=benchmark,
            set_stream_info=True)

    def __iter__(self: TSizedCLStreamWrapper) -> Iterator[ExperienceWrapper[TSizedCLStreamWrapper, TSettableGenericExperience]]:
        exp: TSettableGenericExperience
        for i, exp in enumerate(self.wrapped_stream):
            exp_wrapped = ExperienceWrapper(exp, i, self)
            yield exp_wrapped

    def __len__(self):
        return len(self.wrapped_stream)


class SequenceCLStream(CLStream[TCLScenario, TSettableGenericExperience], 
                       Sequence[TSettableGenericExperience], ABC):
    def __init__(
        self,
        name: str,
        benchmark: TCLScenario,
        set_stream_info: bool = True,
        slice_ids: Optional[Iterable[int]] = None
    ):
        self.set_stream_info: bool = set_stream_info
        self.slice_ids: Optional[List[int]] = \
            list(slice_ids) if slice_ids is not None else None
        """
        Describes which experiences are contained in the current stream slice. 
        Can be None, which means that this object is the original stream.
        """
        super().__init__(
            name=name,
            exps_iter=self,
            benchmark=benchmark)

    def __iter__(self) -> Iterator[TSettableGenericExperience]:
        exp: TSettableGenericExperience
        for i in range(len(self)):
            exp = self[i]
            yield exp

    @overload
    def __getitem__(self, item: int, /) -> TSettableGenericExperience:
        ...

    @overload
    def __getitem__(self: TSequenceCLStream, item: slice, /) -> TSequenceCLStream:
        ...
    
    @final
    def __getitem__(self: TSequenceCLStream, item: Union[int, slice], /) -> Union[TSequenceCLStream, TSettableGenericExperience]:
        # This check allows CL streams slicing
        if isinstance(item, int):
            if item >= len(self):
                raise IndexError(
                    "Experience index out of bounds" + str(int(item))
                )

            curr_exp = item if self.slice_ids is None else self.slice_ids[item]

            exp = self._make_experience(curr_exp)
            if self.set_stream_info:
                exp.current_experience = curr_exp
                exp.origin_stream = self
            return exp
        else:
            new_slice = self._forward_slice(self.slice_ids, item)
            return self._make_slice(new_slice)

    def __len__(self) -> int:
        """
        Gets the number of experiences this stream it's made of.

        :return: The number of experiences in this stream.
        """
        if self.slice_ids is not None:
            return len(self.slice_ids)
        else:
            return self._full_length()

    def _forward_slice(self, *slices: Union[None, slice, Iterable[int]]) -> Optional[Iterable[int]]:
        any_slice = False
        indices = list(range(self._full_length()))
        for sl in slices:
            if sl is None:
                continue
            any_slice = True

            slice_indices = slice_alike_object_to_indices(
                slice_alike_object=sl,
                max_length=len(indices)
            )

            new_indices = [indices[x] for x in slice_indices]
            indices = new_indices

        if any_slice:
            return indices
        else:
            return None  # No slicing

    @abstractmethod
    def _full_length(self) -> int:
        """
        Gets the number of experiences in the originating stream (that is, the non-sliced stream).
        """
        pass

    @abstractmethod
    def _make_experience(self, experience_idx: int) -> TSettableGenericExperience:
        """
        Obtain the experience at the given position in the originating stream (that is, the non-sliced stream).
        """
        pass

    def _make_slice(self: TSequenceCLStream, experience_slice: Optional[Iterable[int]]) -> TSequenceCLStream:
        """
        Obtain a sub-stream given a list of indices of the experiences to include.
        
        Experience ids are the ones of the originating stream (that is, the non-sliced stream).
        """
        stream_copy = copy(self)
        stream_copy.slice_ids = list(experience_slice) if experience_slice is not None else None
        return stream_copy


TSequenceStreamWrapper = TypeVar('TSequenceStreamWrapper', bound='SequenceStreamWrapper')

class SequenceStreamWrapper(SequenceCLStream[TCLScenario, ExperienceWrapper[TSequenceStreamWrapper, TSettableGenericExperience]]):
    def __init__(
            self,
            name: str,
            benchmark: TCLScenario,
            wrapped_stream: SequenceCLStream[TCLScenarioAny, TSettableGenericExperience],
            slice_ids: Optional[Iterable[int]] = None
        ):
        self.wrapped_stream = wrapped_stream
        
        super().__init__(
            name,
            benchmark,
            set_stream_info=True,
            slice_ids=slice_ids)
        
    def _full_length(self) -> int:
        """
        Gets the number of experiences in the wrapped stream.
        """
        return len(self.wrapped_stream)

    def _make_experience(self: TSequenceStreamWrapper, experience_idx: int) -> ExperienceWrapper[TSequenceStreamWrapper, TSettableGenericExperience]:
        """
        Obtain the experience at the given position in the wrapped stream.
        """
        exp = self.wrapped_stream[experience_idx]
        wrapped_exp = ExperienceWrapper(exp, experience_idx, self)
        return wrapped_exp


class EagerCLStream(SequenceCLStream[TCLScenario, TSettableGenericExperience]):
    """A CL stream which is a named list of experiences.

    Eager streams are indexable and sliceable, like python lists.

    NOTE: streams should not be used by training strategies since they
    provide access to past, current, and future data.
    """

    def __init__(
        self,
        name: str,
        exps: Sequence[TSettableGenericExperience],
        benchmark: TCLScenario,
        set_stream_info: bool = True,
        slice_ids: Optional[Iterable[int]] = None
    ):
        """Create a CL stream given a list of experiences.

        `origin_stream` and `current_experience` are set for each experience in
        `exps`.

        :param name: name of the stream.
        :param exps: list of experiences.
        :param benchmark: a reference to the benchmark.
        :param set_stream_info: if True, set the `origin_stream` and
            `current_experience` identifier for each experience. If False,
            the attributes are left unchanged.
        """
        self._exps = list(exps)
        super().__init__(
            name=name,
            benchmark=benchmark,
            set_stream_info=set_stream_info,
            slice_ids=slice_ids)

        if self.set_stream_info:
            slice_ids_enum = self.slice_ids if self.slice_ids is not None else range(len(self._exps))
            for i in slice_ids_enum:
                exp = self._exps[i]
                exp.current_experience = i
                exp.origin_stream = self # type: ignore
            
            self.set_stream_info = False
    
    @property
    def exps(self) -> Tuple[TSettableGenericExperience, ...]:
        return tuple(self.exps_iter)

    def _full_length(self) -> int:
        return len(self._exps)

    def _make_experience(self, experience_idx: int) -> TSettableGenericExperience:
        return self._exps[experience_idx] 


class CLScenario(Generic[TBaseCLStreamCov]):
    """Continual Learning benchmark.

    A Continual Learning benchmark is a container for a set of streams of
    experiences. It may also contain other additional data useful for
    evaluation purposes or logging.

    The content of each experience depends on the underlying problem (data in a
    supervised problem, environments in reinforcement learning, and so on).

    NOTE: benchmarks should not be used by training strategies since they
    provide access to past, current, and future data.
    """

    def __init__(self, streams: Iterable[TBaseCLStreamCov]):
        """Creates an instance of a Continual Learning benchmark.

        :param streams: a list of streams.
        """
        self._streams: Dict[str, TBaseCLStreamCov] = dict()
        for s in streams:
            self._streams[s.name] = s
        for s in streams:
            self.__dict__[s.name + "_stream"] = s

    @property
    def streams(self):
        # we don't want in-place modifications so we return a copy
        return copy(self._streams)


__all__ = [
    "GenericExperienceProtocol",
    "DatasetExperienceProtocol",
    "SettableGenericExperienceProtocol",
    "SettableDatasetExperienceProtocol",
    "ClassificationExperienceProtocol",
    "DetectionExperienceProtocol",
    "ClassificationExperience",
    "MaskedAttributeError",
    "ExperienceMode",
    "ExperienceAttribute",
    "CLExperience",
    "DatasetExperience",
    "AbstractClassTimelineExperience",
    "BaseCLStream",
    "CLStream",
    "SequenceCLStream",
    "SequenceStreamWrapper",
    "EagerCLStream",
    "CLScenario"
]
