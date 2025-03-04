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
from types import GeneratorType
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    Generic,
    overload,
    final,
)

import numpy as np

from avalanche.benchmarks.utils.dataset_utils import (
    slice_alike_object_to_indices,
)


# Typing
T = TypeVar("T")
TCov = TypeVar("TCov", covariant=True)
TCLStream = TypeVar("TCLStream", bound="CLStream")
TSequenceCLStream = TypeVar("TSequenceCLStream", bound="SequenceCLStream")
TCLExperience = TypeVar("TCLExperience", bound="CLExperience")


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


class CLExperience:
    """
    Base Experience.

    Experiences have an index which track the experience's position
    inside the stream for evaluation purposes.
    """

    def __init__(
        self: TCLExperience,
        current_experience: int,
        origin_stream: "Optional[CLStream[TCLExperience]]",
    ):
        super().__init__()
        self._current_experience: int = current_experience
        """Experience identifier (the position in the origin_stream)."""

        self._origin_stream: "Optional[CLStream[TCLExperience]]" = origin_stream
        """Stream containing the experience."""

        self._exp_mode: ExperienceMode = ExperienceMode.LOGGING
        # used to block access to private info (e.g. task labels,
        # past experiences).

        self._unmask_context_depth = 0

        self._as_attributes("_current_experience")

    @property
    def current_experience(self) -> int:
        curr_exp = self._current_experience
        CLExperience._check_unset_attribute("current_experience", curr_exp)
        return curr_exp

    @current_experience.setter
    def current_experience(self, id: int):
        self._current_experience = id

    @property
    def origin_stream(self: TCLExperience) -> "CLStream[TCLExperience]":
        orig_stream = self._origin_stream
        CLExperience._check_unset_attribute("origin_stream", orig_stream)
        return orig_stream

    @origin_stream.setter
    def origin_stream(self: TCLExperience, stream: "CLStream[TCLExperience]"):
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
                mode = "train" if self._exp_mode == ExperienceMode.TRAIN else "eval"
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
        Internal method used to transform plain object fields to
        ExperienceAttribute(s).

        This is needed to ensure that static type checkers will not consider
        those fields as being of type "ExperienceAttribute", as this may be
        detrimental on the user experience.
        """
        for field in fields:
            v = super().__getattribute__(field)
            if isinstance(v, ExperienceAttribute):
                if v.use_in_train != use_in_train:
                    raise RuntimeError(
                        f"Experience attribute {field} redefined with "
                        f"incongruent use_in_train field. Was "
                        f"{v.use_in_train}, overridden with {use_in_train}."
                    )

                if v.use_in_eval != use_in_eval:
                    raise RuntimeError(
                        f"Experience attribute {field} redefined with "
                        f"incongruent use_in_eval field. Was "
                        f"{v.use_in_eval}, overridden with {use_in_train}."
                    )
            else:
                setattr(
                    self,
                    field,
                    ExperienceAttribute(
                        value=v, use_in_train=use_in_train, use_in_eval=use_in_eval
                    ),
                )

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

    @staticmethod
    def _check_unset_attribute(attribute_name: str, attribute_value: Any):
        assert attribute_value is not None, (
            f"Attribute {attribute_name} "
            + "not set. This is an unexpected and usually liked to errors "
            + "in the implementation of the stream's experience factory."
        )


class GeneratorMemo(Generic[T]):
    def __init__(self, generator: Generator[T, None, None]):
        self._generator: Optional[Generator[T, None, None]] = generator
        self._already_generated: List[T] = []

    def __iter__(self):
        idx = 0
        while True:
            if idx < len(self._already_generated):
                yield self._already_generated[idx]
            else:
                if self._generator is None:
                    break
                try:
                    next_item = next(self._generator)
                except StopIteration:
                    self._generator = None
                    break
                self._already_generated.append(next_item)
                yield next_item
            idx += 1


class CLStream(Generic[TCLExperience]):
    """A CL stream is a named iterator of experiences.

    In general, many streams may be generator and not explicit lists to avoid
    keeping many objects in memory.

    NOTE: streams should not be used by training strategies since they
    provide access to past, current, and future data.
    """

    def __init__(
        self: TCLStream,
        name: str,
        exps_iter: Iterable[TCLExperience],
        benchmark: "Optional[CLScenario[TCLStream]]" = None,
        set_stream_info: bool = True,
    ):
        """
        Creates an instance of a experience stream.

        :param name: The name of the stream.
        :param exps_iter: The iterable from which experiences will be obtained.
        :param benchmark: The benchmarks defining this stream.
        :param set_stream_info: If True, will set the `current_experience` and
            `origin_stream` fields on experience objects before returning them.
            Defaults to True.
        """
        self.name: str = name
        """
        The name of the stream (for instance: "train", "test", "valid", ...).
        """

        self.exps_iter: Iterable[TCLExperience] = exps_iter
        """
        The iterable from which experiences will be obtained.
        """

        self.benchmark: "CLScenario[TCLStream]" = benchmark
        """
        A reference to the benchmark.
        """

        self.set_stream_info: bool = set_stream_info
        """
        If True, will set the `current_experience` and `origin_stream` 
        fields on experience objects before returning them.
        """

        if isinstance(self.exps_iter, GeneratorType):
            # Prevent issues when iterating the stream more than once
            self.exps_iter = GeneratorMemo(self.exps_iter)

    def __iter__(self) -> Iterator[TCLExperience]:
        exp: TCLExperience
        for i, exp in enumerate(self.exps_iter):
            if self.set_stream_info:
                exp.current_experience = i
                exp.origin_stream = self
            yield exp


class SizedCLStream(CLStream[TCLExperience], ABC):
    """
    Abstract class for defining CLStreams whose size
    (number of experiences) is known.
    """

    def __init__(
        self: TCLStream,
        name: str,
        exps_iter: Iterable[TCLExperience],
        benchmark: "Optional[CLScenario[TCLStream]]" = None,
        set_stream_info: bool = True,
    ):
        super().__init__(
            name=name,
            exps_iter=exps_iter,
            benchmark=benchmark,
            set_stream_info=set_stream_info,
        )

    @abstractmethod
    def __len__(self) -> int:
        """
        Gets the number of experiences this stream it's made of.

        :return: The number of experiences in this stream.
        """
        pass


class SequenceCLStream(SizedCLStream[TCLExperience], Sequence[TCLExperience], ABC):
    """
    Defines a stream that behaves like a :class:`Sequence`.

    This is the most common base class for streams in Avalanche as
    it implements the basic indexing and slicing functionalities
    for streams.
    """

    def __init__(
        self,
        name: str,
        benchmark: "CLScenario",
        set_stream_info: bool = True,
        slice_ids: Optional[Iterable[int]] = None,
    ):
        self.slice_ids: Optional[List[int]] = (
            list(slice_ids) if slice_ids is not None else None
        )
        """
        Describes which experiences are contained in the current stream slice. 
        Can be None, which means that this object is the original stream.
        """

        super().__init__(
            name=name,
            exps_iter=self,
            benchmark=benchmark,
            set_stream_info=set_stream_info,
        )

    def __iter__(self) -> Iterator[TCLExperience]:
        exp: TCLExperience
        for i in range(len(self)):
            exp = self[i]
            yield exp

    @overload
    def __getitem__(self, item: int) -> TCLExperience:
        ...

    @overload
    def __getitem__(self: TSequenceCLStream, item: slice) -> TSequenceCLStream:
        ...

    @final
    def __getitem__(
        self: TSequenceCLStream, item: Union[int, slice]
    ) -> Union[TSequenceCLStream, TCLExperience]:
        # This check allows CL streams slicing
        if isinstance(item, (int, np.integer)):
            item = int(item)
            if item >= len(self):
                raise IndexError("Experience index out of bounds" + str(int(item)))

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

    def _forward_slice(
        self, *slices: Union[None, slice, Iterable[int]]
    ) -> Optional[Iterable[int]]:
        any_slice = False
        indices = list(range(self._full_length()))
        for sl in slices:
            if sl is None:
                continue
            any_slice = True

            slice_indices = slice_alike_object_to_indices(
                slice_alike_object=sl, max_length=len(indices)
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
        Gets the number of experiences in the originating stream
        (that is, the non-sliced stream).
        """
        pass

    @abstractmethod
    def _make_experience(self, experience_idx: int) -> TCLExperience:
        """
        Obtain the experience at the given position in the originating
        stream (that is, the non-sliced stream).
        """
        pass

    def _make_slice(
        self: TSequenceCLStream, experience_slice: Optional[Iterable[int]]
    ) -> TSequenceCLStream:
        """
        Obtain a sub-stream given a list of indices of the experiences to
        include.

        Experience ids are the ones of the originating stream
        (that is, the non-sliced stream).
        """
        stream_copy = copy(self)
        stream_copy.slice_ids = (
            list(experience_slice) if experience_slice is not None else None
        )
        return stream_copy


class EagerCLStream(SequenceCLStream[TCLExperience]):
    """
    A CL stream build from a pre-initialized list of experience.

    NOTE: streams should not be used by training strategies since they
    provide access to past, current, and future data.
    """

    def __init__(
        self,
        name: str,
        exps: Sequence[TCLExperience],
        benchmark: "Optional[CLScenario]" = None,
        set_stream_info: bool = True,
        slice_ids: Optional[Iterable[int]] = None,
    ):
        """Create a CL stream given a list of experiences.
        :param name: name of the stream.
        :param exps: list of experiences.
        :param benchmark: a reference to the benchmark.
        :param set_stream_info: if True, set the `origin_stream` and
            `current_experience` identifier for each experience. If False,
            the attributes are left unchanged.
        :param slice_ids: The indices of experiences to include. from the
            original stream. Defaults to None. For internal use.
        """
        self._exps: List[TCLExperience] = list(exps)
        super().__init__(
            name=name,
            benchmark=benchmark,
            set_stream_info=set_stream_info,
            slice_ids=slice_ids,
        )

        if self.set_stream_info:
            slice_ids_enum = (
                self.slice_ids if self.slice_ids is not None else range(len(self._exps))
            )
            for i in slice_ids_enum:
                exp = self._exps[i]
                exp.current_experience = i
                exp.origin_stream = self  # type: ignore

            self.set_stream_info = False

    @property
    def exps(self) -> Tuple[TCLExperience, ...]:
        return tuple(self.exps_iter)

    def _full_length(self) -> int:
        return len(self._exps)

    def _make_experience(self, experience_idx: int) -> TCLExperience:
        return self._exps[experience_idx]


class CLScenario(Generic[TCLStream]):
    """
    Continual Learning benchmark.

    A Continual Learning benchmark is a container for a set of streams of
    experiences. It may also contain other additional data useful for
    evaluation purposes or logging.

    The content of each experience depends on the underlying problem (data in a
    supervised problem, environments in reinforcement learning, and so on).

    NOTE: benchmarks should not be used by training strategies since they
    provide access to past, current, and future data.
    """

    def __init__(self, streams: Iterable[TCLStream]):
        """Creates an instance of a Continual Learning benchmark.

        :param streams: a list of streams.
        """
        self._streams: Dict[str, TCLStream] = dict()
        for s in streams:
            self._streams[s.name] = s
        for s in streams:
            self.__dict__[s.name + "_stream"] = s

    @property
    def streams(self):
        # we don't want in-place modifications so we return a copy
        return copy(self._streams)


def make_stream(name: str, exps: Iterable[CLExperience]) -> CLStream:
    """Internal utility used to create a stream.

    Uses the correct class for generators, sized generators, and lists.

    :param new_name: The name of the new stream.
    :param exps: sequence of experiences.
    """
    s_wrapped: CLStream
    if isinstance(exps, List):  # Maintain indexing/slicing functionalities
        return EagerCLStream(name=name, exps=exps)
    elif hasattr(exps, "__len__"):  # Sized stream
        return SizedCLStream(name=name, exps_iter=exps)
    else:  # Plain iterator
        return CLStream(name=name, exps_iter=exps)


__all__ = [
    "MaskedAttributeError",
    "ExperienceMode",
    "ExperienceAttribute",
    "CLExperience",
    "CLStream",
    "SequenceCLStream",
    "EagerCLStream",
    "CLScenario",
    "make_stream",
]
