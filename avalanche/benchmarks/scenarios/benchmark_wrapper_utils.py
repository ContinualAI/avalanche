from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
)

from avalanche.benchmarks.scenarios.generic_scenario import (
    CLExperience,
    CLScenario,
    CLStream,
    SequenceCLStream,
    SizedCLStream,
)

TCLExperience = TypeVar("TCLExperience", bound="CLExperience")
TCLStreamWrapper = TypeVar("TCLStreamWrapper", bound="CLStreamWrapper")
TSizedCLStreamWrapper = TypeVar("TSizedCLStreamWrapper", bound="SizedCLStreamWrapper")
TSequenceStreamWrapper = TypeVar(
    "TSequenceStreamWrapper", bound="SequenceStreamWrapper"
)


class ExperienceWrapper(CLExperience, Generic[TCLExperience]):
    """
    Utility class used to wrap an experience.

    Instances of this class will allow to get attrbitues setted
    in the original experience, but the `origin_stream` and
    `current_experience` attributes will be overridden.
    """

    def __init__(
        self, base_exp: TCLExperience, current_experience: int, origin_stream: CLStream
    ):
        self.wrapped_exp: TCLExperience = base_exp
        super().__init__(
            current_experience=current_experience, origin_stream=origin_stream
        )

    def __getattr__(self, attr):
        if attr == "wrapped_exp" and attr not in self.__dict__:
            # Happens when using copy.copy or copy.deepcopy
            raise AttributeError(attr)

        if attr in self.__dict__:
            return self.__dict__[attr]

        return getattr(self.wrapped_exp, attr)

    @property
    def task_labels(self) -> List[int]:
        return getattr(self.wrapped_exp, "task_labels")


class CLStreamWrapper(CLStream[ExperienceWrapper[TCLExperience]]):
    """
    Utility class used to wrap a stream.

    Objects of this class will return experiences wrapped
    using class:`ExperienceWrapper`.
    """

    def __init__(
        self, name: str, benchmark: CLScenario, wrapped_stream: CLStream[TCLExperience]
    ):
        self._wrapped_stream: CLStream[TCLExperience] = wrapped_stream
        """
        A reference to the wrapped stream.
        """

        super().__init__(
            name=name,
            exps_iter=None,  # type: ignore
            benchmark=benchmark,
            set_stream_info=True,
        )

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._wrapped_exp, attr)

    def __iter__(self) -> Iterator[ExperienceWrapper[TCLExperience]]:
        exp: TCLExperience
        for i, exp in enumerate(self._wrapped_stream):
            exp_wrapped = ExperienceWrapper(exp, i, self)
            yield exp_wrapped


class SizedCLStreamWrapper(CLStreamWrapper[TCLExperience]):
    """
    Utility class used to wrap a sized stream.

    Objects of this class will return experiences wrapped
    using class:`ExperienceWrapper`.
    """

    def __init__(
        self,
        name: str,
        benchmark: CLScenario,
        wrapped_stream: SizedCLStream[TCLExperience],
    ):
        self._wrapped_stream: SizedCLStream[TCLExperience] = wrapped_stream

        super().__init__(name=name, benchmark=benchmark, wrapped_stream=wrapped_stream)

    def __len__(self):
        return len(self._wrapped_stream)


class SequenceStreamWrapper(SequenceCLStream[ExperienceWrapper[TCLExperience]]):
    """
    Utility class used to wrap a sequence stream.

    Objects of this class will return experiences wrapped
    using class:`ExperienceWrapper`.
    """

    def __init__(
        self,
        name: str,
        benchmark: CLScenario,
        wrapped_stream: SequenceCLStream[TCLExperience],
        slice_ids: Optional[Iterable[int]] = None,
    ):
        self._wrapped_stream: SequenceCLStream[TCLExperience] = wrapped_stream

        super().__init__(name, benchmark, set_stream_info=True, slice_ids=slice_ids)

    def _full_length(self) -> int:
        """
        Gets the number of experiences in the wrapped stream.
        """
        return len(self._wrapped_stream)

    def _make_experience(self, experience_idx: int) -> ExperienceWrapper[TCLExperience]:
        """
        Obtain the experience at the given position in the wrapped stream.
        """
        exp = self._wrapped_stream[experience_idx]
        wrapped_exp = ExperienceWrapper(exp, experience_idx, self)
        return wrapped_exp


def wrap_stream(
    new_name: str, new_benchmark: CLScenario, wrapped_stream: CLStream
) -> CLStream:
    """
    Internal utility used to wrap a stream by keeping
    as most functionality as possible.

    :param new_name: The name of the new stream.
    :param new_benchmark: The new benchmark.
    :param wrapped_stream: The stream to be wrapped.
    """
    s_wrapped: CLStream
    if isinstance(wrapped_stream, SequenceCLStream):
        # Maintain indexing/slicing functionalities
        s_wrapped = SequenceStreamWrapper(
            name=new_name, benchmark=new_benchmark, wrapped_stream=wrapped_stream
        )
    elif isinstance(wrapped_stream, SizedCLStream):
        # Sized stream
        s_wrapped = SizedCLStreamWrapper(
            name=new_name, benchmark=new_benchmark, wrapped_stream=wrapped_stream
        )
    else:
        # Plain iter-based stream
        s_wrapped = CLStreamWrapper(
            name=new_name, benchmark=new_benchmark, wrapped_stream=wrapped_stream
        )
    return s_wrapped


__all__ = ["wrap_stream"]
