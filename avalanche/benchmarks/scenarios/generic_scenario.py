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
from copy import copy
from enum import Enum
from typing import Generator, List, Iterable, TypeVar, Union, Generic

T = TypeVar("T")
E = TypeVar("E")


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


class ExperienceAttribute(Generic[T]):
    """Experience attributes are used to define data belonging to an
    experience which may only be available at train or eval time.

    For example, experiences often keep a reference to the entire stream,
    which should be accessible only by the loggers and evaluation system,
    but should never be used by the strategy in the train/eval loops.
    """

    def __init__(
        self, value: T, use_in_train: bool = False, use_in_eval: bool = False
    ):
        """Init.

        :param value: attribute value.
        :param use_in_train: if True the attribute is available at training
            time.
        :param use_in_eval: if True the attribute is available at evaluation
            time.
        """
        self.value = value
        self.use_in_train = use_in_train
        self.use_in_eval = use_in_eval


# experience attributes can be values of a generic type T
# or they can be wrapped with an ExperienceAttribute object to handle
# scope visibility
TExperienceAttribute = Union[T, ExperienceAttribute[T]]


class CLExperience(object):
    """Base Experience.

    Experiences have an index which track the experience's position
    inside the stream for evaluation purposes.
    """

    def __init__(self, current_experience: int = None, origin_stream=None):
        super().__init__()
        self.current_experience: TExperienceAttribute = ExperienceAttribute(
            current_experience
        )
        """Experience identifier (the position in the origin_stream)."""

        self.origin_stream: TExperienceAttribute = ExperienceAttribute(
            origin_stream
        )
        """Stream containing the experience."""

        self._exp_mode: ExperienceMode = ExperienceMode.TRAIN
        # used to block access to private info (e.g. task labels,
        # past experiences).

    def __getattribute__(self, item):
        """Custom getattribute.

        Check that ExperienceAttribute are available in train/eval mode.
        """
        v = super().__getattribute__(item)
        if isinstance(v, ExperienceAttribute):
            if self._exp_mode == ExperienceMode.TRAIN and v.use_in_train:
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

    def train(self):
        """Return training experience.

        This is a copy of the experience itself where the private data (e.g.
        experience IDs) is removed to avoid its use during training.
        """
        exp = copy(self)
        exp._exp_mode = ExperienceMode.TRAIN
        return exp

    def eval(self):
        """Return inference experience.

        This is a copy of the experience itself where the inference data (e.g.
        experience IDs) is available.
        """
        exp = copy(self)
        exp._exp_mode = ExperienceMode.EVAL
        return exp

    def logging(self):
        """Return logging experience.

        This is a copy of the experience itself where all the attributes are
        available. Useful for logging and metric computations.
        """
        exp = copy(self)
        exp._exp_mode = ExperienceMode.LOGGING
        return exp


class CLStream(Generic[E]):
    """A CL stream is a named iterator of experiences.

    In general, many streams may be generator and not explicit lists to avoid
    keeping many objects in memory.

    NOTE: streams should not be used by training strategies since they
    provide access to past, current, and future data.
    """

    def __init__(
        self,
        name: str,
        exps_iter: Iterable[CLExperience],
        benchmark=None,
        set_stream_info: bool = True,
    ):
        self.name = name
        self.exps_iter = exps_iter

        self.benchmark = benchmark
        self.set_stream_info = set_stream_info

        self._iter = None

    def __iter__(self) -> Generator[E, None, None]:
        def foo(self):
            for i, exp in enumerate(self.exps_iter):
                if self.set_stream_info:
                    exp.current_experience = i
                    exp.origin_stream = self
                yield exp

        return foo(self)


class EagerCLStream(CLStream[E]):
    """A CL stream which is a named list of experiences.

    Eager streams are indexable and sliceable, like python lists.

    NOTE: streams should not be used by training strategies since they
    provide access to past, current, and future data.
    """

    def __init__(
        self,
        name: str,
        exps: List[CLExperience],
        benchmark=None,
        set_stream_info: bool = True,
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
        super().__init__(name, exps, benchmark, set_stream_info)
        self.exps = exps

        for i, e in enumerate(self.exps):
            if self.set_stream_info:
                e.origin_stream = self
                e.current_experience = i

    def __getitem__(self, item) -> Union["EagerCLStream[E]", E]:
        # This check allows CL streams slicing
        if isinstance(item, slice):
            return EagerCLStream(
                name=self.name, exps=self.exps[item], set_stream_info=False
            )
        else:
            return self.exps[item]

    def __len__(self):
        return len(self.exps)


class CLScenario:
    """Continual Learning benchmark.

    A Continual Learning benchmark is a container for a set of streams of
    experiences. It may also contain other additional data useful for
    evaluation purposes or logging.

    The content of each experience depends on the underlying problem (data in a
    supervised problem, environments in reinforcement learning, and so on).

    NOTE: benchmarks should not be used by training strategies since they
    provide access to past, current, and future data.
    """

    def __init__(self, streams: List[CLStream]):
        """Creates an instance of a Continual Learning benchmark.

        :param streams: a list of streams.
        """
        self._streams = {}
        for s in streams:
            self._streams[s.name + "_stream"] = s
        for s in streams:
            self.__dict__[s.name + "_stream"] = s

    @property
    def streams(self):
        # we don't want in-place modifications so we return a copy
        return copy(self._streams)


__all__ = [
    "ExperienceAttribute",
    "CLExperience",
    "CLStream",
    "EagerCLStream",
    "CLScenario",
]
