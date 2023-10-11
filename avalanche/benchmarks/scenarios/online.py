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
from functools import partial
import random
from typing import (
    Callable,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union, Protocol,
)
from typing_extensions import Literal
import warnings
from avalanche.benchmarks.utils.data import AvalancheDataset

import torch
from .deprecated.benchmark_wrapper_utils import wrap_stream

from .generic_scenario import (
    CLStream,
    CLScenario,
)
from .dataset_scenario import DatasetExperience


TCLDataset = TypeVar("TCLDataset", bound="AvalancheDataset")
TCLScenario = TypeVar("TCLScenario", bound="CLScenario")
TDatasetScenario = TypeVar("TDatasetScenario", bound="DatasetScenario")
TOnlineCLScenario = TypeVar("TOnlineCLScenario", bound="OnlineCLScenario")
TCLStream = TypeVar("TCLStream", bound="CLStream")
TCLExperience = TypeVar("TCLExperience", bound="CLExperience")
TOnlineCLExperience = TypeVar("TOnlineCLExperience", bound="OnlineCLExperience")


class BoundaryAware(Protocol):
    """Boundary-aware experiences have attributes with task boundary knowledge.

    Online streams may have boundary attributes to help training or metrics logging.

    Task boundaries denote changes of the underlying data distribution used
    to sample the data for the experiences.
    """

    @property
    def is_first_subexp(self) -> bool:
        """True if this is the first experience after a drift."""
        return False

    @property
    def is_last_subexp(self) -> bool:
        """True if this is the last experience before a drift."""
        return False

    @property
    def sub_stream_length(self) -> int:
        """Number of experiences with the same distribution of the current experience."""
        return 0

    @property
    def access_task_boundaries(self) -> bool:
        """True if the model has access to task boundaries.

        If the model is boundary-agnostic, task boundaries are available only
        for logging by setting the experience in logging mode `experience.logging()`.
        """
        return False


class OnlineCLExperience(DatasetExperience, Generic[TCLDataset]):
    """Online CL (OCL) Experience.

    OCL experiences are created by splitting a larger experience. Therefore,
    they keep track of the original experience for logging purposes.
    """

    def __init__(
        self: TOnlineCLExperience,
        *,
        dataset: TCLDataset,
        origin_experience: DatasetExperience,
        is_first_subexp: bool = False,
        is_last_subexp: bool = False,
        sub_stream_length: Optional[int] = None,
        access_task_boundaries: bool = False,
    ):
        """A class representing a continual learning experience in an online setting.

        :param current_experience: The index of the current experience.
        :type current_experience: int
        :param dataset: The dataset containing the experience.
        :type dataset: TCLDataset
        :param origin_experience: The original experience from which this experience was derived.
        :type origin_experience: DatasetExperience
        :param is_first_subexp: Whether this is the first sub-experience.
        :type is_first_subexp: bool, optional
        :param is_last_subexp: Whether this is the last sub-experience.
        :type is_last_subexp: bool, optional
        :param sub_stream_length: The length of the sub-stream.
        :type sub_stream_length: int, optional
        :param access_task_boundaries: Whether to access task boundaries.
        :type access_task_boundaries: bool, optional
        """       
        super().__init__(dataset=dataset)
        self.access_task_boundaries = access_task_boundaries
        self.origin_experience: DatasetExperience = origin_experience
        self.subexp_size: int = len(dataset)
        self.is_first_subexp: bool = is_first_subexp
        self.is_last_subexp: bool = is_last_subexp
        self.sub_stream_length: Optional[int] = sub_stream_length

        self._as_attributes(
            "origin_experience",
            "subexp_size",
            "is_first_subexp",
            "is_last_subexp",
            "sub_stream_length",
            use_in_train=access_task_boundaries,
            use_in_eval=access_task_boundaries,
        )


class FixedSizeExperienceSplitter:
    def __init__(
        self,
        experience: DatasetExperience,
        experience_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        access_task_boundaries: bool = False,
    ) -> Generator[OnlineCLExperience, None, None]:
        """Returns a lazy stream generated by splitting an experience into smaller
        ones.

        Splits the experience in smaller experiences of size `experience_size`.

        Experience decorators (e.g. class attributes) will be stripped from the experience.
        You will need to re-apply them to the resulting experiences if you need them.
        
        :param experience: The experience to split.
        :param experience_size: The experience size (number of instances).
        :param shuffle: If True, instances will be shuffled before splitting.
        :param drop_last: If True, the last mini-experience will be dropped if
            not of size `experience_size`
        :return: The list of datasets that will be used to create the
            mini-experiences.
        """
        self.experience = experience
        self.experience_size = experience_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.access_task_boundaries = access_task_boundaries

        # we need to fix the seed because repeated calls to the generator
        # must return the same order every time.
        self.seed = random.randint(0, 2 ** 32 - 1)

    def __iter__(self):
        exp_dataset = self.experience.dataset
        exp_indices = list(range(len(exp_dataset)))

        g = torch.Generator()
        g.manual_seed(self.seed)

        if self.shuffle:
            exp_indices = torch.as_tensor(exp_indices)[
                torch.randperm(len(exp_indices), generator=g)
            ].tolist()
        sub_stream_length = len(exp_indices) // self.experience_size
        if not self.drop_last and len(exp_indices) % self.experience_size > 0:
            sub_stream_length += 1

        init_idx = 0
        is_first = True
        is_last = False
        exp_idx = 0
        while init_idx < len(exp_indices):
            final_idx = init_idx + self.experience_size  # Exclusive
            if final_idx > len(exp_indices):
                if self.drop_last:
                    break

                final_idx = len(exp_indices)
                is_last = True

            sub_exp_subset = exp_dataset.subset(exp_indices[init_idx:final_idx])
            exp = OnlineCLExperience(
                dataset=sub_exp_subset,
                origin_experience=self.experience,
                is_first_subexp=is_first,
                is_last_subexp=is_last,
                sub_stream_length=sub_stream_length,
                access_task_boundaries=self.access_task_boundaries,
            )

            is_first = False
            yield exp
            init_idx = final_idx
            exp_idx += 1


def _default_online_split(
    shuffle: bool,
    drop_last: bool,
    access_task_boundaries: bool,
    exp: DatasetExperience,
    size: int,
):
    return FixedSizeExperienceSplitter(
        experience=exp,
        experience_size=size,
        shuffle=shuffle,
        drop_last=drop_last,
        access_task_boundaries=access_task_boundaries,
    )


def split_online_stream(
    original_stream: Iterable[DatasetExperience],
    experience_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    experience_split_strategy: Optional[
        Callable[
            [DatasetExperience[TCLDataset], int],
            Iterable[OnlineCLExperience[TCLDataset]],
        ]
    ] = None,
    access_task_boundaries: bool = False,
) -> CLStream[DatasetExperience[TCLDataset]]:
    """Split a stream of large batches to create an online stream of small
    mini-batches.

    The resulting stream can be used for Online Continual Learning (OCL)
    scenarios (or data-incremental, or other online-based settings).

    For efficiency reasons, the resulting stream is an iterator, generating
    experience on-demand.

    :param original_stream: The stream with the original data.
    :param experience_size: The size of the experience, as an int. Ignored
        if `custom_split_strategy` is used.
    :param shuffle: If True, experiences will be split by first shuffling
        instances in each experience. This will use the default PyTorch
        random number generator at its current state. Defaults to False.
        Ignored if `experience_split_strategy` is used.
    :param drop_last: If True, if the last experience doesn't contain
        `experience_size` instances, then the last experience will be dropped.
        Defaults to False. Ignored if `experience_split_strategy` is used.
    :param experience_split_strategy: A function that implements a custom
        splitting strategy. The function must accept an experience and return an
        experience's iterator. Defaults to None, which means
        that the standard splitting strategy will be used (which creates
        experiences of size `experience_size`).
        A good starting to understand the mechanism is to look at the
        implementation of the standard splitting function
        :func:`fixed_size_experience_split`.
    :return: A lazy online stream with experiences of size `experience_size`.
    """

    if experience_split_strategy is None:
        # functools.partial is a more compact option
        # However, MyPy does not understand what a partial is -_-
        def default_online_split_wrapper(e, e_sz):
            return _default_online_split(shuffle, drop_last, access_task_boundaries, e, e_sz)

        split_strategy = default_online_split_wrapper
    else:
        split_strategy = experience_split_strategy

    def exps_iter():
        for exp in original_stream:
            for sub_exp in split_strategy(exp, experience_size):
                yield sub_exp

    stream_name: str = getattr(original_stream, "name", "train") + "_online"
    return CLStream(
        name=stream_name,
        exps_iter=exps_iter(),
        set_stream_info=True,
    )


def _fixed_size_split(
    online_benchmark: "OnlineCLScenario",  # TODO: Deprecated and unused. Remove.
    experience_size: int,
    access_task_boundaries: bool,
    shuffle: bool,
    s: Iterable[DatasetExperience[TCLDataset]],
) -> CLStream[DatasetExperience[TCLDataset]]:
    return split_online_stream(
        original_stream=s,
        experience_size=experience_size,
        access_task_boundaries=access_task_boundaries,
        shuffle=shuffle,
    )


class OnlineCLScenario(CLScenario):
    def __init__(
        self,
        original_streams: Iterable[CLStream[DatasetExperience[TCLDataset]]],
        experiences: Optional[
            Union[
                DatasetExperience[TCLDataset], Iterable[DatasetExperience[TCLDataset]]
            ]
        ] = None,
        experience_size: int = 10,
        stream_split_strategy: Literal["fixed_size_split"] = "fixed_size_split",
        access_task_boundaries: bool = False,
        shuffle: bool = True,
    ):
        """Creates an online scenario from an existing CL scenario

        :param original_streams: The streams from the original CL scenario.
        :param experiences: If None, the online stream will be created
            from the `train_stream` of the original CL scenario, otherwise it
            will create an online stream from the given sequence of experiences.
        :param experience_size: The size of each online experiences, as an int.
            Ignored if `custom_split_strategy` is used.
        :param experience_split_strategy: A function that implements a custom
            splitting strategy. The function must accept an experience and
            return an experience's iterator. Defaults to None, which means
            that the standard splitting strategy will be used (which creates
            experiences of size `experience_size`).
            A good starting to understand the mechanism is to look at the
            implementation of the standard splitting function
            :func:`fixed_size_experience_split`.
        : param access_task_boundaries: If True the attributes related to task
            boundaries such as `is_first_subexp` and `is_last_subexp` become
            accessible during training.
        :param shuffle: If True, experiences will be split by first shuffling
            instances in each experience. Defaults to True.
        """
        warnings.warn("Deprecated. Use `split_online_stream` or similar methods to split"
                     "single streams or experiences instead")
        
        if stream_split_strategy != "fixed_size_split":
            raise ValueError("Unknown experience split strategy")

        split_strat = partial(
            _fixed_size_split, self, experience_size, access_task_boundaries, shuffle
        )

        streams_dict = {s.name: s for s in original_streams}
        if "train" not in streams_dict:
            raise ValueError("Missing train stream for `original_streams`.")
        if experiences is None:
            online_train_stream = split_strat(streams_dict["train"])
        else:
            if not isinstance(experiences, Iterable):
                experiences = [experiences]
            online_train_stream = split_strat(experiences)

        streams: List[CLStream] = [online_train_stream]
        for s in original_streams:
            s_wrapped = wrap_stream(
                new_name="original_" + s.name, new_benchmark=self, wrapped_stream=s
            )

            streams.append(s_wrapped)

        super().__init__(streams=streams)


__all__ = [
    "OnlineCLExperience",
    "FixedSizeExperienceSplitter",
    "split_online_stream",
    "OnlineCLScenario",
]
