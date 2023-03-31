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
from typing import Callable, Generator, Generic, Iterable, List, Literal, Optional, TypeVar, Union

import torch

from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.scenarios.generic_scenario import (
    CLExperience,
    BaseCLStream,
    CLStreamWrapper,
    CLStream,
    DatasetExperienceProtocol,
    CLScenario,
    DatasetExperience,
    CLScenario,
    SequenceCLStream,
    SequenceStreamWrapper,
    SettableDatasetExperienceProtocol,
    SettableGenericExperienceProtocol,
    SizedCLStream,
    SizedCLStreamWrapper
)

from avalanche.benchmarks.scenarios.dataset_scenario import (
    DatasetScenario
)


TCLDataset = TypeVar("TCLDataset", bound="AvalancheDataset")  # Implementation, defined in utils
TCLScenario = TypeVar("TCLScenario", bound="CLScenario")
TDatasetScenario = TypeVar(
    "TDatasetScenario", bound="DatasetScenario"
)
TCLStream = TypeVar('TCLStream', bound='CLStream')
TBaseCLStream = TypeVar('TBaseCLStream', bound='BaseCLStream')
TDatasetExperience = TypeVar('TDatasetExperience', bound='DatasetExperienceProtocol')
TCLExperience = TypeVar('TCLExperience', bound='CLExperience')
TSettableGenericExperience = TypeVar('TSettableGenericExperience', bound='SettableGenericExperienceProtocol')  # Protocol, defined here
TSettableDatasetExperienceProtocol = TypeVar('TSettableDatasetExperienceProtocol', bound='SettableDatasetExperienceProtocol')


class OnlineCLExperience(Generic[TCLScenario, TCLStream, TDatasetExperience, TCLDataset],
                         DatasetExperience[TCLScenario, TCLStream, TCLDataset]):
    """Online CL (OCL) Experience.

    OCL experiences are created by splitting a larger experience. Therefore,
    they keep track of the original experience for logging purposes.
    """

    def __init__(
        self,
        current_experience: int,
        origin_stream: TCLStream,
        benchmark: TCLScenario,
        dataset: TCLDataset,
        origin_experience: TDatasetExperience,
        subexp_size: int = 1,
        is_first_subexp: bool = False,
        is_last_subexp: bool = False,
        sub_stream_length: Optional[int] = None,
        access_task_boundaries: bool = False,
    ):
        """Init.

        :param current_experience: experience identifier.
        :param origin_stream: origin stream.
        :param origin_experience: origin experience used to create self.
        :param is_first_subexp: whether self is the first in the sub-experiences
            stream.
        :param sub_stream_length: the sub-stream length.
        """
        super().__init__(
            current_experience=current_experience,
            origin_stream=origin_stream,
            benchmark=benchmark,
            dataset=dataset)
        self.access_task_boundaries = access_task_boundaries

        self.origin_experience: TDatasetExperience = origin_experience
        self.subexp_size: int = subexp_size
        self.is_first_subexp: bool = is_first_subexp
        self.is_last_subexp: bool = is_last_subexp
        self.sub_stream_length: Optional[int] = sub_stream_length

        self._as_attributes(
            'origin_experience',
            'subexp_size',
            'is_first_subexp',
            'is_last_subexp',
            'sub_stream_length',
            use_in_train=access_task_boundaries)

    @property
    def task_labels(self) -> List[int]:
        return self.origin_experience.task_labels


class OnlineClassificationExperience(OnlineCLExperience[TCLScenario, TCLStream, TDatasetExperience, TCLDataset]):
    """Online CL (OCL) Experience.

    OCL experiences are created by splitting a larger experience. Therefore,
    they keep track of the original experience for logging purposes.
    """

    def __init__(
        self,
        current_experience: int,
        origin_stream: TCLStream,
        benchmark: TCLScenario,
        dataset: TCLDataset,
        origin_experience: TDatasetExperience,
        classes_in_this_experience: List[int],
        subexp_size: int = 1,
        is_first_subexp: bool = False,
        is_last_subexp: bool = False,
        sub_stream_length: Optional[int] = None,
        access_task_boundaries: bool = False,
    ):
        """Init.

        :param current_experience: experience identifier.
        :param origin_stream: origin stream.
        :param origin_experience: origin experience used to create self.
        :param is_first_subexp: whether self is the first in the sub-experiences
            stream.
        :param sub_stream_length: the sub-stream length.
        """
        super().__init__(
            current_experience=current_experience,
            origin_stream=origin_stream,
            benchmark=benchmark,
            dataset=dataset,
            origin_experience=origin_experience,
            subexp_size=subexp_size,
            is_first_subexp=is_first_subexp,
            is_last_subexp=is_last_subexp,
            sub_stream_length=sub_stream_length,
            access_task_boundaries=access_task_boundaries)
        
        self.classes_in_this_experience: List[int] = classes_in_this_experience


TOnlineCLScenario = TypeVar('TOnlineCLScenario', bound='OnlineCLScenario')

# Alas, both PyLance and MyPy do not support Higher-Kinded TypeVars.
# https://github.com/python/typing/issues/548
# If they ever fix this, please change the following signature in this way:
# type for parameter experience: TDatasetExperience[TCLScenario, TCLStream, TCLDataset]
# type for generator (1st element): OnlineClassificationExperience[CLScenario, TCLStream, TDatasetExperience[TCLScenario, TCLStream, TCLDataset], TCLDataset]
def fixed_size_experience_split(
    experience: DatasetExperienceProtocol[TCLScenario, TCLStream, TCLDataset],
    experience_size: int,
    online_benchmark: TOnlineCLScenario,
    shuffle: bool = True,
    drop_last: bool = False,
    access_task_boundaries: bool = False
) -> Generator[OnlineClassificationExperience[TOnlineCLScenario, TCLStream, DatasetExperienceProtocol[TCLScenario, TCLStream, TCLDataset], TCLDataset], None, None]:
    """Returns a lazy stream generated by splitting an experience into smaller
    ones.

    Splits the experience in smaller experiences of size `experience_size`.

    :param experience: The experience to split.
    :param experience_size: The experience size (number of instances).
    :param shuffle: If True, instances will be shuffled before splitting.
    :param drop_last: If True, the last mini-experience will be dropped if
        not of size `experience_size`
    :return: The list of datasets that will be used to create the
        mini-experiences.
    """

    def gen():
        exp_dataset = experience.dataset
        exp_indices = list(range(len(exp_dataset)))
        exp_targets = torch.as_tensor(list(exp_dataset.targets), dtype=torch.long) # type: ignore

        if shuffle:
            exp_indices = torch.as_tensor(exp_indices)[
                torch.randperm(len(exp_indices))
            ].tolist()
        sub_stream_length = len(exp_indices) // experience_size
        if not drop_last and len(exp_indices) % experience_size > 0:
            sub_stream_length += 1

        init_idx = 0
        is_first = True
        is_last = False
        exp_idx = 0
        while init_idx < len(exp_indices):
            final_idx = init_idx + experience_size  # Exclusive
            if final_idx > len(exp_indices):
                if drop_last:
                    break

                final_idx = len(exp_indices)
                is_last = True

            sub_exp_subset = exp_dataset.subset(exp_indices[init_idx:final_idx])
            sub_exp_targets: torch.Tensor = exp_targets[exp_indices[init_idx:final_idx]].unique()
            exp = OnlineClassificationExperience(
                current_experience=exp_idx,
                origin_stream=experience.origin_stream,
                benchmark=online_benchmark,
                dataset=sub_exp_subset,
                origin_experience=experience,
                classes_in_this_experience=sub_exp_targets.tolist(),
                subexp_size=experience_size,
                is_first_subexp=is_first,
                is_last_subexp=is_last,
                sub_stream_length=sub_stream_length,
                access_task_boundaries=access_task_boundaries,
            )

            is_first = False
            yield exp
            init_idx = final_idx
            exp_idx += 1

    return gen()


def _default_online_split(online_benchmark, shuffle: bool, drop_last: bool, access_task_boundaries: bool, exp: DatasetExperienceProtocol[TCLScenario, TCLStream, TCLDataset], size: int):
    return fixed_size_experience_split(
        experience=exp,
        experience_size=size,
        online_benchmark=online_benchmark,
        shuffle=shuffle,
        drop_last=drop_last,
        access_task_boundaries=access_task_boundaries,
    )


def split_online_stream(
    original_stream: Iterable[DatasetExperienceProtocol[TCLScenario, TCLStream, TCLDataset]],
    experience_size: int,
    online_benchmark: TOnlineCLScenario,
    shuffle: bool = False,
    drop_last: bool = False,
    experience_split_strategy: Optional[Callable[
        [DatasetExperienceProtocol[TCLScenario, TCLStream, TCLDataset], int], Iterable[OnlineClassificationExperience[TOnlineCLScenario, TCLStream, DatasetExperienceProtocol[TCLScenario, TCLStream, TCLDataset], TCLDataset]]
    ]] = None,
    access_task_boundaries: bool = False
) -> CLStream[TOnlineCLScenario, OnlineClassificationExperience[TOnlineCLScenario, TCLStream, DatasetExperienceProtocol[TCLScenario, TCLStream, TCLDataset], TCLDataset]]:
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
        :func:`fixed_size_experience_split_strategy`.
    :return: A lazy online stream with experiences of size `experience_size`.
    """
    

    if experience_split_strategy is None:
        # functools.partial is a more compact option
        # However, MyPy does not understand what a partial is -_-
        def default_online_split_wrapper(e, e_sz):
            return _default_online_split(
                online_benchmark,
                shuffle,
                drop_last,
                access_task_boundaries,
                e,
                e_sz
            )

        split_strategy = default_online_split_wrapper
    else:
        split_strategy = experience_split_strategy

    def exps_iter():
        for exp in original_stream:
            for sub_exp in split_strategy(exp, experience_size):
                yield sub_exp

    stream_name: str = getattr(original_stream, "name", "train")
    return CLStream(
        name=stream_name, exps_iter=exps_iter(), set_stream_info=True, benchmark=online_benchmark
    )


def _fixed_size_split(
    online_benchmark: TOnlineCLScenario,
    experience_size: int,
    access_task_boundaries: bool,
    s: Iterable[DatasetExperienceProtocol[TCLScenario, TCLStream, TCLDataset]]) -> \
        CLStream[TOnlineCLScenario, OnlineClassificationExperience[TOnlineCLScenario, TCLStream, DatasetExperienceProtocol[TCLScenario, TCLStream, TCLDataset], TCLDataset]]:
    return split_online_stream(
        original_stream=s,
        experience_size=experience_size,
        online_benchmark=online_benchmark,
        access_task_boundaries=access_task_boundaries,
    )


class OnlineCLScenario(CLScenario[BaseCLStream]):
    def __init__(
        self,
        original_streams: Iterable[BaseCLStream[TDatasetScenario, TSettableDatasetExperienceProtocol]],
        experiences: Optional[Union[SettableDatasetExperienceProtocol[TDatasetScenario, TCLStream, TCLDataset], Iterable[SettableDatasetExperienceProtocol[TDatasetScenario, TCLStream, TCLDataset]]]] = None,
        experience_size: int = 10,
        stream_split_strategy: Literal['fixed_size_split'] = "fixed_size_split",
        access_task_boundaries: bool = False,
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
            :func:`fixed_size_experience_split_strategy`.
        : param access_task_boundaries: If True the attributes related to task
            boundaries such as `is_first_subexp` and `is_last_subexp` become
            accessible during training.
        """
        
        if stream_split_strategy != "fixed_size_split":
            raise ValueError("Unknown experience split strategy")

        split_strat = partial(_fixed_size_split, self, experience_size, access_task_boundaries)

        streams_dict = {s.name: s for s in original_streams}
        if "train" not in streams_dict:
            raise ValueError("Missing train stream for `original_streams`.")
        if experiences is None:
            online_train_stream = split_strat(streams_dict["train"])
        else:
            if not isinstance(experiences, Iterable):
                experiences = [experiences]
            online_train_stream = split_strat(experiences)

        streams: List[BaseCLStream] = [online_train_stream]
        for s in original_streams:
            s_wrapped: BaseCLStream
            if isinstance(s, SequenceCLStream):
                # Maintain indexing/slicing functionalities
                s_wrapped = SequenceStreamWrapper(
                    name="original_" + s.name,
                    benchmark=self,
                    wrapped_stream=s)
            elif isinstance(s, SizedCLStream):
                # Sized stream
                s_wrapped = SizedCLStreamWrapper(
                    name="original_" + s.name,
                    benchmark=self,
                    wrapped_stream=s)
            else:
                # Plain iter-based stream
                s_wrapped = CLStreamWrapper(
                    name="original_" + s.name,
                    benchmark=self,
                    wrapped_stream=s)

            streams.append(s_wrapped)

        super().__init__(
            streams=streams
        )


__all__ = [
    'OnlineCLExperience',
    'OnlineClassificationExperience',
    'fixed_size_experience_split',
    'split_online_stream',
    'OnlineCLScenario'
]
