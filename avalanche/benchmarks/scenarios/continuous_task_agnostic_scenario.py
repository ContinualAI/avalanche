################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 04-08-2023                                                             #
# Author(s): Antonio Carta, Hamed Hemati                                       #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

from functools import partial
from typing import (

    Iterable,
    List,
    Optional,
    Union,
)
from typing_extensions import Literal

import torch
from torch.distributions.categorical import Categorical
from torch.utils.data import Sampler

from avalanche.benchmarks.scenarios.benchmark_wrapper_utils import wrap_stream
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.scenarios.generic_scenario import (
    CLStream,
    CLScenario,
    DatasetExperience,
    CLScenario,
)
from avalanche.benchmarks.utils.classification_dataset import (
    classification_subset,
    concat_classification_datasets
)
from avalanche.benchmarks.scenarios.online_scenario import (
    OnlineClassificationExperience,
    TClassificationDataset, TCLDataset
)


class CyclicSampler(Sampler):
    """Samples elements from [0,..,len(dataset)-1] in a cyclic manner."""

    def __init__(self, n_samples, shuffle=True, rng=None):
        self.n_samples = n_samples
        self.rng = rng
        self.shuffle = shuffle
        self._reset_indices()

    def _reset_indices(self):
        self.indices = torch.arange(self.n_samples).tolist()
        if self.shuffle:
            self.indices = torch.randperm(self.n_samples,
                                          generator=self.rng).tolist()

    def __iter__(self):
        while True:
            for idx in self.indices:
                yield idx
            self._reset_indices()

    def __len__(self):
        return self.n_samples


def create_sub_exp_from_multi_exps(
    online_benchmark: "OnlineCLScenario",
    original_stream: Iterable[DatasetExperience[TClassificationDataset]],
    samplers: Iterable[CyclicSampler],
    exp_per_sample_list: Iterable[torch.Tensor],
    sub_exp_id: int,
    total_iters: int,
    is_first_sub_exp: bool = False,
    is_last_sub_exp: bool = False,
) -> DatasetExperience[TClassificationDataset]:
    """
    Creates a sub-experience from a list of experiences.

    :param online_benchmark: The online benchmark.
    :param original_stream: The original stream.
    :param samplers: A list of samplers, one for each experience in the
            original stream.
    :param exp_per_sample_list: A list of experience ids, one for each sample
            in the sub-experience.
    :param sub_exp_id: The id of the sub-experience.
    :param total_iters: The total number of iterations.
    :param is_first_sub_exp: Whether this is the first sub-experience.
    :param is_last_sub_exp: Whether this is the last sub-experience.

    :return: A sub-experience.
    """

    # Create sub-sets from each experience's dataset
    all_subsets = []
    n_samples_from_each_task = [0 for _ in range(len(samplers))]

    for exp_id in exp_per_sample_list.unique():
        n_samples = sum(exp_per_sample_list == exp_id.item()).item()
        n_samples_from_each_task[exp_id.item()] += n_samples
        rnd_indices = [next(samplers[exp_id]) for _ in range(n_samples)]
        subset_i = classification_subset(
            original_stream[exp_id.item()].dataset, rnd_indices)
        all_subsets.append(subset_i)

    # Concatenate all sub-sets
    sub_exp_subset = concat_classification_datasets(all_subsets)
    experience_size = len(sub_exp_subset)

    exp = OnlineClassificationExperience(
        current_experience=sub_exp_id,
        origin_stream=None,  # type: ignore
        benchmark=online_benchmark,
        dataset=sub_exp_subset,
        origin_experience=None,  # experience,
        classes_in_this_experience=list(set(sub_exp_subset.targets)),
        subexp_size=experience_size,
        is_first_subexp=is_first_sub_exp,
        is_last_subexp=is_last_sub_exp,
        sub_stream_length=total_iters,
        access_task_boundaries=False,
    )

    exp.n_samples_from_each_task = n_samples_from_each_task
    return exp


def _fixed_size_linear_decay_stream(
    online_benchmark: "OnlineCLScenario",
    experience_size: int,
    iters_per_virtual_epoch: int,
    beta: float,
    shuffle: bool,
    original_stream: Iterable[DatasetExperience[TClassificationDataset]],
) -> CLStream[DatasetExperience[TClassificationDataset]]:
    """Creates a stream of sub-experiences from a list of experiences.

    :param online_benchmark: The online benchmark.
    :param experience_size: The size of each sub-experience.
    :param iters_per_virtual_epoch: The number of iterations per virtual epoch.
    :param beta: The beta parameter for the linear decay function.
    :param shuffle: Whether to shuffle the sub-experiences.
    :param original_stream: The original stream.

    :return: A stream of sub-experiences.

    """
    def _get_linear_line(start, end, direction="up"):
        if direction == "up":
            return torch.FloatTensor([(i - start) / (end - start)
                                      for i in range(start, end)])
        return torch.FloatTensor([1 - ((i - start) / (end - start))
                                  for i in range(start, end)])

    def _create_task_probs(iters, tasks, task_id, beta=3):

        if beta <= 1:
            peak_start = int((task_id / tasks) * iters)
            peak_end = int(((task_id + 1) / tasks) * iters)
            start = peak_start
            end = peak_end
        else:
            start = max(int(((beta * task_id - 1) * iters) / (beta * tasks)), 0)
            peak_start = int(((beta * task_id + 1) * iters) / (beta * tasks))
            peak_end = int(((beta * task_id + (beta - 1))
                            * iters) / (beta * tasks))
            end = min(int(((beta * task_id + (beta + 1))
                           * iters) / (beta * tasks)), iters)

        probs = torch.zeros(iters, dtype=torch.float)
        if task_id == 0:
            probs[start:peak_start].add_(1)
        else:
            probs[start:peak_start] = _get_linear_line(
                start, peak_start, direction="up")
        probs[peak_start:peak_end].add_(1)
        if task_id == tasks - 1:
            probs[peak_end:end].add_(1)
        else:
            probs[peak_end:end] = _get_linear_line(peak_end, end,
                                                   direction="down")
        return probs

    # Total number of iterations
    total_iters = len(original_stream) * iters_per_virtual_epoch

    # Probabilities over all iterations (sub-experiences)
    n_experiences = len(original_stream)
    tasks_probs_over_iterations = [
        _create_task_probs(total_iters, n_experiences, exp_id, beta=beta)
        for exp_id in range(n_experiences)]

    # Normalize probabilities
    normalize_probs = torch.zeros_like(tasks_probs_over_iterations[0])
    for probs in tasks_probs_over_iterations:
        normalize_probs.add_(probs)
    for probs in tasks_probs_over_iterations:
        probs.div_(normalize_probs)
    tasks_probs_over_iterations = torch.cat(
        tasks_probs_over_iterations
    ).view(-1, tasks_probs_over_iterations[0].shape[0])
    tasks_probs_over_iterations_lst = []
    for col in range(tasks_probs_over_iterations.shape[1]):
        tasks_probs_over_iterations_lst.append(
            tasks_probs_over_iterations[:, col])
    tasks_probs_over_iterations = tasks_probs_over_iterations_lst

    # Random cylic samplers over the datasets of all experiences in the stream
    samplers = [iter(CyclicSampler(len(exp.dataset)))
                for exp in original_stream]

    # The main iterator for the stream
    def exps_iter():
        for sub_exp_id in range(total_iters):
            is_first_sub_exp = is_last_sub_exp = False
            if sub_exp_id == 0:
                is_first_sub_exp = True
            if sub_exp_id == total_iters - 1:
                is_last_sub_exp = True

            n_samples = torch.Size([experience_size])
            exp_per_sample_list = Categorical(
                probs=tasks_probs_over_iterations[sub_exp_id]
            ).sample(n_samples)

            yield create_sub_exp_from_multi_exps(online_benchmark,
                                                 original_stream,
                                                 samplers,
                                                 exp_per_sample_list,
                                                 sub_exp_id,
                                                 total_iters,
                                                 is_first_sub_exp,
                                                 is_last_sub_exp)

    stream_name: str = getattr(original_stream, "name", "train")
    return CLStream(
        name=stream_name,
        exps_iter=exps_iter(),
        set_stream_info=True,
        benchmark=online_benchmark,
    )


class ContinuousTaskAgnosticScenario(
        CLScenario[CLStream[DatasetExperience[TCLDataset]]]):
    def __init__(
        self,
        original_streams: Iterable[CLStream[DatasetExperience[TCLDataset]]],
        experiences: Optional[
            Union[
                DatasetExperience[TCLDataset],
                Iterable[DatasetExperience[TCLDataset]]
            ]
        ] = None,
        experience_size: int = 10,
        iters_per_virtual_epoch: int = 300,
        overlap_factor: float = 4,
        stream_split_strategy: Literal["fixed_size_split"] = "fixed_size_split",
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
        :param shuffle: If True, experiences will be split by first shuffling
            instances in each experience. Defaults to True.
        """

        if stream_split_strategy != "linear_decay":
            split_strat = partial(
                _fixed_size_linear_decay_stream, self, experience_size,
                iters_per_virtual_epoch, overlap_factor, shuffle

            )
        else:
            raise ValueError("Unknown experience split strategy")

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
                new_name="original_" + s.name, new_benchmark=self,
                wrapped_stream=s
            )

            streams.append(s_wrapped)

        super().__init__(streams=streams)


__all__ = [
    "ContinuousTaskAgnosticScenario",
]
