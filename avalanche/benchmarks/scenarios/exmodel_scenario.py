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
from typing import List, TypeVar

from torch.nn import Module

from . import CLScenario, CLExperience, CLStream

TExModelExperience = TypeVar("TExModelExperience", bound="ExModelExperience")


class ExModelExperience(CLExperience):
    """Ex-Model CL Experience.

    The experience only provides the expert model.
    The original data is not available.
    """

    def __init__(
        self,
        expert_model: Module,
        current_experience: int,
        origin_stream: CLStream,
        classes_in_this_experience=None,
    ):
        super().__init__(
            current_experience=current_experience, origin_stream=origin_stream
        )
        self.expert_model: Module = expert_model
        self.classes_in_this_experience = classes_in_this_experience


class ExModelCLScenario(CLScenario[CLStream[TExModelExperience]]):
    """Ex-Model CL Scenario.

    Ex-Model Continual Learning (ExML) is a continual learning scenario where
    the CL agent learns from a stream of pretrained models instead of raw data.
    These approach allows to integrate knowledge from different CL agents or
    pretrained models.

    Reference: Carta, A., Cossu, A., Lomonaco, V., & Bacciu, D. (2021).
    Ex-Model: Continual Learning from a Stream of Trained Models.
    arXiv preprint arXiv:2112.06511.
    https://arxiv.org/abs/2112.06511
    """

    def __init__(self, original_benchmark: CLScenario, expert_models: List[Module]):
        """Init.

        :param original_benchmark: a reference to the original benchmark
            containing the stream of experiences used to train the experts.
        :param expert_models: pretrained models. The model in position i must
            be trained on the i-th experience of the train stream of
            `original_benchmark`.
        """
        expert_models_l: List[TExModelExperience] = []
        for i, (m, e) in enumerate(
            zip(expert_models, original_benchmark.train_stream)
        ):  # type: ignore
            cine = e.classes_in_this_experience
            expert_models_l.append(
                ExModelExperience(
                    expert_model=m,
                    current_experience=i,
                    origin_stream=None,  # type: ignore
                    classes_in_this_experience=cine,
                )
            )

        expert_stream: CLStream[TExModelExperience] = CLStream(
            name="expert_models", exps_iter=expert_models_l, benchmark=self
        )
        streams: List[CLStream[TExModelExperience]] = [expert_stream]

        self.original_benchmark = original_benchmark
        # for s in original_benchmark.streams.values():
        #     s = copy(s)
        #     s.name = 'original_' + s.name
        #     streams.append(s)
        super().__init__(streams)


__all__ = ["ExModelExperience", "ExModelCLScenario"]
