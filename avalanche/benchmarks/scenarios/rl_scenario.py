################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-04-2022                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""Reinforcement Learning scenario definitions."""
from avalanche.benchmarks.scenarios import CLExperience, ExperienceAttribute, CLScenario, EagerCLStream
from typing import Callable, List, Union, Dict
import numpy as np
import torch
import random

try:
    from gym import Env, Wrapper
except ImportError:
    # empty classes to make sure everything below works without changes
    class Env:
        pass
    class Wrapper:
        pass


class RLExperience(CLExperience):

    def __init__(self, env: Union[Env, Callable[[], Env]], n_envs: int = 1, task_label: int = None, current_experience: int = None, origin_stream=None):
        # current experience and origin stream are set when iterating a CLStream by default
        super().__init__(current_experience, origin_stream)
        self.env = env
        self.n_envs = n_envs
        # task label to be (optionally) used for training purposes
        self.task_label = ExperienceAttribute(task_label, use_in_train=True)

    @property
    def environment(self) -> Env:
        # supports dynamic creation of environment, useful to instantiate envs for evaluation
        if not isinstance(self.env, Env):
            return self.env()
        return self.env


class RLScenario(CLScenario):

    def __init__(self, envs: List[Env],
                 n_experiences: int,
                 n_parallel_envs: Union[int, List[int]],
                 eval_envs: Union[List[Env], List[Callable[[], Env]]],
                 wrappers_generators: Dict[str, List[Wrapper]] = None,
                 task_labels: bool = True,
                 shuffle: bool = False, 
                 seed: int = None):

        assert n_experiences > 0, "Number of experiences must be a positive integer"
        if type(n_parallel_envs) is int:
            n_parallel_envs = [n_parallel_envs] * n_experiences
        assert len(n_parallel_envs) == len(envs)
        assert all([n > 0 for n in n_parallel_envs]
                   ), "Number of parallel environments must be a positive integer"
        tr_envs = envs
        eval_envs = eval_envs or []
        self._num_original_envs = len(tr_envs)
        self.n_envs = n_parallel_envs
        # this shouldn't contain duplicate envs, but it's difficult to ensure if scenario isn't created through a benchmark generator
        tr_task_labels = list(range(len(envs)))

        # eval_task_labels = list(range(len(eval_envs)))
        self._wrappers_generators = wrappers_generators

        if n_experiences < len(tr_envs):
            tr_envs = tr_envs[:n_experiences]
            tr_task_labels = tr_task_labels[:n_experiences]
        elif n_experiences > len(tr_envs):
            # cycle through envs sequentially, referencing same object to create a longer stream
            for i in range(n_experiences - len(tr_envs)):
                tr_envs.append(tr_envs[i % len(tr_envs)])
                tr_task_labels.append(
                    tr_task_labels[i % len(tr_task_labels)])

        # move to template/strategy?
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        if shuffle:
            perm = np.random.permutation(tr_envs)
            tr_envs = [tr_envs[i] for i in perm]
            tr_task_labels = [tr_task_labels[i] for i in perm]

        # decide whether to provide task labels to experiences
        tr_task_labels = tr_task_labels if task_labels else [
            None] * len(tr_envs)

        tr_exps = [RLExperience(
            tr_envs[i], n_parallel_envs[i], tr_task_labels[i]) for i in range(len(tr_envs))]
        tstream = EagerCLStream("train", tr_exps)
        # we're only supporting single process envs in evaluation atm
        eval_exps = [RLExperience(e, 1) for e in eval_envs]
        estream = EagerCLStream("eval", eval_exps)

        super().__init__([tstream, estream])

    @property
    def train_stream(self):
        return self.streams["train_stream"]

    @property
    def eval_stream(self):
        return self.streams["eval_stream"]


__all__ = ["RLExperience", "RLScenario"]
