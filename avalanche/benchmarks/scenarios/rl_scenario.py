################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-04-2022                                                             #
# Author(s): Nicolo' Lucchesi Antonio Carta                                    #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""Reinforcement Learning scenario definitions."""
from avalanche.benchmarks.scenarios import (
    CLExperience, ExperienceAttribute,
    CLScenario, EagerCLStream)
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
    """Experience for Continual Reinforcement Learning purposes.

    The experience provides access to a `gym.Env` environment.
    Such environment can also be created lazily by 
    providing a function. 
    """

    def __init__(self, env: Union[Env, Callable[[], Env]], n_envs: int = 1, 
                 task_label: int = None, current_experience: int = None, 
                 origin_stream=None):
        super().__init__(current_experience, origin_stream)
        self.env = env
        self.n_envs = n_envs
        # task label to be (optionally) used for training purposes
        self.task_label = ExperienceAttribute(
            task_label, use_in_train=True, use_in_eval=True)

    @property
    def environment(self) -> Env:
        # support dynamic/lazy environment creation
        if not isinstance(self.env, Env):
            return self.env()
        return self.env


class RLScenario(CLScenario):
    """Scenario for Continual Reinforcement Learning (CRL) purposes.

    It allows an agent to learn from a stream of environments, generating state
    transitions from the active interaction within each experience.
    This abstraction enables the representation of a continuous stream of tasks
    as requested by CRL settings.

    Reference: Lucchesi, N., Carta, A., Lomonaco, V., & Bacciu, D. (2022).
    Avalanche RL: a Continual Reinforcement Learning Library
    https://arxiv.org/abs/2202.13657
    """

    def __init__(self, envs: List[Env],
                 n_parallel_envs: Union[int, List[int]],
                 eval_envs: Union[List[Env], List[Callable[[], Env]]],
                 wrappers_generators: Dict[str, List[Wrapper]] = None,
                 task_labels: bool = True,
                 shuffle: bool = False):
        """Init.

        Args:
            :param envs: list of gym environments to be used for training the 
                agent.Each environment will be wrapped within a RLExperience.
            :param n_parallel_envs: number of parallel agent-environment 
                interactions to run for each experience. If an int is provided, 
                the same degree of parallelism will be used for every env. 
            :param eval_envs: list of gym environments 
                to be used for evaluating the agent. Each environment will
                be wrapped within a RLExperience.
            :param wrappers_generators: dict mapping env ids to a list of 
                `gym.Wrapper` generator. Wrappers represent behavior 
                added as post-processing steps (e.g. reward scaling).
            :param task_labels: whether to add task labels to RLExperience. 
                A task label is assigned to each different environment, 
                in the order they're provided in `envs`.
            :param shuffle: whether to randomly shuffle `envs`. 
                Defaults to False.
        """

        n_experiences = len(envs)
        if type(n_parallel_envs) is int:
            n_parallel_envs = [n_parallel_envs] * n_experiences
        assert len(n_parallel_envs) == len(envs)
        # this is so that we can infer the task labels
        assert all([isinstance(e, Env) for e in envs]), "Lazy instantation of\
            training environments is not supported"
        assert all([n > 0 for n in n_parallel_envs]
                   ), "Number of parallel environments\
                        must be a positive integer"
        tr_envs = envs
        eval_envs = eval_envs or []
        self._num_original_envs = len(tr_envs)
        self.n_envs = n_parallel_envs
        # this can contain shallow copies of envs to have multiple
        # experiences from the same task
        tr_task_labels = []
        env_occ = {}
        j = 0
        # assign task label by checking whether the same instance of env is 
        # provided multiple times (shallow copy only)
        for e in envs:
            if e in env_occ:
                tr_task_labels.append(env_occ[e])
            else:
                tr_task_labels.append(j)
                env_occ[e] = j
                j += 1

        # eval_task_labels = list(range(len(eval_envs)))
        self._wrappers_generators = wrappers_generators

        if shuffle:
            perm = np.random.permutation(tr_envs)
            tr_envs = [tr_envs[i] for i in perm]
            tr_task_labels = [tr_task_labels[i] for i in perm]

        # decide whether to provide task labels to experiences
        tr_task_labels = tr_task_labels if task_labels else [
            None] * len(tr_envs)

        tr_exps = [RLExperience(tr_envs[i], n_parallel_envs[i], 
                                tr_task_labels[i]) for i in range(len(tr_envs))]
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
