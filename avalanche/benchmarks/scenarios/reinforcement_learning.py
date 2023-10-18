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
    CLExperience,
    CLScenario,
    EagerCLStream,
)
from typing import Any, Callable, List, Optional, TypeVar, Union, Dict
import numpy as np

from avalanche.benchmarks.scenarios.generic_scenario import CLStream

try:
    from gym import Env, Wrapper
except ImportError:
    raise ModuleNotFoundError(
        "gym not found, if you want to use "
        "RL please install avalanche with "
        "the rl dependencies: "
        "pip install avalanche-lib[rl]"
    )


TCLStream = TypeVar("TCLStream", bound="CLStream")
TRLScenario = TypeVar("TRLScenario", bound="RLScenario")
TRLExperience = TypeVar("TRLExperience", bound="RLExperience")


class RLExperience(CLExperience):
    """Experience for Continual Reinforcement Learning purposes.

    The experience provides access to a `gym.Env` environment.
    Such environment can also be created lazily by
    providing a function.
    """

    def __init__(
        self,
        current_experience: int,
        origin_stream: CLStream,
        env: Union[Env, Callable[[], Env]],
        n_envs: int = 1,
        task_label: Optional[int] = None,
    ):
        super().__init__(current_experience, origin_stream)
        self.env = env
        self.n_envs = n_envs
        # task label to be (optionally) used for training purposes
        self.task_label = task_label

        self._as_attributes("task_label", use_in_train=True, use_in_eval=True)

    @property
    def environment(self) -> Env:
        # support dynamic/lazy environment creation
        if not isinstance(self.env, Env):
            return self.env()
        return self.env


class RLScenario(CLScenario[CLStream[TRLExperience]]):
    """Scenario for Continual Reinforcement Learning (CRL) purposes.

    It allows an agent to learn from a stream of environments, generating state
    transitions from the active interaction within each experience.
    This abstraction enables the representation of a continuous stream of tasks
    as requested by CRL settings.

    Reference: Lucchesi, N., Carta, A., Lomonaco, V., & Bacciu, D. (2022).
    Avalanche RL: a Continual Reinforcement Learning Library
    https://arxiv.org/abs/2202.13657
    """

    def __init__(
        self,
        envs: List[Env],
        n_parallel_envs: Union[int, List[int]],
        eval_envs: Optional[Union[List[Env], List[Callable[[], Env]]]] = None,
        wrappers_generators: Optional[Dict[str, List[Wrapper]]] = None,
        task_labels: bool = True,
        shuffle: bool = False,
    ):
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
                Passing None or `[]` will result in no evaluation.
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
        if isinstance(n_parallel_envs, int):
            n_parallel_envs = [n_parallel_envs] * n_experiences
        assert len(n_parallel_envs) == len(envs)
        # this is so that we can infer the task labels
        assert all(
            [isinstance(e, Env) for e in envs]
        ), "Lazy instantation of\
            training environments is not supported"
        assert all(
            [n > 0 for n in n_parallel_envs]
        ), "Number of parallel environments\
                        must be a positive integer"
        tr_envs = envs
        eval_envs = eval_envs or []
        self.n_envs = n_parallel_envs

        def get_unique_task_labels(env_list):
            # assign task label by checking whether the same instance of env is
            # provided multiple times, using object hash as key
            tlabels: List[int] = []
            env_occ: Dict[Any, int] = {}
            j = 0
            for e in env_list:
                if e in env_occ:
                    tlabels.append(env_occ[e])
                else:
                    tlabels.append(j)
                    env_occ[e] = j
                    j += 1
            return tlabels

        # accounts for shallow copies of envs to have multiple
        # experiences from the same task
        tr_task_labels = get_unique_task_labels(tr_envs)
        eval_task_labels = get_unique_task_labels(eval_envs)

        self._wrappers_generators = wrappers_generators

        if shuffle:
            perm = np.random.permutation(len(tr_envs))
            tr_envs = [tr_envs[i] for i in perm]
            tr_task_labels = [tr_task_labels[i] for i in perm]

        # decide whether to provide task labels to experiences
        tr_task_labels = tr_task_labels if task_labels else [None] * len(tr_envs)

        tr_exps: List[TRLExperience] = [
            RLExperience(
                current_experience=i,
                origin_stream=None,  # type: ignore
                env=tr_envs[i],
                n_envs=n_parallel_envs[i],
                task_label=tr_task_labels[i],
            )
            for i in range(len(tr_envs))
        ]
        tstream: EagerCLStream[TRLExperience] = EagerCLStream(
            name="train", exps=tr_exps, benchmark=self
        )
        # we're only supporting single process envs in evaluation atm
        print("EVAL ", eval_task_labels)
        eval_exps: List[TRLExperience] = [
            RLExperience(
                current_experience=i,
                origin_stream=None,  # type: ignore
                env=e,
                n_envs=1,
                task_label=l,
            )
            for i, (e, l) in enumerate(zip(eval_envs, eval_task_labels))
        ]
        estream: EagerCLStream[TRLExperience] = EagerCLStream(
            name="eval", exps=eval_exps, benchmark=self
        )

        super().__init__([tstream, estream])


__all__ = ["RLExperience", "RLScenario"]
