from collections import defaultdict
from typing import Union, Optional, Sequence, List
import enum
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from gym import Env
from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche_rl.benchmarks.rl_benchmark import RLExperience, RLScenario
from avalanche_rl.training.plugins.strategy_plugin import RLStrategyPlugin
from avalanche_rl.training.strategies.env_wrappers import *
from avalanche_rl.training import default_rl_logger
from avalanche_rl.training.strategies.vectorized_env import VectorizedEnvironment
from .buffers import Rollout, Step
from itertools import count
from avalanche.training.plugins.clock import Clock


class TimestepUnit(enum.IntEnum):
    STEPS = 0
    EPISODES = 1


@dataclass
class Timestep:
    value: int
    unit: TimestepUnit = TimestepUnit.STEPS


class RLBaseStrategy(BaseStrategy):
    def __init__(
            self, model: nn.Module, optimizer: Optimizer,
            per_experience_steps: Union[int, Timestep, List[Timestep]],
            criterion=nn.MSELoss(),
            rollouts_per_step: int = 1, max_steps_per_rollout: int = -1,
            updates_per_step: int = 1, device='cpu', max_grad_norm=None,
            plugins: Sequence[RLStrategyPlugin] = [],
            discount_factor: float = 0.99, evaluator=default_rl_logger,
            eval_every=-1, eval_episodes: int = 1):
        """
            RLBaseStrategy specializes BaseTemplate to handle Reinforcement Learning tasks
            in the continual learning setting and should be subclassed by all RL strategies.
            It implements a general training loop building on top of the one provided by BaseTemplate,
            which should be valid for both on and off-policy algorithms, in which at each training
            experience iteration a 'rollout phase' is followed by an 'update'.
            All continual learning scenarios described in the super-class are still supported.
            Additional callbacks have been provided before and after the rollout phase and can be
            used to further enhance the training loop with additional behavior.

            **Training loop**
            The training loop is organized as follows::
                train
                    train_exp  # for each experience
                        make_train_env
                        make_optimizer
                        # for `steps_per_experience` times
                        rollout
                        # for `updates_per_step` times
                            update, to be defined by sub-strategy (e.g. DQN/A2C..)
                            backward
                            optimizer step

            **Evaluation loop**
            The evaluation loop is organized as follows::
                eval
                    eval_exp  # for each experience
                        make_eval_env
                        evaluate_experience  # for `eval_episodes` episodes
                            # get action `a ~ model.get_action(..)`
                            # step environment simulator `env.step(a)`
                            # record reward and episode length

        Args:
            :param model: PyTorch model.
            :param optimizer: PyTorch optimizer.
            :param per_experience_steps (Union[int, Timestep, List[Timestep]]): Number of steps to perform for each experience.
                    Steps can either be expressed in terms of number of episodes or actual time steps.
                    If a single integer is passed, each experience will be run for the same amount of time steps.
            :param criterion: loss function. Defaults to nn.MSELoss().
            :param rollouts_per_step (int, optional): Number of rollouts to be performed at each step. A rollout is
                     to be intended as a sequence of environment interactions which terminate upon reaching the
                     terminal episode signal (`done=True` in Gym interface). Defaults to 1.
            :param max_steps_per_rollout (int, optional): Number of rollouts steps to be performed at each step. Note
                     that we don't wait for the episode to be done but we advance for a fixed amount of steps, resetting
                     the environment on terminal signal. If `rollouts_per_step` is also > 0 these parameters are combined,
                     unrolling `rollouts_per_step` rollouts of at most `max_steps_per_rollout` length. Defaults to -1.
            :param updates_per_step (int, optional): Number of update steps to perform at each experience step. Defaults to 1.
            :param device: PyTorch device where the model will be allocated.
            :param plugins: (optional) list of StrategyPlugins.
            :param discount_factor (float, optional): Also known as \gamma in RL literature, discount factor used
                    in weighting rewards importance. Defaults to 0.99.
            :param evaluator: (optional) instance of EvaluationPlugin for logging
                and metric computations. None to remove logging.
            :param eval_every: the frequency of the calls to `eval` inside the
                training loop.
                    if -1: no evaluation during training (Default).
                    if  0: calls `eval` after the final epoch of each training
                        experience.
                    if >0: calls `eval` every `eval_every` epochs and at the end
                        of all the epochs for a single experience.
            :param eval_episodes (int, optional): Number of episodes to run during evaluation. Defaults to 1.
        """

        super().__init__(model, optimizer, criterion=criterion, device=device,
                         plugins=plugins, eval_every=eval_every, evaluator=evaluator)

        assert rollouts_per_step > 0 or max_steps_per_rollout > 0, "Must specify at least one terminal condition for rollouts!"
        assert updates_per_step > 0, "Number of updates per step must be positve"

        # if a single number is passed, assume it's steps
        if isinstance(per_experience_steps, (int, float)):
            per_experience_steps = [Timestep(int(per_experience_steps))]
        elif isinstance(per_experience_steps, Timestep):
            per_experience_steps = [per_experience_steps]

        self.per_experience_steps: List[Timestep] = per_experience_steps
        self.rollouts_per_step = rollouts_per_step
        self.max_steps_per_rollout = max_steps_per_rollout
        self.updates_per_step = updates_per_step
        self.total_steps = 0
        self._obs: torch.Tensor = None
        self.gamma = discount_factor
        # defined by the experience
        self.n_envs: int = None
        self.eval_episodes = eval_episodes
        self.max_grad_norm = max_grad_norm
        # TODO: support Clock?
        for i in range(len(self.plugins)):
            if isinstance(self.plugins[i], Clock):
                self.plugins.pop(i)
                break
        self.plugins: Sequence[RLStrategyPlugin] = self.plugins

    @property
    def current_experience_steps(self) -> Timestep:
        """
            Return number of steps to perform for current experience (only valid during training).
        """
        return self.per_experience_steps[self.experience.current_experience %
                                         len(self.per_experience_steps)]

    # Additional callback added by RLBaseStrategy
    def before_rollout(self, **kwargs):
        for p in self.plugins:
            p.before_rollout(self, **kwargs)

    def after_rollout(self, **kwargs):
        for p in self.plugins:
            p.after_rollout(self, **kwargs)

    def sample_rollout_action(self, observations: torch.Tensor) -> np.ndarray:
        """Implements the action sampling a~Pi(s) where Pi is the parameterized
           function we're trying to learn.
           Output of this function should be a numpy array to comply with
           `VectorizedEnvironment` interface.

        Args:
            observations (torch.Tensor): batch of observations/state at current time t.

        Returns:
            np.ndarray: batch of actions to perform during rollout of shape `n_envs` x A.
        """
        raise NotImplementedError(
            "`sample_rollout_action` must be implemented by every RL strategy")

    def rollout(
            self, env: Env, n_rollouts: int, max_steps: int = -1) -> List[Rollout]:
        """
        Gather experience from Environment leveraging VectorizedEnvironment for parallel interaction and
        handling auto reset behavior.
        Args:
            env (Env): [description]
            n_rollouts (int): [description]
            max_steps (int, optional): [description]. Defaults to -1.

        Returns:
            Tuple[List[Rollout], int]: A list of rollouts, one per episode if `n_rollouts` is defined, where an episode
            is considered over as soon as one of the actors returns done=True.
            Otherwise a single rollout will be returned with the number of steps defined by `max_steps`.
            A combination of both `n_rollouts` and `max_steps` will result in returning `n_rollouts` episodes of
            length at most `max_steps`.
            The number of steps performed is also always returned along with the rollouts.
        """
        # gather experience from env
        rollout_counter = 0
        rollouts = []
        step_experiences = []

        # to compute timestep differences more efficiently
        ep_len_sum = [sum(self.ep_lengths[k]) for k in range(self.n_envs)]

        # reset environment on first run
        if self._obs is None:
            self._obs = env.reset()

        for t in count(start=1):
            # sample action(s) from policy moving observation to device; actions of shape `n_envs`xA
            action = self.sample_rollout_action(
                self._obs.to(self.device))

            # observations returned are one for each parallel environment
            next_obs, rewards, dones, info = env.step(action)

            step_experiences.append(
                Step(self._obs, action, dones, rewards, next_obs))
            self.rollout_steps += 1
            # keep track of all rewards for parallel environments
            self.rewards['curr_returns'] += rewards.reshape(-1,)

            self._obs = next_obs

            dones_idx = dones.reshape(-1, 1).nonzero()[0]
            for env_done in dones_idx:
                self.ep_lengths[env_done].append(
                    self.rollout_steps-ep_len_sum[env_done])
                ep_len_sum[env_done] += self.ep_lengths[env_done][-1]
                # record done episode returns
                self.rewards['past_returns'].append(
                    self.rewards['curr_returns'][env_done])
                self.rewards['curr_returns'][env_done] = 0.

            # Vectorized env auto resets on done by default, check this flag to count episodes
            if n_rollouts > 0:
                # check if any actor has finished an episode or `max_steps` reached
                if dones.any() or (max_steps > 0 and len(step_experiences) >= max_steps):
                    rollouts.append(
                        Rollout(step_experiences, n_envs=self.n_envs))
                    step_experiences = []
                    rollout_counter += 1
                    # TODO: if not auto_reset: self._obs = env.reset

            # check terminal condition(s)
            if n_rollouts > 0 and rollout_counter >= n_rollouts:
                break

            if max_steps > 0 and n_rollouts <= 0 and t >= max_steps:
                rollouts.append(Rollout(step_experiences, n_envs=self.n_envs))
                break

        return rollouts

    def update(self, rollouts: List[Rollout]):
        raise NotImplementedError(
            "`update` must be implemented by every RL strategy")

    def make_train_env(self, **kwargs):
        # maintain vectorized env interface without parallel overhead if `n_envs` is 1
        if self.n_envs == 1:
            env = VectorizedEnvWrapper(self.environment, auto_reset=True)
        else:
            import multiprocessing
            cpus = min(self.n_envs, multiprocessing.cpu_count())
            env = VectorizedEnvironment(
                self.environment, self.n_envs, auto_reset=True,
                wrappers_generators=self.experience.scenario.
                _wrappers_generators[self.environment.spec.id],
                ray_kwargs={'num_cpus': cpus})
        # NOTE: `info['terminal_observation']`` is NOT converted to tensor
        return Array2Tensor(env)

    def make_eval_env(self, **kwargs):
        # during evaluation we do not use a vectorized environment
        return Array2Tensor(self.environment)

    def train(self, experiences: Union[RLExperience, Sequence[RLExperience]],
              eval_streams: Optional[Sequence[Union[RLExperience,
                                                    Sequence[
                                                        RLExperience]]]] = None,
              **kwargs):
        self.is_training = True
        self.model.train()
        self.model.to(self.device)

        # Normalize training and eval data.
        if isinstance(experiences, RLExperience):
            experiences = [experiences]
        if eval_streams is None:
            eval_streams = [experiences]
        # for i, exp in enumerate(eval_streams):
            # if isinstance(exp, RLExperience):
            # eval_streams[i] = [exp]

        self._before_training(**kwargs)
        for self.experience in experiences:
            # make sure env is reset on new experience
            self._obs = None
            self.train_exp(self.experience, eval_streams, **kwargs)
        self._after_training(**kwargs)

        self.is_training = False
        res = self.evaluator.get_last_metrics()
        return res

    def train_exp(self, experience: RLExperience, eval_streams, **kwargs):
        self.environment = experience.environment
        self.n_envs = experience.n_envs
        # TODO:  keep track in default evaluator
        self.rollout_steps = 0
        # one per (parallel) environment
        self.ep_lengths: Dict[int, List[float]] = defaultdict(lambda: list([0]))
        # curr episode returns (per actor) - previous episodes returns
        self.rewards = {'curr_returns': np.zeros(
                            (self.n_envs,),
                            dtype=np.float32),
                        'past_returns': []}

        # Environment creation
        self.environment = self.make_train_env(**kwargs)

        # Model Adaptation (e.g. freeze/add new units)
        # self.model_adaptation()
        self.make_optimizer()

        self._before_training_exp(**kwargs)

        # either run N episodes or steps depending on specified `per_experience_steps`
        for self.timestep in range(self.current_experience_steps.value):
            self.before_rollout(**kwargs)
            self.rollouts = self.rollout(
                env=self.environment, n_rollouts=self.rollouts_per_step,
                max_steps=self.max_steps_per_rollout)
            self.after_rollout(**kwargs)

            for self.update_step in range(self.updates_per_step):
                # update must instatiate `self.loss`
                self.update(self.rollouts)

                # Backward
                self.optimizer.zero_grad()
                self._before_backward(**kwargs)
                self.loss.backward()
                self._after_backward(**kwargs)

                # Gradient norm clipping
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm)

                # Optimization step
                self._before_update(**kwargs)
                self.optimizer.step()
                self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
            # periodic evaluation
            self._periodic_eval(eval_streams, do_final=False)

        self.total_steps += self.rollout_steps
        self.environment.close()

        # Final evaluation
        self._periodic_eval(eval_streams, do_final=(
            self.timestep % self.eval_every != 0))
        self._after_training_exp(**kwargs)

    def _periodic_eval(self, eval_streams, do_final):
        """ Periodic eval controlled by `self.eval_every`. """
        # Since we are switching from train to eval model inside the training
        # loop, we need to save the training state, and restore it after the
        # eval is done.
        _prev_state = (
            self.timestep,
            self.experience,
            self.environment,
            self.n_envs,
            self.is_training)

        if (self.eval_every >= 0 and do_final) or \
           (self.eval_every > 0 and self.timestep % self.eval_every == 0):
            for exp in eval_streams:
                self.eval(exp)

        # restore train-state variables and training mode.
        self.timestep, self.experience, self.environment = _prev_state[:3]
        self.n_envs, self.is_training = _prev_state[3:]
        self.model.train()

    @torch.no_grad()
    def eval(
            self, exp_list: Union[RLExperience, Sequence[RLExperience]],
            **kwargs):
        """
        Evaluate the current model on a series of experiences and
        returns the last recorded value for each metric.

        :param exp_list: CL experience information.
        :param kwargs: custom arguments.

        :return: dictionary containing last recorded value for
            each metric name
        """
        self.is_training = False
        self.model.eval()

        if isinstance(exp_list, RLExperience):
            exp_list: List[RLExperience] = [exp_list]

        self._before_eval(*kwargs)
        for self.experience in exp_list:
            self.environment = self.experience.environment
            # only single env supported during evaluation
            self.n_envs = self.experience.n_envs

            # Create test Environment
            self.environment = self.make_eval_env(**kwargs)

            # Model Adaptation (e.g. freeze/add new units)
            # self.model_adaptation()

            self._before_eval_exp(**kwargs)
            self.evaluate_exp(**kwargs)
            self._after_eval_exp(**kwargs)

        self._after_eval(**kwargs)

        res = self.evaluator.get_last_metrics()

        return res

    def evaluate_exp(self):
        # rewards per episode
        self.eval_rewards = {'past_returns': [
            0. for _ in range(self.eval_episodes)]}
        # single env only here
        self.eval_ep_lengths = {0: []}
        # TODO: evaluate on self.eval_episodes parallel environments at once (evaluate_exp_parallel function)
        for ep_no in range(self.eval_episodes):
            obs = self.environment.reset()
            for t in count(start=1):
                # this may get stuck with games such as breakout and deterministic dqn
                # if we let no op action be selected indefinitely
                action = self.model.get_action(
                    obs.unsqueeze(0).to(self.device),
                    task_label=self.experience.task_label)
                obs, reward, done, info = self.environment.step(action.item())
                # TODO: use info
                self.eval_rewards['past_returns'][ep_no] += reward
                if done:
                    break

            self.eval_ep_lengths[0].append(t)

        # needed if env comes from train stream and is thus shared
        self.environment.reset()
        self.environment.close()

    def _model_forward(self, model: nn.Module, observations: torch.Tensor, *
                       args, **kwargs):
        """Method for handling forward passage of model handling task label retrieval.
        Args:
            model (nn.Module): Pytorch model to feed observations to.
            observations (torch.Tensor): Input to model.
        Returns:
            Model output.
        """

        exp: RLExperience = getattr(self, 'experience', None)

        task_label = None
        if exp is not None:
            task_label = exp.task_label

        return model(observations, *args, **kwargs, task_label=task_label)
