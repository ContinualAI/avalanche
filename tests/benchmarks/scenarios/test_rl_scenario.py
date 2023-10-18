import unittest
import numpy as np

try:
    import gym
    from avalanche.benchmarks.scenarios.reinforcement_learning import (
        RLScenario,
        RLExperience,
    )
    import pygame

    skip = False
except ImportError:
    skip = True


class RLScenarioTests(unittest.TestCase):
    @unittest.skipIf(skip, reason="Need gym and pygame to run these tests")
    def test_simple_scenario(self):
        from packaging import version

        n_envs = 3
        envs = [gym.make("CartPole-v1")] * n_envs
        rl_scenario = RLScenario(
            envs, n_parallel_envs=1, task_labels=True, eval_envs=[]
        )
        tr_stream = rl_scenario.train_stream
        assert len(tr_stream) == n_envs
        assert not len(rl_scenario.eval_stream)

        for i, exp in enumerate(tr_stream):
            assert exp.current_experience == i
            env = exp.environment
            # same envs
            assert exp.task_label == 0
            assert isinstance(env, gym.Env)
            obs = env.reset()
            if version.parse(gym.__version__) >= version.parse("0.26.0"):
                self.assertIsInstance(obs[0], np.ndarray)
                self.assertIsInstance(obs[1], dict)
            else:
                self.assertIsInstance(obs, np.ndarray)

    @unittest.skipIf(skip, reason="Need gym and pygame to run these tests")
    def test_multiple_envs_shuffle(self):
        envs = [
            gym.make("CartPole-v0"),
            gym.make("CartPole-v1"),
            gym.make("Acrobot-v1"),
        ]
        rl_scenario = RLScenario(
            envs=envs,
            n_parallel_envs=1,
            task_labels=True,
            eval_envs=envs[:2],
            shuffle=True,
        )
        tr_stream = rl_scenario.train_stream
        assert len(tr_stream) == 3

        all_t_labels = set()
        for i, exp in enumerate(tr_stream):
            assert exp.current_experience == i
            all_t_labels.add(exp.task_label)

        self.assertSetEqual(set(range(3)), all_t_labels)

        assert len(rl_scenario.eval_stream) == 2
        for i, exp in enumerate(rl_scenario.eval_stream):
            assert exp.task_label == i
            assert exp.environment.spec.id == envs[i].spec.id

    @unittest.skipIf(skip, reason="Need gym and pygame to run these tests")
    def test_multiple_envs(self):
        envs = [
            gym.make("CartPole-v0"),
            gym.make("CartPole-v1"),
            gym.make("Acrobot-v1"),
        ]
        rl_scenario = RLScenario(
            envs, n_parallel_envs=1, task_labels=True, eval_envs=envs[:2]
        )
        tr_stream = rl_scenario.train_stream
        assert len(tr_stream) == 3

        for i, exp in enumerate(tr_stream):
            assert exp.current_experience == i == exp.task_label

        assert len(rl_scenario.eval_stream) == 2
        for i, exp in enumerate(rl_scenario.eval_stream):
            assert exp.task_label == i
            assert exp.environment.spec.id == envs[i].spec.id

        # deep copies of the same env are considered as different tasks
        envs = [gym.make("CartPole-v1") for _ in range(3)]
        eval_envs = [gym.make("CartPole-v1")] * 2
        rl_scenario = RLScenario(
            envs, n_parallel_envs=1, task_labels=True, eval_envs=eval_envs
        )
        for i, exp in enumerate(rl_scenario.train_stream):
            assert exp.task_label == i
        # while shallow copies in eval behave like the ones in train
        assert len(rl_scenario.eval_stream) == 2
        for i, exp in enumerate(rl_scenario.eval_stream):
            assert exp.task_label == 0
            assert exp.environment.spec.id == envs[0].spec.id


if __name__ == "__main__":
    unittest.main()
