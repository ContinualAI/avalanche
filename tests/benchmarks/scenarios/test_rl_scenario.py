from avalanche.benchmarks.scenarios.rl_scenario import RLScenario, RLExperience
import unittest
import numpy as np
try:
    import gym
    skip = False
except ImportError:
    skip = True


@unittest.skipIf(skip, reason="Need gym to run these tests")
def test_simple_scenario():
    n_envs = 3
    envs = [gym.make('CartPole-v1') for _ in range(n_envs)]
    rl_scenario = RLScenario(envs, n_parallel_envs=1,
                             task_labels=True, eval_envs=[])        
    tr_stream = rl_scenario.train_stream
    assert len(tr_stream) == n_envs
    assert not len(rl_scenario.eval_stream) 

    for i, exp in enumerate(tr_stream):
        assert exp.current_experience == i
        env = exp.environment     
        assert isinstance(env, gym.Env)   
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
