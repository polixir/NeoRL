import ray
import gym
import numpy as np
from copy import deepcopy

from d3pe.evaluator import Evaluator, Policy
from d3pe.utils.data import OPEDataset
from d3pe.utils.env import get_env

@ray.remote
def test_one_trail(env : gym.Env, policy : Policy):
    env = deepcopy(env)
    policy = deepcopy(policy)

    state, done = env.reset(), False
    rewards = 0
    lengths = 0
    while not done:
        state = state[np.newaxis]
        action = policy.get_action(state).reshape(-1)
        state, reward, done, _ = env.step(action)
        rewards += reward
        lengths += 1

    return (rewards, lengths)

def test_on_real_env(env : gym.Env, policy : Policy, number_of_runs : int = 10):
    rewards = []
    episode_lengths = []

    results = ray.get([test_one_trail.remote(env, policy) for _ in range(number_of_runs)])
    rewards = [result[0] for result in results]
    episode_lengths = [result[1] for result in results]

    rew_mean = np.mean(rewards)
    len_mean = np.mean(episode_lengths)

    return {
        "online_reward" : rew_mean,
        "online_length" : len_mean,
    }

class OnlineEvaluator(Evaluator):
    def initialize(self, 
                   train_dataset : OPEDataset, 
                   val_dataset : OPEDataset, 
                   task : str, 
                   number_of_runs : int = 10, 
                   *args, **kwargs):
        self.task = task
        self.number_of_runs = number_of_runs
        self.env = get_env(self.task)
        self.is_initialized = True

    def __call__(self, policy : Policy) -> float:
        assert self.is_initialized, "`initialize` should be called before callback."
        policy = deepcopy(policy).cpu()
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        return test_on_real_env(self.env, policy, self.number_of_runs)['online_reward']