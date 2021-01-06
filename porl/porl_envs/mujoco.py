import gym
from porl.porl_envs import core

def make_env(task):
    env = gym.make(task, exclude_current_positions_from_observation=False)
    env.get_dataset = core.EnvData.get_dataset
    return env
