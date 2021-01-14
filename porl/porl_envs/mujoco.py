import gym
from porl import core


def make_env(task):
    env = gym.make(task, exclude_current_positions_from_observation=False)
    env_data = core.EnvData()
    env.set_name = env_data.set_name
    env.get_dataset = env_data.get_dataset
    env.set_reward_func = env_data.set_reward_func
    return env
