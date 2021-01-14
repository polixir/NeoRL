import gym
from porl import core


def make_env(task):
    env = gym.make(task, exclude_current_positions_from_observation=False)
    env_data = core.EnvData()
    env_data.set_name(task)
    env.get_dataset = env_data.get_dataset
    return env
