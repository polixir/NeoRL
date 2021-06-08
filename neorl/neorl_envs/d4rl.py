import gym
from neorl import core
import d4rl


def make_env(task):
    env = gym.make(task)
    env_data = core.EnvData()
    env.set_name = env_data.set_name
    env.set_reward_func = env_data.set_reward_func
    env.get_reward_func = env_data.get_reward_func
    return env
