import gym
from neorl import core


def make_env(task):
    env = gym.make(task, exclude_current_positions_from_observation=False)
    env_data = core.EnvData()
    env.set_name = env_data.set_name
    env.get_dataset = env_data.get_dataset
    env.set_reward_func = env_data.set_reward_func
    env.get_reward_func = env_data.get_reward_func
    env.set_done_func = env_data.set_done_func
    env.get_done_func = env_data.get_done_func
    
    
    return env
