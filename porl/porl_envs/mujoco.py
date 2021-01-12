import gym
from porl.porl_envs import core

def make_env(task):
    env = gym.make(task, exclude_current_positions_from_observation=False)
    env_data = core.EnvData()
    env_data.name = task
    env.get_dataset = env_data.get_dataset
    env.get_dataset_by_traj_num = env_data.get_dataset_by_traj_num
    return env
