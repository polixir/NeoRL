from porl import core
import gym


def make_env(task):
    env = gym.make(task)
    env_data = core.EnvData()
    env_data.set_name(task)
    env.get_dataset = env_data.get_dataset
    env.get_dataset_by_traj_num = env_data.get_dataset_by_traj_num
    return env
