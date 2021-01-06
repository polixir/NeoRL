from porl.porl_envs import core
import gym
import d4rl

def make_env(task):
    env = gym.make(task)
    #env.get_dataset = core.EnvData.get_dataset
    return env
