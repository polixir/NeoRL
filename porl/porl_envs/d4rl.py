import gym
import d4rl

def make_env(task):
    env = gym.make(task)
    
    return env