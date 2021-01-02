import gym

def make_env(task):
    env = gym.make(task, exclude_current_positions_from_observation=False)
    
    return env