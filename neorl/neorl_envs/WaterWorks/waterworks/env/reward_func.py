import numpy as np
from typing import Dict

def get_reward(data, is_day):

    def day_reward(x):
        reward = np.zeros_like(x, dtype=x.dtype)
        reward[x<0.25] = -1
        reward[x>0.26] = 0.01 / (x[x>0.26] - 0.26 + 0.01)
        return reward
        
    def night_reward(x):
        reward = np.zeros_like(x, dtype=x.dtype)
        reward[x<0.22] = -1
        reward[x>0.23] = 0.01 / (x[x>0.23] - 0.23 + 0.01)
        return reward

    press = data
    reward = np.where(is_day, day_reward(press), night_reward(press))
    return reward
