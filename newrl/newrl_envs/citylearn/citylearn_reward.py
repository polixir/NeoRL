import numpy as np
import torch


def get_reward(data):    
    obs = data["obs"]
    action = data["action"]
    obs_next = data["next_obs"]
    
    single_reward = False
    if len(obs.shape) == 1:
        obs = obs.reshape(1, -1)
        single_reward = True
    if len(action.shape) == 1:
        action = action.reshape(1, -1)
    if len(obs_next.shape) == 1:
        obs_next = obs_next.reshape(1, -1)

    index = [25, 32, 38, 45, 52, 59, 66, 73]
    electricity_demand = - obs_next[:, index]
    
    if isinstance(obs, np.ndarray):
        reward_ = - np.sum(electricity_demand, axis=1)
        reward_ = np.clip(reward_, 0, np.max(reward_))
    else:
        reward_ = - torch.sum(electricity_demand, axis=1)
        reward_ = torch.clamp(reward_, 0, torch.max(reward_))

    reward = reward_ ** 3.0 * 0.00001

    if single_reward:
        reward = reward[0].item()
    else:
        reward = reward.reshape(-1, 1)

    return reward
