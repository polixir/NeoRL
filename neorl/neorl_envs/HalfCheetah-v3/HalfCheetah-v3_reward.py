import numpy as np
import torch

def get_reward(data):    
    obs = data["obs"]
    action = data["action"]
    obs_next = data["next_obs"]
    singel_reward = False
    if len(obs.shape) == 1:
        obs = obs.reshape(1,-1)
        singel_reward = True
    if len(action.shape) == 1:
        action = action.reshape(1,-1)
    if len(obs_next.shape) == 1:
        obs_next = obs_next.reshape(1,-1)
    
    
    forward_reward_weight = 1.0 
    ctrl_cost_weight = 0.1
    dt = 0.05
    
    if isinstance(obs, np.ndarray):
        array_type = np
    else:
        array_type = torch
    
    ctrl_cost = ctrl_cost_weight * array_type.sum(array_type.square(action),axis=1)
    
    x_position_before = obs[:,0]
    x_position_after = obs_next[:,0]
    x_velocity = ((x_position_after - x_position_before) / dt)

    forward_reward = forward_reward_weight * x_velocity
    
    reward = forward_reward - ctrl_cost
    
    if singel_reward:
        reward = reward[0].item()
    else:
        reward = reward.reshape(-1,1)
    return reward