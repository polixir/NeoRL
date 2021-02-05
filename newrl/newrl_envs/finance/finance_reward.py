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

    reward_scaling = 1e-4
    stock_dim = 30

    if isinstance(obs, np.ndarray):
        array_type = np
    else:
        array_type = torch

    begin_total_asset = obs[:, 0] + array_type.sum(
        obs[:, 1:(stock_dim + 1)] * obs[:, (stock_dim + 1):(stock_dim * 2 + 1)], axis=1)
    end_total_asset = obs_next[:, 0] + array_type.sum(
        obs_next[:, 1:(stock_dim + 1)] * obs_next[:, (stock_dim + 1):(stock_dim * 2 + 1)], axis=1)
    reward = end_total_asset - begin_total_asset
    reward = reward_scaling * reward

    if single_reward:
        reward = reward[0].item()
    else:
        reward = reward.reshape(-1, 1)

    return reward
