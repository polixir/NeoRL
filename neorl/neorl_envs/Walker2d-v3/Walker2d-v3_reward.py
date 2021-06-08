import numpy as np

def get_reward(data):
    obs = data["obs"]
    action = data["action"]
    obs_next = data["next_obs"]

    singel_reward = False
    if len(obs.shape) == 1:
        singel_reward = True
        obs = obs.reshape(1, -1)
    if len(action.shape) == 1:
        action = action.reshape(1, -1)
    if len(obs_next.shape) == 1:
        obs_next = obs_next.reshape(1, -1)

    timestep = 0.002
    frame_skip = 4
    dt = timestep * frame_skip
    x_velocity = (obs_next[:, 0] - obs[:, 0]) / dt
    forward_reward = x_velocity

    #healthy_z = [0.8 < i < 2.0 for i in obs[:,1]]
    #healthy_angle = [-1 < i < 1 for i in obs[:,2]]
    #is_healthy = healthy_z and healthy_angle
    healthy_reward = 1 #is_healthy

    rewards = forward_reward + healthy_reward
    costs = 1e-3 * (action ** 2).sum(axis=-1)
    reward = rewards - costs

    if singel_reward:
        reward = reward[0].item()
    else:
        reward = reward.reshape(-1, 1)
    return reward