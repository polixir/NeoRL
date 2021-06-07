def get_reward(data):
    obs = data["obs"]
    action = data["action"]
    obs_next = data["next_obs"]

    single_reward = False
    if len(obs.shape) == 1:
        single_reward = True
        obs = obs.reshape(1, -1)
    if len(action.shape) == 1:
        action = action.reshape(1, -1)
    if len(obs_next.shape) == 1:
        obs_next = obs_next.reshape(1, -1)

    CRF = 3.0
    CRC = 1.0

    fatigue = obs_next[:, 4]
    consumption = obs_next[:, 5]

    cost = CRF * fatigue + CRC * consumption

    reward = -cost

    if single_reward:
        reward = reward[0].item()
    else:
        reward = reward.reshape(-1, 1)

    return reward
