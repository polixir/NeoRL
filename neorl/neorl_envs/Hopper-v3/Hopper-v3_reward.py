def get_reward(data):
    o = data['obs']
    a = data['action']
    o_ = data['next_obs']

    x_position_before = o[:, 0]
    x_position_after = o_[:, 0]
    dt = 0.008
    _forward_reward_weight = 1.0
    x_velocity = (x_position_after - x_position_before) / dt

    forward_reward = _forward_reward_weight * x_velocity
    healthy_reward = 1.0

    rewards = forward_reward + healthy_reward
    costs = (a ** 2).sum(axis=-1)

    reward = rewards - 1e-3 * costs

    # (batch_size, )
    return reward
