import numpy as np
import torch
from neorl.neorl_envs.logistics_distribution.ld_env import NORTH, EAST, SOUTH, WEST


def get_reward(data):
    obs = data["obs"]
    action = data["action"]
    obs_next = data["next_obs"]

    singel_reward = False

    if len(obs.shape) == 1:
        obs = obs.reshape(1, -1)
        singel_reward = True
    if len(action.shape) == 1:
        action = action.reshape(1, -1)
    if len(obs_next.shape) == 1:
        obs_next = obs_next.reshape(1, -1)

    if isinstance(obs, np.ndarray):
        array_type = np
    else:
        array_type = torch

    GRID = 3
    _same_street = lambda m, n: m == n and m % GRID == 0 and n % GRID == 0
    _same_direction = lambda m, n: m % GRID == 0 and n % GRID == 0

    def _find_inter(m, n):

        if m // GRID == n // GRID:
            if m % GRID + n % GRID < GRID:
                ret = m - m % GRID
            else:
                ret = m - m % GRID + GRID
        else:
            if m > n:
                ret = m - m % GRID
            else:
                ret = m - m % GRID + GRID
        return ret

    reward = array_type.zeros([obs.shape[0], ])

    for i in range(len(obs)):

        to_2d = lambda p: (p // 8, p % 8)

        points = tuple(map(to_2d, array_type.where(obs[i][:-3] == 2)[0]))

        speed, direction, SPEED_MAX = obs[i][-3:]

        agent_pos = array_type.where(obs[i][:-3] == 3)[0]
        agent_pos = agent_pos.item()
        x, y = to_2d(agent_pos)

        a, b = to_2d(action[i])
        a, b = a.item(), b.item()

        def _dash(a, b):
            nonlocal x, y, direction, speed
            assert _same_street(x, a) or _same_street(y, b)

            ret_time = 0

            if x == a and y == b:
                ret_time += 10
            else:
                if y < b:
                    new_direction = EAST
                elif y > b:
                    new_direction = WEST
                elif x < a:
                    new_direction = SOUTH
                elif x > a:
                    new_direction = NORTH
                else:
                    assert 0
                dist = abs(y - b) + abs(x - a)
                if direction == new_direction:
                    pass
                elif direction == - new_direction:
                    speed = 0
                else:
                    speed = speed / 2
                direction = new_direction
                s_tomax = (SPEED_MAX * SPEED_MAX - speed * speed) / 2
                if dist <= s_tomax:
                    ret_time += np.sqrt(2 * dist + speed * speed) - speed
                    speed += ret_time
                else:
                    ret_time += SPEED_MAX - speed + (dist - s_tomax) / SPEED_MAX
                    speed = SPEED_MAX

                x, y = a, b

            for point in points:
                if (a, b) == point:
                    ret_time -= 10
            return ret_time

        def _reach_place(a, b):
            rew = 0
            if _same_street(x, a) or _same_street(y, b):
                rew += _dash(a, b)
            elif _same_direction(x, b):
                rew += _dash(x, b)
                rew += _dash(a, b)
            elif _same_direction(a, y):
                rew += _dash(a, y)
                rew += _dash(a, b)
            elif _same_direction(x, a):
                tmp = _find_inter(y, b)
                rew += _dash(x, tmp)
                rew += _dash(a, tmp)
                rew += _dash(a, b)
            elif _same_direction(y, b):
                tmp = _find_inter(x, a)
                rew += _dash(tmp, y)
                rew += _dash(tmp, b)
                rew += _dash(a, b)
            else:
                assert 0

            return rew

        if (a % GRID == 0) or (b % GRID == 0):
            r = _reach_place(a, b)
        else:
            r = 100

        reward[i] = -1 * r

    if singel_reward:
        reward = reward[0].item()
    else:
        reward = reward.reshape(-1, 1)
    return reward
