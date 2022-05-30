import copy
import itertools
import os
import numpy as np
import torch
from tqdm import tqdm
from neorl.neorl_envs.logistics_distribution.ld_env import LogisticsDistributionEnv


log_path = 'log/tsp-v0/dqn/'


def cal_optim(env: LogisticsDistributionEnv):
    """
    Theoretically optimal strategy
    :return: (optim_return, length)
    """
    points = list(env.points.keys())
    envs = [copy.deepcopy(env) for _ in range(6)]
    _res = []
    for i, traj in enumerate(itertools.permutations(points)):
        ret, len_traj, d = 0, 0, False
        for a, b in traj:
            if (envs[i].unfinished == 0) or d:
                break
            if envs[i].points[(a, b)]:
                s, r, d, info = envs[i].step(a * env.LENGTH + b)
                len_traj += 1
                ret += r
        _res.append((ret, len_traj))
    optim_return, optim_length = sorted(_res, key=lambda x: x[0], reverse=True)[0]
    return optim_return, optim_length


def print_res(level, res):
    print("-" * 5 + level + "-" * 5)
    for logged_return, test_mean, test_length_mean in sorted(res, key=lambda x: x[1], reverse=True):
        print(f"{logged_return}\t| {test_mean:.3f}\t {test_length_mean:.0f}")
    print('\n')


def lod_net(net, path):
    tmp_model = dict()
    logged_model = torch.load(os.path.join(log_path, path), map_location=torch.device('cpu'))
    for k, v in logged_model.items():
        if k.startswith('model.model.'):
            new_key = k.replace('model.model.', 'model.')
            tmp_model[new_key] = v
    net.load_state_dict(tmp_model)
    return net.eval()


def mc_return(env, net, pi_list, max_length=30, mc_times=1000):
    res = []
    with torch.no_grad():
        #     for path in os.listdir(log_path):
        for path in tqdm(pi_list):
            net = lod_net(net, path)
            episode_rewards, episode_lengths = [], []
            for _ in range(mc_times):
                s, d = env.reset(), False
                R = 0
                i = 0
                while not d:
                    q_s = net(s[np.newaxis])[0]
                    a = torch.argmax(q_s).item()
                    s, r, d, info = env.step(a)
                    R += r
                    i += 1
                    if i > max_length:
                        break
                episode_rewards.append(R)
                episode_lengths.append(i)
            res.append((int(path.split('_')[0]), np.mean(episode_rewards), np.mean(episode_lengths)))
    return res
