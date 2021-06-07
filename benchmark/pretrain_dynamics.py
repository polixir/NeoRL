import os
import ray
import torch
import numpy as np
from ray import tune

from offlinerl.utils.exp import setup_seed
from offlinerl.data import load_data_from_neorl
from offlinerl.utils.net.model.ensemble import EnsembleTransition

SEEDS = [7, 42, 210]

def _select_best_indexes(metrics, n):
    pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
    pairs = sorted(pairs, key=lambda x: x[0])
    selected_indexes = [pairs[i][1] for i in range(n)]
    return selected_indexes

def _train_transition(transition, data, optim, device='cuda'):
    data.to_torch(device=device)
    dist = transition(torch.cat([data['obs'], data['act']], dim=-1))
    loss = - dist.log_prob(torch.cat([data['obs_next'], data['rew']], dim=-1))
    loss = loss.mean()

    loss = loss + 0.01 * transition.max_logstd.mean() - 0.01 * transition.min_logstd.mean()

    optim.zero_grad()
    loss.backward()
    optim.step()
    
def _eval_transition(transition, valdata, device='cuda'):
    with torch.no_grad():
        valdata.to_torch(device=device)
        dist = transition(torch.cat([valdata['obs'], valdata['act']], dim=-1))
        loss = ((dist.mean - torch.cat([valdata['obs_next'], valdata['rew']], dim=-1)) ** 2).mean(dim=(1,2))
        return list(loss.cpu().numpy())

def training_dynamics(config):
    if config["task"] == 'finance' and config["amount"] == 10000:
        return {
            'performance' : [],
            'path' : '',
        }


    seed = config['seed']
    setup_seed(seed)

    train_buffer, val_buffer = load_data_from_neorl(config["task"], config["level"], config["amount"])

    obs_shape = train_buffer['obs'].shape[-1]
    action_shape = train_buffer['act'].shape[-1]

    device = 'cuda'

    hidden_units = 1024 if config["task"] in ['ib', 'finance', 'citylearn'] else 256
    transition = EnsembleTransition(obs_shape, action_shape, hidden_units, 4, 7).to(device)
    transition_optim = torch.optim.AdamW(transition.parameters(), lr=1e-3, weight_decay=0.000075)

    data_size = len(train_buffer)
    val_size = min(int(data_size * 0.2) + 1, 1000)
    train_size = data_size - val_size
    train_splits, val_splits = torch.utils.data.random_split(range(data_size), (train_size, val_size))
    valdata = train_buffer[val_splits.indices]
    train_buffer = train_buffer[train_splits.indices]

    batch_size = 256

    val_losses = [float('inf') for i in range(7)]

    epoch = 0
    cnt = 0

    while True:
        epoch += 1
        idxs = np.random.randint(train_buffer.shape[0], size=[7, train_buffer.shape[0]])
        for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
            batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            batch = train_buffer[batch_idxs]
            _train_transition(transition, batch, transition_optim, device)
        new_val_losses = _eval_transition(transition, valdata, device)

        indexes = []
        for i, new_loss, old_loss in zip(range(len(val_losses)), new_val_losses, val_losses):
            if new_loss < old_loss:
                indexes.append(i)
                val_losses[i] = new_loss

        if len(indexes) > 0:
            transition.update_save(indexes)
            cnt = 0
        else:
            cnt += 1

        if cnt >= 5:
            break
    
    indexes = _select_best_indexes(val_losses, n=5)
    transition.set_select(indexes)
    performance = _eval_transition(transition, valdata, device)
    transition_path = os.path.join(config['dynamics_path'], f'{config["task"]}-{config["level"]}-{config["amount"]}-{seed}.pt')

    torch.save(transition, transition_path)

    return {
        'performance' : performance,
        'path' : transition_path,
    }

if __name__ == '__main__':
    if not os.path.exists('dynamics'):
        os.makedirs('dynamics')

    ray.init()

    abs_path = os.path.abspath('dynamics')

    config = {}
    config['task'] = tune.grid_search(['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3', 'ib', 'finance', 'citylearn'])
    config['level'] = tune.grid_search(['low', 'medium', 'high'])
    config['amount'] = tune.grid_search([100, 1000, 10000])
    config['seed'] = tune.grid_search(SEEDS)
    config['dynamics_path'] = abs_path

    analysis = tune.run(
        training_dynamics,
        name='dynamics',
        config=config,
        queue_trials=True,
        resources_per_trial={
            "cpu": 1,
            "gpu": 1.0,
        }
    )

    df = analysis.results_df

    df.to_pickle(os.path.join(abs_path, 'summary.pkl'))