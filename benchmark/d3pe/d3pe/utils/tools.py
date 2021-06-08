''' This file contain common tools shared across different OPE algorithms '''

import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import Optional, Union
from d3pe.utils.data import OPEDataset, to_torch
from d3pe.evaluator import Policy
from d3pe.utils.net import MLP, DistributionalCritic, TanhGaussianActor
from d3pe.utils.func import hard_clamp

def bc(dataset : OPEDataset, 
       min_actions : Optional[Union[torch.Tensor, np.ndarray, float]] = None,
       max_actions : Optional[Union[torch.Tensor, np.ndarray, float]] = None,
       policy_features : int = 256,
       policy_layers : int = 2,
       val_ratio : float = 0.2,
       batch_size : int = 256,
       epoch : int = 20,
       lr : float = 3e-4,
       weight_decay : float = 1e-5,
       device : str = "cuda" if torch.cuda.is_available() else "cpu",
       verbose : bool = False) -> TanhGaussianActor:

    ''' clone the policy in the dataset '''

    data = dataset[0]
    if min_actions is None: min_actions = dataset.get_action_boundary()[0]
    if max_actions is None: max_actions = dataset.get_action_boundary()[1]
    policy = TanhGaussianActor(data['obs'].shape[-1], data['action'].shape[-1], policy_features, policy_layers, min_actions, max_actions).to(device)
    max_actions = torch.as_tensor(max_actions, dtype=torch.float32, device=device)
    min_actions = torch.as_tensor(min_actions, dtype=torch.float32, device=device)

    best_parameters = deepcopy(policy.state_dict())
    best_loss = float('inf')
    
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    optim = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    
    device = next(policy.parameters()).device
    
    if verbose: timer = tqdm(total=epoch)

    for _ in range(epoch):
        policy.train()
        train_losses = []
        for data in iter(train_loader):
            data = to_torch(data, device=device)
            action_dist = policy(data['obs'])
            loss = - action_dist.log_prob(hard_clamp(data['action'], min_actions, max_actions, shrink=5e-5)).mean()
            train_losses.append(loss.item())
            
            optim.zero_grad()
            loss.backward()
            optim.step()

        policy.eval()
        with torch.no_grad():
            val_loss = 0
            for data in iter(val_loader):
                data = to_torch(data, device=device)
                action_dist = policy(data['obs'])
                val_loss += - action_dist.log_prob(hard_clamp(data['action'], min_actions, max_actions, shrink=5e-5)).sum().item()
        val_loss /= len(val_dataset) * data['action'].shape[-1]

        if val_loss < best_loss:
            best_loss = val_loss
            best_parameters = deepcopy(policy.state_dict())

        if verbose: 
            timer.update(1)
            timer.set_description('train : %.3f, val : %.3f, best : %.3f' % (np.mean(train_losses), val_loss, best_loss))

    if verbose: timer.close()

    policy.load_state_dict(best_parameters)

    return policy

def FQE(dataset : OPEDataset,
        policy : Policy,
        num_steps : int = 500000,
        batch_size : int = 256,
        lr : float = 1e-4,
        weight_decay : float = 1e-5,
        init_critic : Optional[Union[MLP, DistributionalCritic]] = None,
        critic_hidden_features : int = 1024,
        critic_hidden_layers : int = 4,
        critic_type : str = 'distributional',
        atoms : int = 51,
        gamma : float = 0.99,
        device : str = "cuda" if torch.cuda.is_available() else "cpu",
        log : str = None,
        verbose : bool = False,
        *args, **kwargs) -> Union[MLP, DistributionalCritic]:

        ''' solve the value function of the policy given the dataset '''

        writer = torch.utils.tensorboard.SummaryWriter(log) if log is not None else None

        min_value, max_value = dataset.get_value_boundary(gamma)

        policy = deepcopy(policy)
        policy = policy.to(device)

        data = dataset.sample(batch_size)
        if init_critic is not None:
            critic = deepcopy(init_critic)
        else:
            if critic_type == 'mlp':
                critic = MLP(data['obs'].shape[-1] + data['action'].shape[-1], 1, critic_hidden_features, critic_hidden_layers).to(device)
            elif critic_type == 'distributional':
                critic = DistributionalCritic(data['obs'].shape[-1], data['action'].shape[-1], 
                                            critic_hidden_features, critic_hidden_layers,
                                            min_value, max_value, atoms).to(device)

        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr, weight_decay=weight_decay)
        target_critic = deepcopy(critic).to(device)
        target_critic.requires_grad_(False)

        if verbose:
            counter = tqdm(total=num_steps)

        for t in range(num_steps):
            batch = dataset.sample(batch_size)
            data = to_torch(batch, torch.float32, device=device)
            r = data['reward']
            terminals = data['done']
            o = data['obs']
            a = data['action']

            o_ = data['next_obs']
            a_ = torch.as_tensor(policy.get_action(o_), dtype=torch.float32, device=device)

            if isinstance(critic, MLP):
                q_target = target_critic(torch.cat((o_, a_), -1)).detach()
                current_discount = gamma * (1 - terminals)
                backup = r + current_discount * q_target
                backup = torch.clamp(backup, min_value, max_value) # prevent explosion
                
                q = critic(torch.cat((o, a), -1))
                critic_loss = ((q - backup) ** 2).mean()
            elif isinstance(critic, DistributionalCritic):
                q, p = critic(o, a, with_p=True)
                target_p = target_critic.get_target(o_, a_, r, gamma * (1 - terminals))
                critic_loss = - (target_p * torch.log(p + 1e-8)).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            if writer is not None:
                writer.add_scalar('q', scalar_value=q.mean().item(), global_step=t)
        
            if t % 100 == 0:
                with torch.no_grad():
                    target_critic.load_state_dict(critic.state_dict())

            if verbose:
                counter.update(1)
                counter.set_description('loss : %.3f, q : %.3f' % (critic_loss.item(), q.mean().item()))
        
        if verbose: counter.close()

        return critic