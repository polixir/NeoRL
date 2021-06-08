import torch
import numpy as np
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from d3pe.evaluator import Evaluator, Policy
from d3pe.utils.func import vector_stack, hard_clamp
from d3pe.utils.data import OPEDataset, to_torch
from d3pe.utils.tools import bc

class ISEvaluator(Evaluator):
    def initialize(self, 
                   train_dataset : OPEDataset = None, 
                   val_dataset : OPEDataset = None, 
                   bc_epoch : int = 50,
                   gamma : float = 0.99,
                   device : str = 'cuda' if torch.cuda.is_available() else 'cpu',
                   mode : str = 'step',
                   log : str = None,
                   verbose : bool = False,
                   *args, **kwargs):
        assert train_dataset is not None or val_dataset is not None, 'you need to provide at least one dataset to run IS'
        self.dataset = val_dataset or train_dataset
        assert self.dataset.has_trajectory, 'Important Sampling Evaluator only work with trajectory dataset!'
        assert mode in ['trajectory', 'step'], 'mode should be chosen from `trajectory` and `step`'
        self.bc_epoch = bc_epoch
        self.gamma = gamma
        self.device = device
        self.mode = mode
        self.verbose = verbose
        self.writer = SummaryWriter(log) if log is not None else None

        self.min_actions, self.max_actions = self.dataset.get_action_boundary()
        self.min_actions = torch.as_tensor(self.min_actions, dtype=torch.float32, device=self.device)
        self.max_actions = torch.as_tensor(self.max_actions, dtype=torch.float32, device=self.device)

        ''' clone the behaviorial policy '''
        self.behavior_policy = bc(self.dataset, epoch=self.bc_epoch, verbose=self.verbose)

        self.is_initialized = True

    def __call__(self, policy : Policy) -> float:
        assert self.is_initialized, "`initialize` should be called before call."

        policy = deepcopy(policy)
        policy = policy.to(self.device)

        ''' recover the evaluated policy '''
        # relabel the dataset with action from evaluated policy
        recover_dataset = deepcopy(self.dataset)
        obs = recover_dataset.data['obs']
        recovered_action = []
        with torch.no_grad():
            for i in range(obs.shape[0] // 256 + (obs.shape[0] % 256 > 0)):
                recovered_action.append(policy.get_action(obs[i*256:(i+1)*256]))
            recover_dataset.data['action'] = np.concatenate(recovered_action, axis=0)
        # recover the conditional distribution of evaluated policy
        recover_policy = bc(recover_dataset, min_actions=self.min_actions, max_actions=self.max_actions, epoch=self.bc_epoch, verbose=self.verbose)

        if self.mode == 'trajectory':
            with torch.no_grad():
                ratios = []
                discounted_rewards = []
                for traj in self.dataset.get_trajectory():
                    traj = to_torch(traj, device=self.device)
                    behavior_action_dist = self.behavior_policy(traj['obs'])
                    action = hard_clamp(traj['action'], self.min_actions, self.max_actions, shrink=5e-5)
                    behavior_policy_log_prob = behavior_action_dist.log_prob(action).sum(dim=-1, keepdim=True)
                    evaluated_action_dist = recover_policy(traj['obs'])
                    evaluated_policy_log_prob = evaluated_action_dist.log_prob(action).sum(dim=-1, keepdim=True)
                    ratio = evaluated_policy_log_prob - behavior_policy_log_prob
                    ratio = torch.sum(ratio, dim=0)
                    ratios.append(ratio)
                    discounted_reward = traj['reward'] * (self.gamma ** torch.arange(traj['reward'].shape[0], device=self.device).unsqueeze(dim=-1))
                    discounted_reward = torch.sum(discounted_reward, dim=0)
                    discounted_rewards.append(discounted_reward)
                ratios = torch.cat(ratios)
                ratios = torch.softmax(ratios, dim=0)
                discounted_rewards = torch.cat(discounted_rewards)
                
            return torch.sum(discounted_rewards * ratios).item()
        elif self.mode == 'step':
            with torch.no_grad():
                ratios = []
                discounted_rewards = []
                for traj in self.dataset.get_trajectory():
                    traj = to_torch(traj, device=self.device)
                    behavior_action_dist = self.behavior_policy(traj['obs'])
                    action = hard_clamp(traj['action'], self.min_actions, self.max_actions, shrink=5e-5)
                    behavior_policy_log_prob = behavior_action_dist.log_prob(action).sum(dim=-1)
                    evaluated_action_dist = recover_policy(traj['obs'])
                    evaluated_policy_log_prob = evaluated_action_dist.log_prob(action).sum(dim=-1)
                    ratio = evaluated_policy_log_prob - behavior_policy_log_prob
                    ratio = torch.cumsum(ratio, dim=0)
                    discounted_reward = traj['reward'].squeeze() * (self.gamma ** torch.arange(traj['reward'].shape[0], device=self.device))
                    
                    ratios.append(ratio.cpu().numpy())
                    discounted_rewards.append(discounted_reward.cpu().numpy())

                ratios = vector_stack(ratios, - float('inf'))
                ratios = torch.as_tensor(ratios)
                ratios = torch.softmax(ratios, dim=0).numpy() * ratios.shape[0]
                discounted_rewards = vector_stack(discounted_rewards, 0)
                discounted_rewards = np.sum(discounted_rewards * ratios, axis=1)
                
            return float(np.mean(discounted_rewards))