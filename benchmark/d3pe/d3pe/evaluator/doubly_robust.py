import torch
import numpy as np
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from d3pe.evaluator import Evaluator, Policy
from d3pe.utils.data import OPEDataset, to_torch
from d3pe.utils.tools import bc, FQE
from d3pe.utils.func import hard_clamp

class DREvaluator(Evaluator):
    r'''
        Implementation of Doubly-Robust algorithm, which compute the following equation:
        
        $DR_{H+1-t}^i = \mathbb{E}_{a \sim \pi(a|s_t^i)}[Q(s_t^i, a)] + \rho_t^i (r_t^i + \gamma DR_{H-t}^{i} - Q(s_t^i, a_t^i))$
    '''
    def initialize(self, 
                   train_dataset : OPEDataset = None, 
                   val_dataset : OPEDataset = None, 
                   bc_epoch : int = 20,
                   fqi_steps : int = 100000,
                   gamma : float = 0.99,
                   critic_hidden_features : int = 1024,
                   critic_hidden_layers : int = 4,
                   critic_type : str = 'mlp',
                   atoms : int = 51,
                   device : str = 'cuda' if torch.cuda.is_available() else 'cpu',
                   log : str = None,
                   verbose : bool = False,
                   *args, **kwargs):
        assert train_dataset is not None or val_dataset is not None, 'you need to provide at least one dataset to run IS'
        self.dataset = val_dataset or train_dataset
        assert self.dataset.has_trajectory, 'Doubly-Robust Evaluator only work with trajectory dataset!'
        self.bc_epoch = bc_epoch
        self.fqi_steps = fqi_steps
        self.critic_hidden_features = critic_hidden_features
        self.critic_hidden_layers = critic_hidden_layers
        self.critic_type = critic_type
        self.atoms = atoms
        self.gamma = gamma
        self.device = device
        self.verbose = verbose
        self.log = log
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

        critic = FQE(self.dataset, policy, 
                     num_steps=self.fqi_steps, 
                     critic_hidden_features=self.critic_hidden_features,
                     critic_hidden_layers=self.critic_hidden_layers,
                     critic_type=self.critic_type,
                     atoms=self.atoms,
                     gamma=self.gamma,
                     log=self.log,
                     verbose=self.verbose)

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

        with torch.no_grad():
            estimations = []
            for traj in self.dataset.get_trajectory():
                traj = to_torch(traj, device=self.device)

                # compute rho
                behavior_action_dist = self.behavior_policy(traj['obs'])
                action = hard_clamp(traj['action'], self.min_actions, self.max_actions, shrink=5e-5)
                behavior_policy_log_prob = behavior_action_dist.log_prob(action).sum(dim=-1)
                evaluated_action_dist = recover_policy(traj['obs'])
                evaluated_policy_log_prob = evaluated_action_dist.log_prob(action).sum(dim=-1)
                rho = torch.exp(evaluated_policy_log_prob - behavior_policy_log_prob)

                # compute reward and value
                reward = traj['reward'].squeeze()

                if self.critic_type == 'mlp':
                    q_value = critic(torch.cat([traj['obs'], traj['action']], dim=-1))
                    policy_action = torch.as_tensor(policy.get_action(traj['obs'])).to(traj['action'])
                    v_value = critic(torch.cat([traj['obs'], policy_action], dim=-1))
                elif self.critic_type == 'distributional':
                    q_value = critic(traj['obs'], traj['action'])
                    policy_action = torch.as_tensor(policy.get_action(traj['obs'])).to(traj['action'])
                    v_value = critic(traj['obs'], policy_action)

                # compute DR
                DR = 0
                for t in reversed(range(len(rho))):
                    DR = v_value[t] + rho[t] * (reward[t] + self.gamma * DR - q_value[t])

                estimations.append(DR.item())

        # filter unpractical values
        min_value, max_value = self.dataset.get_value_boundary(self.gamma)
        # estimations = list(filter(lambda x: min_value < x < max_value, estimations))
        estimations = np.array(estimations)
        estimations = np.clip(estimations, min_value, max_value)

        return float(np.mean(estimations))