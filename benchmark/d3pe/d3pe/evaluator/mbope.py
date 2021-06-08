import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from d3pe.evaluator import Evaluator, Policy
from d3pe.utils.data import OPEDataset, to_torch
from d3pe.utils.net import MLP

class MBOPEEvaluator(Evaluator):
    def initialize(self, 
                   train_dataset : OPEDataset = None, 
                   val_dataset : OPEDataset = None, 
                   model_type : str = 'mlp',
                   batch_size : int = 256,
                   gamma : float = 0.99,
                   horizon : int = 500,
                   device : str = "cuda" if torch.cuda.is_available() else "cpu",
                   log : str = None,
                   *args, **kwargs):
        assert train_dataset is not None or val_dataset is not None, 'you need to provide at least one dataset to run MBOPE!'
        self.dataset = val_dataset or train_dataset
        self.model_type = model_type
        self.batch_size = batch_size
        self.gamma = gamma
        self.horizon = horizon
        self.device = device
        self.log = log
        self.writer = SummaryWriter(log) if log is not None else None

        self.max_values = np.concatenate([self.dataset[:]['next_obs'], self.dataset[:]['reward']], axis=-1).max(axis=0)
        self.min_values = np.concatenate([self.dataset[:]['next_obs'], self.dataset[:]['reward']], axis=-1).min(axis=0)
        self.max_values = torch.as_tensor(self.max_values, dtype=torch.float32, device=self.device)
        self.min_values = torch.as_tensor(self.min_values, dtype=torch.float32, device=self.device)

        '''learn a model here'''
        data = self.dataset[0]
        if self.model_type == 'mlp':
            self.trainsition = MLP(data['obs'].shape[-1] + data['action'].shape[-1], data['obs'].shape[-1] + 1, 1024, 4).to(self.device)
        optim = torch.optim.Adam(self.trainsition.parameters(), lr=1e-3, weight_decay=1e-5)
        for _ in range(100000):
            data = self.dataset.sample(self.batch_size)
            data = to_torch(data, device=self.device)
            next = self.trainsition(torch.cat([data['obs'], data['action']], dim=-1))
            loss = ((next - torch.cat([data['reward'], data['next_obs']], dim=-1)) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

        self.is_initialized = True

    def __call__(self, policy : Policy) -> float:
        assert self.is_initialized, "`initialize` should be called before call."

        init_state = self.dataset.get_initial_states()
        init_state = to_torch(init_state, torch.float32, device=self.device)

        with torch.no_grad():
            reward = 0
            obs = init_state['obs']
            for t in range(self.horizon):
                action = torch.as_tensor(policy.get_action(obs)).to(obs)
                next = self.trainsition(torch.cat([obs, action], dim=-1))
                next = torch.min(next, self.max_values)
                next = torch.max(next, self.min_values)
                r = next[..., 0]
                obs = next[..., 1:]
                reward += self.gamma ** t * r

        return reward.mean().item()