import torch
from copy import deepcopy

from d3pe.evaluator import Evaluator, Policy
from d3pe.utils.data import OPEDataset
from d3pe.utils.tools import bc, FQE

class FQEEvaluator(Evaluator):
    def initialize(self, 
                   train_dataset : OPEDataset = None, 
                   val_dataset : OPEDataset = None, 
                   pretrain : bool = False, 
                   fqi_steps : int = 250000,
                   critic_hidden_features : int = 1024,
                   critic_hidden_layers : int = 4,
                   critic_type : str = 'mlp',
                   atoms : int = 51,
                   gamma : float = 0.99,
                   device : str = "cuda" if torch.cuda.is_available() else "cpu",
                   log : str = None,
                   verbose : bool = False,
                   *args, **kwargs):
        assert train_dataset is not None or val_dataset is not None, 'you need to provide at least one dataset to run FQE'
        self.dataset = val_dataset or train_dataset
        self.pretrain = pretrain
        self.fqi_steps = fqi_steps
        self.critic_hidden_features = critic_hidden_features
        self.critic_hidden_layers = critic_hidden_layers
        self.critic_type = critic_type
        self.atoms = atoms
        self.gamma = gamma
        self.device = device
        self.log = log
        self.verbose = verbose

        if self.pretrain:
            '''implement a base value function here'''

            # clone the behaviorial policy
            policy = bc(self.dataset, epoch=20, verbose=verbose)

            policy.get_action = lambda x: policy(x).mean
            self.init_critic = FQE(self.dataset, policy, 
                                   num_steps=self.fqi_steps // 2, 
                                   init_critic=self.init_critic,
                                   critic_hidden_features=self.critic_hidden_features,
                                   critic_hidden_layers=self.critic_hidden_layers,
                                   critic_type=self.critic_type,
                                   atoms=self.atoms,
                                   gamma=self.gamma,
                                   log=self.log,
                                   verbose=self.verbose)
        else:
            self.init_critic = None

        self.is_initialized = True

    def __call__(self, policy : Policy) -> float:
        assert self.is_initialized, "`initialize` should be called before call."

        policy = deepcopy(policy)
        policy = policy.to(self.device)
        
        num_steps = self.fqi_steps // 2 if self.pretrain else self.fqi_steps 
        critic = FQE(self.dataset, policy, 
                     num_steps=num_steps, 
                     init_critic=self.init_critic,
                     critic_hidden_features=self.critic_hidden_features,
                     critic_hidden_layers=self.critic_hidden_layers,
                     critic_type=self.critic_type,
                     atoms=self.atoms,
                     gamma=self.gamma,
                     log=self.log,
                     verbose=self.verbose)

        if self.dataset.has_trajectory:
            data = self.dataset.get_initial_states()
            obs = data['obs']
            obs = torch.tensor(obs).float()
            batches = torch.split(obs, 256, dim=0)
        else:
            batches = [torch.tensor(self.dataset.sample(256)['obs']).float() for _ in range(100)]

        estimate_q0 = []
        with torch.no_grad():
            for o in batches:
                o = o.to(self.device)
                a = torch.as_tensor(policy.get_action(o)).to(o)
                if self.critic_type == 'mlp':
                    init_sa = torch.cat((o, a), -1).to(self.device)
                    estimate_q0.append(critic(init_sa).cpu())
                elif self.critic_type == 'distributional':
                    q = critic(o, a)
                    estimate_q0.append(q.cpu())
        estimate_q0 = torch.cat(estimate_q0, dim=0)

        return estimate_q0.mean().item()