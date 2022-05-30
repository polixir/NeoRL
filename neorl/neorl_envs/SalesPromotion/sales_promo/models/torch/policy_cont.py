from typing import Type
import torch
import torch.nn as nn
from torch.distributions import Normal


LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0


class GaussianPolicy(nn.Module):
    """
    Gaussian policy for continuous action space
    """

    def __init__(self,
                 observation_dim: int,
                 action_dim: int,
                 # action_low: list,
                 hidden_size: tuple = (256, 128, 64),
                 activation_fn: Type[nn.Module] = nn.LeakyReLU(inplace=True)):
        super(GaussianPolicy, self).__init__()
        self.activation_fn = activation_fn
        self.action_dim = action_dim

        self.gaussian_net_model = nn.ModuleList()
        last_dim = observation_dim
        for size in hidden_size:
            self.gaussian_net_model.append(nn.Linear(last_dim, size))
            last_dim = size
        
        self.mean_layer = nn.Linear(last_dim, action_dim)
        self.log_std_layer = nn.Linear(last_dim, action_dim)
      

    def forward(self, x):
        for affine in self.gaussian_net_model:
            x = self.activation_fn(affine(x))
        mean = self.mean_layer(x)
        mean = torch.sigmoid(mean)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std, log_std

    def select_action(self, observation, eval=False):
        mean, std, _ = self.forward(observation)
        gaussian = Normal(mean, std)
        if not eval:
            action = gaussian.sample()
        else:
            action = mean
        for i in range(action.shape[1]):
            action[:, [i]] = torch.clamp(action[:, [i]], min=1e-10, max=1.0)
        return action

    def get_log_prob(self, observation, action):
        mean, std, _ = self.forward(observation)
        distribution = Normal(mean, std)
        log_prob = distribution.log_prob(action).sum(axis=-1, keepdim=True)
        log_prob = torch.clamp(log_prob, -16.0, 5.0)
        
        return log_prob

    def entropy(self, observation):
        mean, std, _ = self.forward(observation)
        distribution = Normal(mean, std)
        return distribution.entropy().sum(-1, keepdim=True)

    def avg_std(self, observation):
        _, std, _ = self.forward(observation)
        return std.mean().item()
