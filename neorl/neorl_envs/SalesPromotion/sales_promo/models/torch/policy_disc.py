import torch
import torch.nn as nn
from typing import Type
from torch.distributions import Categorical
from .common import CustomSoftMax


class DiscretePolicy(nn.Module):
    """
    Common policy one dimension discrete action
    """
    def __init__(self,
                 observation_dim: int,
                 action_size: int,
                 hidden_size: tuple = (256, 128, 64),
                 activation_fn: Type[nn.Module] = nn.LeakyReLU(inplace=True)):
        super(DiscretePolicy, self).__init__()
        self.activation_fn = activation_fn

        self.discrete_policy_model = nn.ModuleList()
        last_dim = observation_dim
        for size in hidden_size:
            self.discrete_policy_model.append(nn.Linear(last_dim, size))
            last_dim = size
        self.output_layer = nn.Sequential(
            nn.Linear(last_dim, action_size),
            nn.Softmax(-1)
        )

    def forward(self, x):
        for affine in self.discrete_policy_model:
            x = self.activation_fn(affine(x))
        probs = self.output_layer(x)
        return Categorical(probs)

    def select_action(self, x, eval=False):
        dist = self.forward(x)
        if not eval:
            action = dist.sample().unsqueeze(-1)
        else:
            action = dist.probs.argmax(1, keepdim=True)
        return action

    def get_log_prob(self, x, actions):
        dist = self.forward(x)
        log_prob =dist.log_prob(actions).unsqueeze(-1)
        log_prob = torch.clamp(log_prob, -20.0, 0.0)
        return log_prob

    def entropy(self, x):
        dist = self.forward(x)
        return dist.entropy().unsqueeze(1)


class MultiDiscretePolicy(nn.Module):
    """
    Common policy for multi-dimension discrete action
    """

    def __init__(self,
                 observation_dim: int,
                 action_num: list,
                 hidden_size: tuple = (256, 128, 64),
                 exclude_zero: bool = False,
                 activation_fn: Type[nn.Module] = nn.LeakyReLU(inplace=True)):
        super(MultiDiscretePolicy, self).__init__()
        self.activation_fn = activation_fn
        self.output_dim = len(action_num)
        self.discrete_action_num = action_num
        self.exclude_zero = exclude_zero

        self.discrete_policy_model = nn.ModuleList()
        last_dim = observation_dim
        for size in hidden_size:
            self.discrete_policy_model.append(nn.Linear(last_dim, size))
            last_dim = size
        self.output_layer = nn.Linear(last_dim, sum(action_num))
        self.custom_softmax = CustomSoftMax(sum(action_num), action_num)

    def forward(self, x):
        for affine in self.discrete_policy_model:
            x = self.activation_fn(affine(x))
        x = self.output_layer(x)
        return self.custom_softmax(x)

    def select_action(self, observation, greedy=False):
        action_probs = self.forward(observation)
        return self.sample_action_by_prob(action_probs, observation.shape[0], greedy)

    def get_log_prob(self, observation, action):
        action_probs = self.forward(observation)
        return self.calculate_log_prob(action_probs, observation.shape[0], action)

    def entropy(self, observation):
        action_probs = self.forward(observation)
        total_entropy = self.calculate_entropy(action_probs, observation.shape[0])
        return total_entropy

    def sample_action_by_prob(self, action_probs, num, eval=False):
        action = torch.empty([num, self.output_dim], dtype=torch.float32, device=action_probs.device) 
        for i, num in enumerate(self.discrete_action_num):
            action_prob = action_probs[:,
                          sum(self.discrete_action_num[:i]):sum(self.discrete_action_num[:i + 1])]
            if not eval:
                tmp = action_prob.multinomial(1)
            else:
                tmp = action_prob.argmax(1, keepdim=True)
            if not self.exclude_zero:
                action[:, i] = tmp.squeeze(1).float() / torch.tensor(num - 1, dtype=torch.float32, device=action_probs.device)
            else:
                action[:, i] = (tmp.squeeze(1) + 1).float() / torch.tensor(num, dtype=torch.float32, device=action_probs.device)
        return action

    def calculate_log_prob(self, action_probs, num, action):
        log_prob = torch.zeros((num, 1), dtype=torch.float32, device=action_probs.device)
        for i, num in enumerate(self.discrete_action_num):
            coupon_prob = action_probs[:, sum(self.discrete_action_num[:i]):
                                          sum(self.discrete_action_num[:i + 1])]
            if not self.exclude_zero:
                log_prob += \
                    torch.clamp(torch.log(coupon_prob.gather(1, (action[:, i] * torch.tensor(num - 1, dtype=torch.float32, device=action_probs.device)).long().unsqueeze(1))), -20.0, 0.0)
            else:
                log_prob = torch.clamp(torch.log(coupon_prob.gather(1, (action[:, i]).long().unsqueeze(1))), -20.0, 0.0)
        
        return log_prob

    def calculate_entropy(self, action_probs, num):
        total_entropy = torch.zeros((num, 1), dtype=torch.float32, device=action_probs.device)
        for i, num in enumerate(self.discrete_action_num):
            d = Categorical(action_probs[:, sum(self.discrete_action_num[:i]):
                                            sum(self.discrete_action_num[:i + 1])])
            total_entropy += d.entropy().unsqueeze(1)
        return total_entropy
