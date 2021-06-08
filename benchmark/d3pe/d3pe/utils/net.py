import torch
import numpy as np
from torch import nn
from typing import Optional, Tuple, Union

from d3pe.utils.func import soft_clamp

ACTIVATION_CREATORS = {
    'relu' : lambda dim: nn.ReLU(inplace=True),
    'elu' : lambda dim: nn.ELU(),
    'leakyrelu' : lambda dim: nn.LeakyReLU(negative_slope=0.1, inplace=True),
    'tanh' : lambda dim: nn.Tanh(),
    'sigmoid' : lambda dim: nn.Sigmoid(),
    'identity' : lambda dim: nn.Identity(),
    'prelu' : lambda dim: nn.PReLU(dim),
    'gelu' : lambda dim: nn.GELU(),
    'swish' : lambda dim: Swish(),
}

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MLP(nn.Module):
    r"""
        Multi-layer Perceptron
        Inputs:
            in_features : int, features numbers of the input
            out_features : int, features numbers of the output
            hidden_features : int, features numbers of the hidden layers
            hidden_layers : int, numbers of the hidden layers 
            norm : str, normalization method between hidden layers, default : None 
            hidden_activation : str, activation function used in hidden layers, default : 'leakyrelu' 
            output_activation : str, activation function used in output layer, default : 'identity' 
    """
    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 hidden_features : int, hidden_layers : int, 
                 norm : str = None, 
                 hidden_activation : str = 'leakyrelu', 
                 output_activation : str = 'identity'):
        super(MLP, self).__init__()

        hidden_activation_creator = ACTIVATION_CREATORS[hidden_activation]
        output_activation_creator = ACTIVATION_CREATORS[output_activation]

        if hidden_layers == 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, out_features),
                output_activation_creator(out_features)
            )
        else:
            net = []
            for i in range(hidden_layers):
                net.append(nn.Linear(in_features if i == 0 else hidden_features, hidden_features))
                if norm:
                    if norm == 'ln':
                        net.append(nn.LayerNorm(hidden_features))
                    elif norm == 'bn':
                        net.append(nn.BatchNorm1d(hidden_features))
                    else:
                        raise NotImplementedError(f'{norm} does not supported!')
                net.append(hidden_activation_creator(hidden_features))
            net.append(nn.Linear(hidden_features, out_features))
            net.append(output_activation_creator(out_features))
            self.net = nn.Sequential(*net)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        r"""forward method of MLP only assume the last dim of x matches `in_features`"""
        head_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        out = self.net(x)
        out = out.view(*head_shape, out.shape[-1])
        return out

class GaussianActor(torch.nn.Module):
    MAX_LOGSTD = 2
    MIN_LOGSTD = -5
    
    def __init__(self,
                 obs_dim : int,
                 action_dim : int,
                 features : int,
                 layers : int,
                 std : Optional[np.ndarray] = None) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.features = features
        self.layers = layers

        if std is not None:
            self.register_buffer('std', torch.as_tensor(std).float())
            self.backbone = MLP(obs_dim, action_dim, features, layers)
        else:
            self.std = None
            self.backbone = MLP(obs_dim, 2 * action_dim, features, layers)

    def forward(self, obs : torch.Tensor) -> torch.distributions.Distribution:
        output = self.backbone(obs)
        if self.std is not None:
            mu = output
            std = self.std
        else:
            mu, log_std = torch.chunk(output, 2, dim=-1)
            log_std = soft_clamp(log_std, self.MIN_LOGSTD, self.MAX_LOGSTD)
            std = torch.exp(log_std)
        return torch.distributions.Normal(mu, std)

class TransformedDistribution(torch.distributions.TransformedDistribution):
    @property
    def mean(self) -> torch.Tensor:
        x = self.base_dist.mean
        for transform in self.transforms:
            x = transform(x)
        return x

class TanhGaussianActor(torch.nn.Module):
    # Make std greater than 1 will increase the logprob on contrary direction.
    # See more in https://www.desmos.com/calculator/wweev47yyp
    MAX_LOGSTD = 0
    MIN_LOGSTD = -5

    def __init__(self,
                 obs_dim : int,
                 action_dim : int,
                 features : int,
                 layers : int,
                 min_actions : np.ndarray,
                 max_actions : np.ndarray,) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.features = features
        self.layers = layers

        self.backbone = MLP(obs_dim, 2 * action_dim, features, layers)

        self.register_buffer('action_mean', torch.nn.Parameter(torch.as_tensor((max_actions + min_actions) / 2, dtype=torch.float32), requires_grad=False))
        self.register_buffer('action_scale', torch.nn.Parameter(torch.as_tensor((max_actions - min_actions) / 2, dtype=torch.float32), requires_grad=False))

    def forward(self, obs : torch.Tensor) -> torch.distributions.Distribution:
        output = self.backbone(obs)
        mu, log_std = torch.chunk(output, 2, dim=-1)
        log_std = self.MIN_LOGSTD + (self.MAX_LOGSTD - self.MIN_LOGSTD) / 2 * (torch.tanh(log_std) + 1)
        std = torch.exp(log_std)

        return TransformedDistribution(
            torch.distributions.Normal(mu, std),
            torch.distributions.transforms.ComposeTransform([
                torch.distributions.transforms.TanhTransform(),
                torch.distributions.transforms.AffineTransform(self.action_mean, self.action_scale)
            ]))

class DistributionalCritic(torch.nn.Module):
    def __init__(self, 
                 obs_dim : int, 
                 action_dim : int, 
                 features : int, 
                 layers : int, 
                 min_value : int, 
                 max_value : int,
                 atoms : int = 51) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.atoms = atoms

        self.net = MLP(obs_dim + action_dim, atoms, features, layers)

        self.register_buffer('z', torch.linspace(min_value, max_value, atoms))
        self.delta_z = (max_value - min_value) / (atoms - 1)

    def forward(self, obs : torch.Tensor, 
                      action : torch.Tensor, 
                      with_p : bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        obs_action = torch.cat([obs, action], dim=-1)
        logits = self.net(obs_action)
        p = torch.softmax(logits, dim=-1)
        q = torch.sum(p * self.z, dim=-1, keepdim=True)
        if with_p:
            return q, p
        else:
            return q

    @torch.no_grad()
    def get_target(self, obs : torch.Tensor, action : torch.Tensor, reward : torch.Tensor, discount : float):
        p = self(obs, action) # [*B, N]

        # shift the atoms by reward
        target_z = reward + discount * self.z # [*B, N]
        target_z = torch.clamp(target_z, self.min_value, self.max_value) # [*B, N]

        # reproject the value to the nearby atoms
        target_z = target_z.unsqueeze(dim=-1) # [*B, N, 1]
        distance = torch.abs(target_z - self.z) # [*B, N, N]
        ratio = torch.clamp(1 - distance / self.delta_z, 0, 1) # [*B, N, N]
        target_p = torch.sum(p.unsqueeze(dim=-1) * ratio, dim=-2) # [*B, N]

        return target_p