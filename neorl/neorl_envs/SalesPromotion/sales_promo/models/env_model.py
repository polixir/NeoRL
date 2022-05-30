import torch
import torch.nn as nn
import numpy as np
from typing import List

from .torch.policy_disc import MultiDiscretePolicy, DiscretePolicy
from .torch.policy_cont import GaussianPolicy
from .torch.ensemble import EnsembleBaseModel
 

class VenvPolicy(nn.Module):
    """User emulator class
    """
    def __init__(self,
                 observation_a_dim: int=5,
                 observation_b_dim: int=5,
                 observation_c_dim: int=8,
                 discrete_action_b_num: int=6,
                 continuous_action_b_size: int=1,
                 hidden_size: tuple = (256, 128, 64)):
        super(VenvPolicy, self).__init__()
        self.dim = {'observation_a_dim': observation_a_dim, 'observation_b_dim': observation_b_dim, 'observation_c_dim': observation_c_dim}
        self.discrete_action_b_num = discrete_action_b_num

        # self.do_action_model = OrderPolicy(observation_c_dim, 2, 2, hidden_size)
        self.do_action_model = DiscretePolicy(observation_c_dim, 2, hidden_size)

        # order_num
        self.disc_order_policy = MultiDiscretePolicy(observation_a_dim-1, [discrete_action_b_num],
                                                     hidden_size, True)

        # avg_order_fee
        self.cont_order_policy = GaussianPolicy(observation_b_dim-1, continuous_action_b_size, hidden_size)

    def _cut_observation(self, observation):
        """split obs into obs_a, obs_b, obs_c
        """
        if len(observation.shape) == 2:
            return [observation[:, 1:self.dim['observation_a_dim']],
                observation[:, self.dim['observation_a_dim']+1:(self.dim['observation_a_dim'] + self.dim['observation_b_dim'])],
                observation[:, (self.dim['observation_a_dim'] + self.dim['observation_b_dim']):]]
        elif len(observation.shape) == 1:
            return [observation[1:self.dim['observation_a_dim']],
                observation[self.dim['observation_a_dim']+1:(self.dim['observation_a_dim'] + self.dim['observation_b_dim'])],
                observation[(self.dim['observation_a_dim'] + self.dim['observation_b_dim']):]]
        else:
            raise ValueError('The shape of the input is not supported!')

    def select_action(self, observation, eval=False):
        observation_a, observation_b, observation_c = self._cut_observation(observation)
        do_action = self.do_action_model.select_action(observation_c, eval)
   
        non_zero_disc_order_action = self.disc_order_policy.select_action(observation_a, eval)
        order_num_action = non_zero_disc_order_action * do_action
        non_zero_cont_order_action = self.cont_order_policy.select_action(observation_b, eval)
        order_fee_action = non_zero_cont_order_action * do_action

        return torch.cat([order_num_action, order_fee_action], 1)

    def get_log_prob(self, observation, action):
        order_num_action, order_fee_action = action.split(1, -1)
        observation_a, observation_b, observation_c = self._cut_observation(observation)
        do_action = (order_num_action > 0.0).long().squeeze(1)
        log_prob = self.do_action_model.get_log_prob(observation_c, do_action)

        order_num_action = order_num_action * torch.tensor(self.discrete_action_b_num, dtype=torch.float32,
                                                           device=observation.device)
        order_num_action = (order_num_action - 1) * do_action.unsqueeze(1)
        log_prob += self.disc_order_policy.get_log_prob(observation_a, order_num_action) * do_action.unsqueeze(1)

        log_prob += self.cont_order_policy.get_log_prob(observation_b, order_fee_action) * do_action.unsqueeze(1)

        return log_prob

    def entropy(self, observation):
        observation_a, observation_b, observation_c = self._cut_observation(observation)
        return self.do_action_model.entropy(observation_c) + self.disc_order_policy.entropy(observation_a) + \
               self.cont_order_policy.entropy(observation_b)


class EnsembleVenvModel(EnsembleBaseModel):
    def __init__(self, models: List[VenvPolicy]):
        super().__init__(models)
    
    def select_action(self, observation, eval=False):
        action_list = []
        for model in self.models:
            action_list.append(model.select_action(observation, eval).view(1, observation.shape[0],-1))
        # (batch, ensemble, action_size)
        ensemble_action = torch.cat(action_list, 0).transpose(0, 1)

        # get random pick one action
        random_index = np.random.randint(0, ensemble_action.shape[1], (ensemble_action.shape[0],))
        random_pick_action = ensemble_action[np.arange(0, ensemble_action.shape[0]), random_index]

        # caculate disagreement variance
        mean_action = torch.mean(ensemble_action, 1)
        diff = ensemble_action.transpose(0, 1) - mean_action
        disagreement_uncertainty = torch.max(torch.norm(diff, dim=-1, keepdim=True), dim=0)[0]

        return ensemble_action, random_pick_action, disagreement_uncertainty
    
    def get_log_prob(self, observation, action):
        log_prob_list = []
        for model in self.models:
            log_prob_list.append(model.get_log_prob(observation, action).view(1, observation.shape[0], -1))
        
        # (ensemble, batch, 1) -> (batch, ensemble, 1)
        return torch.cat(log_prob_list, 0).transpose(0, 1)
    
    def entropy(self, observation):
        entropy_list = []
        for model in self.models:
            entropy_list.append(model.entropy(observation).view(1, observation.shape[0], -1))
        ensemble_entropy = torch.cat(entropy_list, 0)
        return torch.mean(ensemble_entropy, 0)
