import torch
import torch.nn as nn


class CustomSoftMax(nn.Module):
    def __init__(self, onehot_action_dim, sections):
        super(CustomSoftMax, self).__init__()
        self.onehot_action_dim = onehot_action_dim
        self.sections = sections

    def forward(self,
                input_tensor: torch.Tensor):
        out = torch.zeros(input_tensor.shape, dtype=torch.float32, device=input_tensor.device) 
        out[:, 0:self.onehot_action_dim] = torch.cat(
            [tensor.softmax(dim=-1) for tensor in
             torch.split(input_tensor[:, 0:self.onehot_action_dim], self.sections, dim=-1)],
            dim=-1)
        return out
