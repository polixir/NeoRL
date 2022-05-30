import torch
import torch.nn as nn
from typing import List


class EnsembleBaseModel(nn.Module):  # type: ignore
    _models: nn.ModuleList

    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_parameter('best_model_indices', torch.nn.Parameter(torch.zeros(len(self.models))))
