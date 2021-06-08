import torch
import numpy as np
from abc import ABC
from typing import Union

from d3pe.utils.data import OPEDataset

class Policy(ABC):
    def forward(self, obs : torch.Tensor) -> torch.distributions.Distribution:
        raise NotImplementedError

    def get_action(self, obs : Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError

class Evaluator:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.is_initialized = False
    
    def initialize(self, train_dataset : OPEDataset, val_dataset : OPEDataset, *args, **kwargs) -> None:
        self.is_initialized = True

    def __call__(self, policy : Policy) -> float:
        assert self.is_initialized, "`initialize` should be called before calls."
        raise NotImplementedError

