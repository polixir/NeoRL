import torch
import os
import numpy as np
from ..models.env_model import VenvPolicy, EnsembleVenvModel
from .sp_env import SalesPromotion_v0


def get_env_instance():
    return SalesPromotion_v0()
