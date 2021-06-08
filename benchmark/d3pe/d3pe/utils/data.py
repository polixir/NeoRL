import gym
import torch
import numpy as np

from typing import *

class OPEDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data : Dict[str, np.ndarray], 
                 start_indexes : Optional[np.ndarray] = None,
                 obs_space : Optional[gym.Space] = None,
                 action_space : Optional[gym.Space] = None,) -> None:
        super().__init__()
        self.data = data
        self.total_size = self.data['obs'].shape[0]
        self.start_indexes = start_indexes
        if self.has_trajectory:
            self.trajectory_number = self.start_indexes.shape[0]
            self.end_indexes = np.concatenate([self.start_indexes[1:], np.array([self.total_size])])

        self.obs_space = obs_space
        self.action_space = action_space
        self.data['obs'] = np.clip(self.data['obs'], *self.get_obs_boundary())
        self.data['next_obs'] = np.clip(self.data['next_obs'], *self.get_obs_boundary())
        self.data['action'] = np.clip(self.data['action'], *self.get_action_boundary())
    
    @property
    def has_trajectory(self) -> bool:
        return self.start_indexes is not None

    def get_trajectory(self,) -> List[Dict[str, np.ndarray]]:
        assert self.has_trajectory
        return [{k : v[start : end] for k, v in self.data.items()} for start, end in zip(self.start_indexes, self.end_indexes)]

    def get_initial_states(self,) -> Dict[str, np.ndarray]:
        assert self.has_trajectory
        return {k : v[self.start_indexes] for k, v in self.data.items()}

    def sample(self, batch_size) -> Dict[str, np.ndarray]:
        indexes = np.random.randint(0, self.total_size, size=(batch_size))
        return {k : v[indexes] for k, v in self.data.items()}

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        return {k : v[index] for k, v in self.data.items()}

    def get_reward_boundary(self) -> Tuple[float, float]:
        min_reward = self.data['reward'].min()
        max_reward = self.data['reward'].max()
        return min_reward, max_reward

    def get_value_boundary(self, gamma : float, enlarge_ratio : float = 0.2) -> Tuple[float, float]:
        min_reward, max_reward = self.get_reward_boundary()
        min_value = (min_reward - enlarge_ratio * (max_reward - min_reward)) / (1 - gamma)
        max_value = (max_reward + enlarge_ratio * (max_reward - min_reward)) / (1 - gamma)
        return min_value, max_value

    def get_action_boundary(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.action_space is None:
            return self.data['action'].min(axis=0), self.data['action'].max(axis=0)
        else:
            return self.action_space.low, self.action_space.high

    def get_obs_boundary(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.obs_space is None:
            return self.data['obs'].min(axis=0), self.data['obs'].max(axis=0)
        else:
            return self.obs_space.low, self.obs_space.high

def get_neorl_datasets(task : str, level : str, amount : int) -> Tuple[OPEDataset, OPEDataset]:
    import neorl
    env = neorl.make(task)
    train_data, val_data = env.get_dataset(data_type=level, train_num=amount)
    train_start_indexes = train_data.pop('index')
    val_start_indexes = val_data.pop('index')
    return (
        OPEDataset(train_data, train_start_indexes, obs_space=env.observation_space, action_space=env.action_space), 
        OPEDataset(val_data, val_start_indexes, obs_space=env.observation_space, action_space=env.action_space)
    )

def to_torch(data : dict, dtype = torch.float32, device = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, torch.Tensor]:
    return {k : torch.as_tensor(v, dtype=dtype, device=device) for k, v in data.items()}

def to_numpy(data : dict) -> Dict[str, np.ndarray]:
    return {k : v if isinstance(v, np.ndarray) else v.detach().cpu().numpy() for k, v in data.items()}