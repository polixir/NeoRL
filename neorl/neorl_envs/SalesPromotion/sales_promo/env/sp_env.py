import torch
import os
import numpy as np
from ..models.env_model import VenvPolicy, EnsembleVenvModel
from .marketing import MarketingEnv
from gym.utils.seeding import np_random
from gym.spaces import Box, MultiDiscrete
from neorl import core
from neorl.utils import get_json, sample_dataset, LOCAL_JSON_FILE_PATH, DATA_PATH


def get_market_env():
    dir, _ = os.path.split(os.path.abspath(__file__))
    # device = torch.device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_list = [VenvPolicy() for _ in range(5)]
    user_model = EnsembleVenvModel(model_list)
    user_model.load_state_dict(torch.load(os.path.join(dir, 'user_model.pt'), map_location=device))
    user_model.to(device)

    val_initial_states = np.load(os.path.join(dir, f'test_initial_states_10000_people.npy')) # init internal stats for marking env
    
    return MarketingEnv(val_initial_states, user_model, device)


class SalesPromotion_v0(core.EnvData):
    """
    NeoRL environment wrapper for continuous action space version of sales promotion env.
    Indeed this is a bacth/vectorized environment with 10,000 users 
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.inner_env = get_market_env()  # get the market model
        self.num_users = self.inner_env.val_initial_states.shape[0]
        # self.action_space = Box(low=np.tile([0,0.6],(self.num_users, 1)), high=np.tile([6,1.0], (self.num_users, 1)), shape=(self.num_users, 2,), dtype=np.float32)
        self.action_space = Box(low=np.array([0.0, 0.6]), high=np.array([5., 1.0]), shape=(2,), dtype=np.float32)
        # the state contains (total orders, avg orders over accmulated days, avg fee over accumulated days, week of the day) 
        self._scale = np.array([200.0, 10.0, 100.0, 6.0])
        # self.observation_space = Box(low=0, high=1, shape=(self.num_users, 4,), dtype=np.float32)
        self.observation_space = Box(low=np.array([0,0,0,0]), high=np.array([200.0, 10.0, 100.0, 6.0]), shape=(4,), dtype=np.float32)
        self.accumulated_days = 60
        self.week_of_day = 1  # the init day is Tuesday
        self.obs = None
        
    def _get_next_state(self, states, day_order_num, day_average_fee, coupon_num, coupon_discount, next_states):
        # If you define day_order_num to be continuous instead of discrete/category, apply round function here.
        day_order_num = day_order_num.clip(0, 6).round()
        day_average_fee = day_average_fee.clip(0.0, 100.0)
        # Rules on the user action: if either action is 0 (order num or fee), the other action should also be 0.
        day_order_num[day_average_fee <= 0.0] = 0
        day_average_fee[day_order_num <= 0] = 0.0
        self.accumulated_days += 1
        self.week_of_day = (self.week_of_day + 1) % 7
        # Compute next state
        next_states[..., 0] = states[..., 0] + day_order_num # Total num
        next_states[..., 1] = states[..., 1] + 1 / (self.accumulated_days) * (day_order_num - states[..., 1]) # Average order num
        next_states[..., 2] = states[..., 2] + 1 / (self.accumulated_days) * (day_average_fee - states[..., 2]) # Average order fee across days
        next_states[...,3] = self.week_of_day
        return next_states

    def _get_next_state_numpy(self, states, day_order_num, day_average_fee, coupon_num, coupon_discount):
        """Will be referenced in data_preprocess.py, virtual_env.py"""
        with np.errstate(invalid="ignore", divide="ignore"): # Ignore nan and inf result from division by 0
            return self._get_next_state(states, day_order_num, day_average_fee, coupon_num, coupon_discount, np.empty(states.shape))

    def _get_next_state_torch(self, states, user_action, coupon_action):
        """Will be referenced in venv.py specified from venv.yaml"""
        day_order_num, day_average_fee = user_action[..., 0], user_action[..., 1]
        if coupon_action is not None:
            coupon_num, coupon_discount = coupon_action[..., 0], coupon_action[..., 1]
        else:
            coupon_num, coupon_discount = None, None
        return self.get_next_state(states, day_order_num, day_average_fee, coupon_num, coupon_discount, states.new_empty(states.shape))
    
    def step(self, action):
        """
        Step func receives a batch input actions (10,000 users), and returns with batch obs, 
        a scalar reward (can be viewed as the same for all users in a day), a single terminal signal and infos.

        Args:
            action (_type_): Batch or single action for all the users. If single, it will be handled as a batch action 
            where each single action is exactly equal to the single action input.

        Returns:
            obs: batch form
            rew: single form
            done: single form
            info: a null dict
        """
        if len(np.array(action).shape) == 1:
            action = np.array([action])
        if (action[:,0].min() >=-1 and action[:,0].max() < 1+1e-6) and ((1e-6 < action[:,0]) & (action[:,0] < 1.0)).sum()>0:  
            # Action is normed to (-1,1] or [0,1]. However. Recover it by multiplying the max action
            # However, action<0 will be clipped to 0d.
            action[:,0] *= self.action_space.high[0]
        action[:,0] = action[:,0].round()  # round the number of discount coupons
        d_total_cost, d_total_gmv, d_user_actions = self.inner_env.step(action)
        next_obs = self._get_next_state_numpy(self.obs, day_order_num=d_user_actions[:,0], day_average_fee=d_user_actions[:,1],
                                              coupon_num=action[:,0], coupon_discount=action[:,1])
        self.obs = next_obs
        rew = (d_total_gmv - d_total_cost)/self.num_users
        if self.accumulated_days < self.inner_env.validation_length + 60:
            done = False
        else:
            done = True
        return next_obs, rew, done, {}
    
    def reset(self):
        self.inner_env.reset()
        dir, _ = os.path.split(os.path.abspath(__file__))
        self.obs = np.load(os.path.join(dir,'evaluation_init_obs.npy'))
        self.accumulated_days = 60  # reset to the 61th day
        self.week_of_day = 1  # the init day is Tuesday
        return self.obs
        
    def seed(self, seed_number):
        return np_random(seed_number)[0]
    
    @property
    def validation_length(self,):
        return self.inner_env.validation_length
    
    def get_dataset(self, task_name_version: str = None, data_type: str = "human", train_num: int = 10000,
                    need_val: bool = True, val_ratio: float = 0.1, path: str = DATA_PATH, use_data_reward: bool = True):
        """
        Get dataset from given env. However, SalesPromotion env only has one training dataset and one validation dataset, where 
        the actions are from human or a learned promotion policy model.

        :param task_name_version: The name and version (if applicable) of the task,
            default is the same as `task` while making env
        :param data_type: Which type of policy is used to collect data. It should
            be one of ["high", "medium", "low"], default to `high`
        :param train_num: The num of trajectory of training data. Note that the num
            should be less than 10,000, default to `100`
        :param need_val: Whether needs to download validation data, default to `True`
        :param val_ratio: The ratio of validation data to training data, default to `0.1`
        :param path: The directory of data to load from or download to `./data/`
        :param use_data_reward: Whether uses default data reward. If false, a customized
            reward function should be provided by users while making env

        :return train_samples, val_samples
        """

        if data_type.lower() == "human" or "model" or 'low' or 'medium' or 'high':
            data_type = 'v0'
        else:
            raise Exception(f"Please check `data_type`, {data_type} is not supported!")

        # task_name_version = self._name if task_name_version is None else task_name_version
        task_name_version = 'sp'

        data_json = get_json(LOCAL_JSON_FILE_PATH)

        train_samples = sample_dataset(task_name_version, path, train_num, data_json, data_type, use_data_reward,
                                       self._reward_func, "train")
        train_samples['action'] /=self.action_space.high[0]  # to make the action in [-1,1]. It will be better to revise the output of the policy network
        val_samples = None
        if need_val:
            val_samples = sample_dataset(task_name_version, path, int(train_num * val_ratio), data_json, data_type,
                                         use_data_reward, self._reward_func, "val")
            val_samples['action'] /=self.action_space.high[0]
        return train_samples, val_samples