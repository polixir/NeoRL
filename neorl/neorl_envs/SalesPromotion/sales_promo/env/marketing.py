import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from tqdm import tqdm
from gym import Env
from gym.utils.seeding import np_random
from gym.spaces import Box, MultiDiscrete
from neorl import core


class MarketingEnv(core.EnvData):
    """
    Description:
        The virtual environment class for the competition validation: input the current user states and the platform action(given from contestant),
        and return the corresponding total reward from all the users and the user actions.

    User State:
        Type: Box(18, low=0.0, high=1.0), except the first dim and the sixth dim, they may be larger than 1 in some cases.

    Platform Action:
        Type: MultiDiscrete([6, 8])
            num_1: the number of the coupon, discrete, [0, 1, 2, 3, 4, 5]
            num_2: the number of diffenent type of discount coupon, discrete, [0, 1, 2, 3, 4, 5, 6, 7], 0 correspond to 0.95-discount-coupon,
            1 correspond to 0.90-discount-coupon, etc.
        Note the expiry date of all the coupons is 1 days.

    Combined with the user neural network model and some basic rules, we can construct the virtual marketing environment.
    User neural network get the input : [User State, Platform Action], and output the user action.
    User Action:
        dim_1: the number of the order
        dim_2: the average order fee of the user
    And we use some background rules, we can caculate the following variable:
        v_1: the number of the used coupons
        v_2: the average discount of the used coupons
        v_3: the worst used coupon today
    After get all the variables, we can caculate the next user state.

    Validation State:
        The initial validation state is from a offline dataset

    Reward function:
        GMV = dim_1 * dim_2 - (1 - v_2) * v_1 * dim_2
        ROI = GMV / (1 - v_2) * v_1 * dim_2
    """
    MAX_DELIVER_NUMBER = 5 # Maximum number of distributed coupons in a single day
    OBS_SIZE, PLATFORM_ACTION_SIZE, USER_ACTION_SIZE = 18, 2, 5
    MAX_ENV_STEP = 60 # The maximum allowable trajectory length of the environment
    ENV_DISCOUNT = 0.99
    EFFECTIVE_DAY = 1 # The coupon is valid for one day
    DISCOUNT_COUPON_LIST = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60] # The option of discount range
    # A preset maximum value for each dimension, used to standardize data
    DIM_MAX_DICT = {'total_num': 36, 'avg_num': 4, 'std_num': 24, 'min_num': 4, 'max_num': 12,
                    'total_fee': 400, 'avg_fee': 60, 'std_fee': 1600, 'min_fee': 50, 'max_fee': 120,
                    'active_time': 45, 'discount_order_num': 16, 'discount_order_day': 12, 'discount_total_fee': 400, 'history_accept_min_discount':1 - DISCOUNT_COUPON_LIST[-1],
                    'week_day_index': 6, 'coupon_num_now': MAX_DELIVER_NUMBER, 'coupon_discount_now': 1 - DISCOUNT_COUPON_LIST[-1],
                    'day_coupon_num': MAX_DELIVER_NUMBER, 'day_avg_discount_ratio': 1,
                    'day_order_num': 6, 'day_avg_fee': 100, 'day_coupon_used_num': 6, 'avg_discount_ratio': DISCOUNT_COUPON_LIST[0],
                    'day_used_min_discount': 1 - DISCOUNT_COUPON_LIST[-1]} # 0.40

    def __init__(self,
                 val_offline_data_obs: np.ndarray,
                 user_policy_net: nn.Module,
                 device: torch.device,
                 seed_number: int = None):
        """
         Args:
            val_offline_data_obs: The virtual user initial state set used to validate the policy
            user_policy_net: The user policy model that forms part of the platform environment
            device: Models' device
            seed_number: Random seed
        """
        if seed_number is not None:
            torch.manual_seed(seed_number)
            torch.cuda.manual_seed_all(seed_number)
            np.random.seed(seed_number)
            self.rng = self.seed(seed_number)
        self.val_initial_states = val_offline_data_obs
        self.user_policy_net = user_policy_net
        self.state_scaler = list(MarketingEnv.DIM_MAX_DICT.values())[0:MarketingEnv.OBS_SIZE]
        self.current_env_step_list = None
        self.states = None
        self.done_list = None
        self.device = device
        self.effective_coupons_list = None
        self._set_action_space(num_list=[MarketingEnv.MAX_DELIVER_NUMBER+1, len(MarketingEnv.DISCOUNT_COUPON_LIST)])
        self._set_observation_space(MarketingEnv.OBS_SIZE)

    def seed(self, seed_number):
        return np_random(seed_number)[0]

    def _set_action_space(self, num_list=None):
        self.action_space = MultiDiscrete(num_list)

    def _set_observation_space(self, obs_size, low=0, high=1):
        self.observation_space = Box(low=low, high=high, shape=(obs_size,), dtype=np.float32)

    def _get_coupon_info(self, index):
        cur_total_num = len(self.effective_coupons_list[index])
        total_discount = 0
        for discount, _ in self.effective_coupons_list[index]:
            total_discount += discount
        cur_avg_discount = 1 - total_discount / cur_total_num if cur_total_num > 0 else 0.0

        return cur_total_num, cur_avg_discount

    def _update_user_state(self, index):
        """Update the current user state according to the current effective coupons
        """
        cur_total_num, cur_avg_discount = self._get_coupon_info(index)
        self.states[index][-1] = cur_avg_discount / MarketingEnv.DIM_MAX_DICT['coupon_discount_now']
        self.states[index][-2] = np.clip(cur_total_num, 0, MarketingEnv.DIM_MAX_DICT['coupon_num_now']) / MarketingEnv.DIM_MAX_DICT['coupon_num_now']

    def step(self, action):
        assert self.states is not None
        total_cost, total_gmv, user_action_list = 0, 0, []
        # order_num_array, order_fee_array = np.empty((self.states.shape[0], 1)), np.empty((self.states.shape[0], 1))
        num_users = self.states.shape[0]
        action_shape = np.shape(action)
        same_action_for_all = False
        if len(action_shape) == 2 and action_shape[0] == num_users:
            # batch actions
            num, discount = np.clip(action[:,0], 0, MarketingEnv.MAX_DELIVER_NUMBER), np.clip(action[:,1], MarketingEnv.DISCOUNT_COUPON_LIST[-1], MarketingEnv.DISCOUNT_COUPON_LIST[0])
        elif (len(action_shape)==1 and action_shape[0]== 2) or (len(action_shape)==2 and action_shape[0]==1):
            # the same single action for all users
            same_action_for_all = True
            if len(action_shape) == 2:
                action = action[0]
            num, discount = np.clip(action[0], 0, MarketingEnv.MAX_DELIVER_NUMBER), np.clip(action[1], MarketingEnv.DISCOUNT_COUPON_LIST[-1], MarketingEnv.DISCOUNT_COUPON_LIST[0])
        else:
            raise('Not supported action types.')
        print("coupon num max and min: ", np.max(num), np.min(num))
        for index in range(num_users):
            # 1. Insert the newly distributed coupon
            if same_action_for_all:
                new_coupons = [(discount, MarketingEnv.EFFECTIVE_DAY) for _ in range(round(num))]
            else:
                new_coupons = [(discount[index], MarketingEnv.EFFECTIVE_DAY) for _ in range(round(num[index]))]
            self.effective_coupons_list[index].extend(new_coupons)
            if len(self.effective_coupons_list[index]) > 0:
                self.effective_coupons_list[index].sort(key=lambda x: (-x[0], -x[1]))

            # 2. Update the user state and calculate the user action
            self._update_user_state(index)
        
        batch_real_state_value = self.states * self.state_scaler

        user_input = torch.from_numpy(self.states).float()
                
        # 2.1 Network output user action
        with torch.no_grad():
            result = self.user_policy_net.select_action(user_input.to(self.device), eval=False)[1]  # the raw output is (ensembled_action_prob, action_sampled_from_ensemble_prob, disagreement_uncertainty)
            batch_user_action = result.cpu().numpy() # order_num, order_fee
        # print("action shape: ", batch_user_action.shape, batch_user_action[0:5, :])
        for index in range(num_users):
            total_num_h, avg_num_h, std_num_h, min_num_h, max_num_h, total_fee_h, avg_fee_h, std_fee_h, min_fee_h, \
                max_fee_h, active_time_h, discount_order_num_h, discount_order_day_h, discount_total_fee_h, history_accept_min_discount_h, \
                    week_index, coupon_num_now, coupon_discount_now = batch_real_state_value[index]

            user_action = batch_user_action[index]
            day_order_num, day_avg_fee = round(user_action[0] * MarketingEnv.DIM_MAX_DICT['day_order_num']), user_action[1] * MarketingEnv.DIM_MAX_DICT['day_avg_fee']

            """The environment needs to be filtered through a set of rules found in real-world testing to make it more robust and realistic
            """
            if (week_index == 4 or week_index == 0): # Monday or Friday
                # In actual tests, Monday and Friday orders were about 25% higher than usual
                day_order_num += torch.bernoulli(torch.tensor(0.25)).item()
            elif week_index == 5 and day_order_num >= 1: # Saturday
                # In actual tests, Saturday orders showed a drop, about 25 percent below normal
                day_order_num -= torch.bernoulli(torch.tensor(0.25)).item()
            elif week_index == 6 and day_order_num >= 1: # Sunday
                # In actual tests, orders on Sundays were a steep drop, about 40 percent below normal
                day_order_num -= torch.bernoulli(torch.tensor(0.4)).item()
            else:
                pass

            # rule 2: Based on coupons and users' psychological expectations, personalized correction of daily orders
            if coupon_num_now > 0 and day_order_num == 0 and coupon_discount_now < history_accept_min_discount_h:
                # The coupon did not meet the user's expectation and the user took no order
                day_order_num += torch.bernoulli(torch.tensor(coupon_discount_now)).item()
            elif coupon_num_now > 0 and day_order_num == 0 and coupon_discount_now >= history_accept_min_discount_h:
                # The coupon met the user's expectation and the user took no order
                day_order_num = 1
            elif coupon_num_now > 0 and day_order_num >= 1 and coupon_discount_now < history_accept_min_discount_h:
                # The coupon did not meet the user's expectations, but the user still took a taxi
                day_order_num -= torch.bernoulli(torch.tensor(1 - coupon_discount_now)).item()
            elif coupon_num_now == 0 and day_order_num >= 1:
                # No delivery and high user expectations (at least 15% discount)
                if history_accept_min_discount_h >= 0.15:
                    day_order_num -= round(0.75 * day_order_num)
                # No delivery and low user expectations
                else:
                    day_order_num -= torch.bernoulli(torch.tensor(0.25)).item()

            # rule 3: To improve the effect of coupon 60、65、70、75, We increase
            # the number of orders based discount of the coupon
            if coupon_discount_now >= 0.25 and coupon_discount_now < 0.35:
                day_order_num += torch.bernoulli(torch.tensor(coupon_discount_now)).item()
            elif coupon_discount_now >= 0.35:
                day_order_num += torch.bernoulli(torch.tensor(coupon_discount_now*1.5)).item()

            # rule 4: clip average order fee
            if day_avg_fee > 0:
                day_avg_fee = np.clip(day_avg_fee, a_min=max(min_fee_h - (std_fee_h**0.5), 0), a_max=max(max_fee_h, avg_fee_h+3*(max_fee_h**0.5)))

            # rule 5: The symbols of day_order_num and day_avg_fee are unified
            day_order_num = round(day_order_num)
            if day_order_num >= 1.0 and day_avg_fee <= 0.1:
                day_avg_fee = (avg_fee_h + (torch.randn(1) * 1.5).item())
                day_order_num = float(day_avg_fee > 0.1) * day_order_num
                day_avg_fee = float(day_avg_fee > 0.1) * day_avg_fee
            elif day_order_num == 0:
                day_avg_fee = 0.0

            # 2.2 Calculate other part of the use action by rule
            day_coupon_used_num = min(day_order_num, len(self.effective_coupons_list[index]))
            if day_coupon_used_num > 0:
                used_coupons = self.effective_coupons_list[index][-round(day_coupon_used_num):]
                day_used_min_discount, discount_list = 1, []
                for (d, _) in used_coupons:
                    day_used_min_discount = min(day_used_min_discount, 1 - d)
                    discount_list.append(d)
                avg_discount_ratio = np.mean(discount_list)
            else:
                # means not using any coupons
                avg_discount_ratio = MarketingEnv.DISCOUNT_COUPON_LIST[0]
                day_used_min_discount = 1 - MarketingEnv.DISCOUNT_COUPON_LIST[-1]

            # 3. Reduce the validity of the remaining coupons by 1
            new_list = []
            for item in self.effective_coupons_list[index][0:len(self.effective_coupons_list[index])-round(day_coupon_used_num)]:
                if item[1] - 1 > 0:
                    new_list.append((item[0], item[1]-1))
            self.effective_coupons_list[index] = new_list
            assert len(self.effective_coupons_list[index]) == 0

            # caculate the next user state by all the variables
            total_num = total_num_h + day_order_num
            size = round(total_num_h / (avg_num_h + 1e-10) * (avg_num_h > 0))
            avg_num = avg_num_h + 1 / (size + 1) * (day_order_num - avg_num_h) * (day_order_num > 0)
            std_num = std_num_h + (day_order_num - avg_num) * (day_order_num - avg_num_h) * (day_order_num > 0)
            min_num = (min_num_h == 0) * day_order_num + (day_order_num == 0) * min_num_h + \
                    ((min_num_h > 0) & (day_order_num > 0)) * min(min_num_h, day_order_num)
            max_num = max(max_num_h, day_order_num)

            total_fee = total_fee_h + day_avg_fee
            size = round(total_fee_h / (avg_fee_h + 1e-10) * (avg_fee_h > 0))
            avg_fee = avg_fee_h + 1 / (size + 1) * (day_avg_fee - avg_fee_h) * (day_avg_fee > 0)
            std_fee = std_fee_h + (day_avg_fee - avg_fee) * (day_avg_fee - avg_fee_h) * (day_avg_fee > 0)
            min_fee = (min_fee_h == 0) * day_avg_fee + (day_avg_fee == 0) * min_fee_h + \
                    ((min_fee_h > 0) & (day_avg_fee > 0)) * min(day_avg_fee, min_fee_h)
            max_fee = max(max_fee_h, day_avg_fee)

            active_time = (day_order_num < 1) * (active_time_h + 1)
            discount_order_num = discount_order_num_h * MarketingEnv.ENV_DISCOUNT + day_order_num
            discount_order_day = discount_order_day_h * MarketingEnv.ENV_DISCOUNT + (day_order_num > 0)
            day_fee = day_order_num * day_avg_fee
            discount_total_fee = discount_total_fee_h * MarketingEnv.ENV_DISCOUNT + day_fee
            week_index = (week_index + 1) % 7
            update_flag = history_accept_min_discount_h > day_used_min_discount
            history_accept_min_discount = update_flag * day_used_min_discount + \
                (~update_flag) * history_accept_min_discount_h
            coupon_num_now, coupon_discount_now = self._get_coupon_info(index)

            self.states[index] = np.array([total_num, avg_num, std_num, min_num, max_num, total_fee, avg_fee, std_fee, min_fee, max_fee,
                active_time, discount_order_num, discount_order_day, discount_total_fee, history_accept_min_discount,
                week_index, coupon_num_now, coupon_discount_now])

            self.states[index] = self.states[index] / self.state_scaler
            self.states[index][1:5] = np.clip(self.states[index][1:5], 0, 1)
            self.states[index][6:] = np.clip(self.states[index][6:], 0, 1)

            self.done_list[index] = ((self.current_env_step_list[index] + 1) == MarketingEnv.MAX_ENV_STEP)
            self.current_env_step_list[index] += 1
            per_cost = (1 - avg_discount_ratio) * day_coupon_used_num * day_avg_fee
            per_gmv = day_avg_fee * day_order_num - (1 - avg_discount_ratio) * day_coupon_used_num * day_avg_fee
            user_action_list.append(np.array([round(day_order_num), day_avg_fee]))
            total_gmv += per_gmv
            total_cost += per_cost

        return total_cost, total_gmv, np.array(user_action_list)
    
    def _forloop_step(self, action):
        """
        Deprecated: handle the users in sequential mode. This will be slow if GPU or powerful CPU devices are available.
        """
        print('Warnings: Deprecated env_step function which handles the users in sequential mode. This will be slow if GPU or powerful CPU devices are available.')
        assert self.states is not None
        total_cost, total_gmv, user_action_list = 0, 0, []
        # order_num_array, order_fee_array = np.empty((self.states.shape[0], 1)), np.empty((self.states.shape[0], 1))
        num, discount = np.clip(action[0], 0, MarketingEnv.MAX_DELIVER_NUMBER), np.clip(action[1], MarketingEnv.DISCOUNT_COUPON_LIST[-1], MarketingEnv.DISCOUNT_COUPON_LIST[0])
        for index in tqdm(range(self.states.shape[0])):
            # 1. Insert the newly distributed coupon
            new_coupons = [(discount, MarketingEnv.EFFECTIVE_DAY) for _ in range(round(num))]
            self.effective_coupons_list[index].extend(new_coupons)
            if len(self.effective_coupons_list[index]) > 0:
                self.effective_coupons_list[index].sort(key=lambda x: (-x[0], -x[1]))

            # 2. Update the user state and calculate the user action
            self._update_user_state(index)
            real_state_value = self.states[index] * self.state_scaler
            total_num_h, avg_num_h, std_num_h, min_num_h, max_num_h, total_fee_h, avg_fee_h, std_fee_h, min_fee_h, \
                max_fee_h, active_time_h, discount_order_num_h, discount_order_day_h, discount_total_fee_h, history_accept_min_discount_h, \
                    week_index, coupon_num_now, coupon_discount_now = real_state_value
            user_input = torch.from_numpy(np.expand_dims(self.states[index], 0)).float()

            # 2.1 Network output user action
            with torch.no_grad():
                user_action = self.user_policy_net.select_action(user_input.to(self.device), eval=False)[1][0].cpu().numpy() # order_num, order_fee
            day_order_num, day_avg_fee = round(user_action[0] * MarketingEnv.DIM_MAX_DICT['day_order_num']), user_action[1] * MarketingEnv.DIM_MAX_DICT['day_avg_fee']


            """The environment needs to be filtered through a set of rules found in real-world testing to make it more robust and realistic
            """
            if (week_index == 4 or week_index == 0): # Monday or Friday
                # In actual tests, Monday and Friday orders were about 25% higher than usual
                day_order_num += torch.bernoulli(torch.tensor(0.25)).item()
            elif week_index == 5 and day_order_num >= 1: # Saturday
                # In actual tests, Saturday orders showed a drop, about 25 percent below normal
                day_order_num -= torch.bernoulli(torch.tensor(0.25)).item()
            elif week_index == 6 and day_order_num >= 1: # Sunday
                # In actual tests, orders on Sundays were a steep drop, about 40 percent below normal
                day_order_num -= torch.bernoulli(torch.tensor(0.4)).item()
            else:
                pass

            # rule 2: Based on coupons and users' psychological expectations, personalized correction of daily orders
            if coupon_num_now > 0 and day_order_num == 0 and coupon_discount_now < history_accept_min_discount_h:
                # The coupon did not meet the user's expectation and the user took no order
                day_order_num += torch.bernoulli(torch.tensor(coupon_discount_now)).item()
            elif coupon_num_now > 0 and day_order_num == 0 and coupon_discount_now >= history_accept_min_discount_h:
                # The coupon met the user's expectation and the user took no order
                day_order_num = 1
            elif coupon_num_now > 0 and day_order_num >= 1 and coupon_discount_now < history_accept_min_discount_h:
                # The coupon did not meet the user's expectations, but the user still took a taxi
                day_order_num -= torch.bernoulli(torch.tensor(1 - coupon_discount_now)).item()
            elif coupon_num_now == 0 and day_order_num >= 1:
                # No delivery and high user expectations (at least 15% discount)
                if history_accept_min_discount_h >= 0.15:
                    day_order_num -= round(0.75 * day_order_num)
                # No delivery and low user expectations
                else:
                    day_order_num -= torch.bernoulli(torch.tensor(0.25)).item()

            # rule 3: To improve the effect of coupon 60、65、70、75, We increase
            # the number of orders based discount of the coupon
            if coupon_discount_now >= 0.25 and coupon_discount_now < 0.35:
                day_order_num += torch.bernoulli(torch.tensor(coupon_discount_now)).item()
            elif coupon_discount_now >= 0.35:
                day_order_num += torch.bernoulli(torch.tensor(coupon_discount_now*1.5)).item()

            # rule 4: clip average order fee
            if day_avg_fee > 0:
                day_avg_fee = np.clip(day_avg_fee, a_min=max(min_fee_h - (std_fee_h**0.5), 0), a_max=max(max_fee_h, avg_fee_h+3*(max_fee_h**0.5)))

            # rule 5: The symbols of day_order_num and day_avg_fee are unified
            day_order_num = round(day_order_num)
            if day_order_num >= 1.0 and day_avg_fee <= 0.1:
                day_avg_fee = (avg_fee_h + (torch.randn(1) * 1.5).item())
                day_order_num = float(day_avg_fee > 0.1) * day_order_num
                day_avg_fee = float(day_avg_fee > 0.1) * day_avg_fee
            elif day_order_num == 0:
                day_avg_fee = 0.0

            # 2.2 Calculates other part of the use action by rule
            day_coupon_used_num = min(day_order_num, len(self.effective_coupons_list[index]))
            if day_coupon_used_num > 0:
                used_coupons = self.effective_coupons_list[index][-round(day_coupon_used_num):]
                day_used_min_discount, discount_list = 1, []
                for (d, _) in used_coupons:
                    day_used_min_discount = min(day_used_min_discount, 1 - d)
                    discount_list.append(d)
                avg_discount_ratio = np.mean(discount_list)
            else:
                # means not using any coupons
                avg_discount_ratio = MarketingEnv.DISCOUNT_COUPON_LIST[0]
                day_used_min_discount = 1 - MarketingEnv.DISCOUNT_COUPON_LIST[-1]

            # 3. Reduce the validity of the remaining coupons by 1
            new_list = []
            for item in self.effective_coupons_list[index][0:len(self.effective_coupons_list[index])-round(day_coupon_used_num)]:
                if item[1] - 1 > 0:
                    new_list.append((item[0], item[1]-1))
            self.effective_coupons_list[index] = new_list
            assert len(self.effective_coupons_list[index]) == 0

            # caculate the next user state by all the variables
            total_num = total_num_h + day_order_num
            size = round(total_num_h / (avg_num_h + 1e-10) * (avg_num_h > 0))
            avg_num = avg_num_h + 1 / (size + 1) * (day_order_num - avg_num_h) * (day_order_num > 0)
            std_num = std_num_h + (day_order_num - avg_num) * (day_order_num - avg_num_h) * (day_order_num > 0)
            min_num = (min_num_h == 0) * day_order_num + (day_order_num == 0) * min_num_h + \
                    ((min_num_h > 0) & (day_order_num > 0)) * min(min_num_h, day_order_num)
            max_num = max(max_num_h, day_order_num)

            total_fee = total_fee_h + day_avg_fee
            size = round(total_fee_h / (avg_fee_h + 1e-10) * (avg_fee_h > 0))
            avg_fee = avg_fee_h + 1 / (size + 1) * (day_avg_fee - avg_fee_h) * (day_avg_fee > 0)
            std_fee = std_fee_h + (day_avg_fee - avg_fee) * (day_avg_fee - avg_fee_h) * (day_avg_fee > 0)
            min_fee = (min_fee_h == 0) * day_avg_fee + (day_avg_fee == 0) * min_fee_h + \
                    ((min_fee_h > 0) & (day_avg_fee > 0)) * min(day_avg_fee, min_fee_h)
            max_fee = max(max_fee_h, day_avg_fee)

            active_time = (day_order_num < 1) * (active_time_h + 1)
            discount_order_num = discount_order_num_h * MarketingEnv.ENV_DISCOUNT + day_order_num
            discount_order_day = discount_order_day_h * MarketingEnv.ENV_DISCOUNT + (day_order_num > 0)
            day_fee = day_order_num * day_avg_fee
            discount_total_fee = discount_total_fee_h * MarketingEnv.ENV_DISCOUNT + day_fee
            week_index = (week_index + 1) % 7
            update_flag = history_accept_min_discount_h > day_used_min_discount
            history_accept_min_discount = update_flag * day_used_min_discount + \
                (~update_flag) * history_accept_min_discount_h
            coupon_num_now, coupon_discount_now = self._get_coupon_info(index)

            self.states[index] = np.array([total_num, avg_num, std_num, min_num, max_num, total_fee, avg_fee, std_fee, min_fee, max_fee,
                active_time, discount_order_num, discount_order_day, discount_total_fee, history_accept_min_discount,
                week_index, coupon_num_now, coupon_discount_now])

            self.states[index] = self.states[index] / self.state_scaler
            self.states[index][1:5] = np.clip(self.states[index][1:5], 0, 1)
            self.states[index][6:] = np.clip(self.states[index][6:], 0, 1)

            self.done_list[index] = ((self.current_env_step_list[index] + 1) == MarketingEnv.MAX_ENV_STEP)
            self.current_env_step_list[index] += 1
            per_cost = (1 - avg_discount_ratio) * day_coupon_used_num * day_avg_fee
            per_gmv = day_avg_fee * day_order_num - (1 - avg_discount_ratio) * day_coupon_used_num * day_avg_fee
            user_action_list.append(np.array([round(day_order_num), day_avg_fee]))
            total_gmv += per_gmv
            total_cost += per_cost

        return total_cost, total_gmv, user_action_list

    def reset(self):
        """The states of all users is reset to the initial states of 2021-05-18
        Return:
            None
        """
        self.states = self.val_initial_states
        self.done_list = [False for _ in range(self.states.shape[0])]
        self.current_env_step_list = [0 for _ in range(self.states.shape[0])]
        self.effective_coupons_list = [[] for _ in range(self.states.shape[0])]

    @property
    def validation_length(self,):
        if self.val_initial_states.shape[0] == 1000:
            return 14 # test for next two weeks
        elif self.val_initial_states.shape[0] == 10000:
            return 30 # test for the next month
        else:
            raise ValueError('The given data is wrong, please check it!!!')
