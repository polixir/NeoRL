from ctypes import util
import gym
from neorl import core
from gym.utils.seeding import np_random
from gym.spaces import Box, MultiDiscrete
import onnxruntime as ort
import numpy as np
from copy import deepcopy
import os
# import onnx
from gym.utils.seeding import np_random


class Waterworks(core.EnvData):
    """
    This is a waterworks simulator. The state consists of two parts: obs and external variables. 
    "obs" contains the water flows and pressure of multiple staions, and the external variables include 
    temperature, day of the week, holiday or not, and the time embedding. The action is to control the pressure 
    of the stations, so that the pressure of the critical station (the 5-th dimension of the obs) is under control.
    The transition dynamics maps (obs, ex_var, action) to the next_obs, where the ex_var is from the static data. This transition 
    model is trained from another batch of real-world data.

    The policy should respond every 5 minutes and a trajectory lasts for 1 day (1440/5 - 1 = 287 time steps).
    """

    def __init__(self):
        # env_model = onnx.load('env_model.onnx')
        # onnx.checker.check_model(env_model)
        dir, _ = os.path.split(os.path.abspath(__file__))
        self.ort_session = ort.InferenceSession(os.path.join(dir, "env_model.onnx"))  # Load the offline learned environment model. This model is deterministic.
        self.env_data = np.load(os.path.join(dir, 'env_data.npz'))  # load init states and external variables

        self.observation_space = Box(low=np.array([0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0]), high=np.array([1000, 1000, 1000, 1000, 10, 100, 1, 1, 1, 1, 1, 1, 1, 1]), shape=(14,), dtype=np.float32)
        self.action_space = Box(low=np.array([0.0, 0.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0, 1.0]), shape=(4,), dtype=np.float32)
        self.state = self.observation_space.sample()
        self.total_days = self.env_data['init_obs'].shape[0]  # different days
        self.day_case = 0
        self.cur_step = 0
        super().__init__()

    def seed(self, seed):
        return np_random(seed)[0]

    def step(self, act):
        outputs = self.ort_session.run(  # infer one step using the model
            ['next_obs'],
            {"obs": np.array([self.state[:5]]),
             "act_pump": np.array([act]),
             "outsider": np.array([self.ex_var[self.cur_step]])}
        )
        next_obs = outputs[0][0]
        rew = self.get_reward(self.state[4], self.ex_var[self.cur_step][-1])
        self.cur_step += 1
        done = False
        if self.cur_step >= 287:
            done = True
        self.state = np.concatenate((next_obs, self.ex_var[self.cur_step]))
        return deepcopy(self.state), rew, done, {}

    def reset(self):
        self.day_case = np.random.randint(0, self.total_days)  # randomly select a day as the init_state
        # print('day: ', self.day_case)
        self.cur_step = 0
        self.ex_var = self.env_data['ex_var'][self.day_case]  # get the external vars
        self.init_obs = self.env_data['init_obs'][self.day_case]  # get the init obs
        self.state = np.concatenate((self.init_obs, self.ex_var[self.cur_step]))
        return deepcopy(self.state)

    def get_reward(self, obs, is_day):
        def day_reward(x):
            reward = np.zeros_like(x, dtype=x.dtype)
            reward[x < 0.25] = -1
            reward[x > 0.26] = 0.01 / (x[x > 0.26] - 0.26 + 0.01)
            return reward

        def night_reward(x):
            reward = np.zeros_like(x, dtype=x.dtype)
            reward[x < 0.22] = -1
            reward[x > 0.23] = 0.01 / (x[x > 0.23] - 0.23 + 0.01)
            return reward

        press = obs
        reward = np.where(is_day, day_reward(press), night_reward(press))
        return reward


if __name__ == "__main__":
    env = Waterworks()
    print(env.reset())
    ret_list = []
    for i in range(100):
        done = False
        step = 1
        ret = 0
        while not done:  # test random action
            act = env.action_space.sample()
            # print(act)
            obs, rew, done, _ = env.step(act)
            step += 1
            ret += rew
        ret_list.append(ret)
        # print(step, rew)
        env.reset()

    print(np.mean(ret_list), np.std(ret_list))
