# https://github.com/AI4Finance-LLC/FinRL-Library
from neorl import core
import os
import numpy as np
import pandas as pd
from gym.utils import seeding
from gym import spaces


class FinrlEnv:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        stock_dim: int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount: int
            start money
        transaction_cost_pct : float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        tech_indicator_list: list
            a list of technical indicator names (modified from config.py)

    Methods
    -------
    create_env_training()
        create env class for training
    create_env_validation()
        create env class for validation
    create_env_trading()
        create env class for trading

    """
    def __init__(self, 
        stock_dim = 30,
        state_space = 181,
        hmax = 100,
        initial_amount = 1000000,
        transaction_cost_pct = 0.001,
        reward_scaling = 1e-4,
        tech_indicator_list = ['macd', 'rsi_30', 'cci_30', 'dx_30']):

        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list
        # account balance + close price + shares + technical indicators
        self.state_space = state_space
        self.action_space = self.stock_dim
        self.turbulence_threshold = 150
        
        path = os.path.dirname(__file__)
        self.train_data = pd.read_csv(os.path.join(path, "train.csv"), index_col=0)
        self.trade_data = pd.read_csv(os.path.join(path, "trade.csv"), index_col=0)


    def create_env_training(self):
        env_train = StockEnvTrain(df = self.train_data,
                                stock_dim = self.stock_dim,
                                hmax = self.hmax,
                                initial_amount = self.initial_amount,
                                transaction_cost_pct = self.transaction_cost_pct,
                                reward_scaling = self.reward_scaling,
                                state_space = self.state_space,
                                action_space = self.action_space,
                                tech_indicator_list = self.tech_indicator_list,
                                turbulence_threshold = self.turbulence_threshold )
        return env_train

    def create_env_trading(self):

        env_trade = StockEnvTrade(df = self.trade_data,
                                            stock_dim = self.stock_dim,
                                            hmax = self.hmax,
                                            initial_amount = self.initial_amount,
                                            transaction_cost_pct = self.transaction_cost_pct,
                                            reward_scaling = self.reward_scaling,
                                            state_space = self.state_space,
                                            action_space = self.action_space,
                                            tech_indicator_list = self.tech_indicator_list,
                                            turbulence_threshold = self.turbulence_threshold)

        return env_trade

def create_env(trade=False):
    env = FinrlEnv()

    if trade:
        return env.create_env_trading()
    else:
        return env.create_env_training()


class StockEnvTrain(core.EnvData):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df,
                stock_dim,
                hmax,
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold,
                day = 0):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.action_space,)) 
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30] 
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.terminal = False             
        # initalize state
        self.state = [self.initial_amount] + \
                      self.data.close.values.tolist() + \
                      [0]*self.stock_dim  + \
                      sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
        # initialize reward
        self.reward = 0
        self.cost = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.trades = 0
        #self.reset()
        self._seed()



    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.state[index+self.stock_dim+1] > 0:
            #update balance
            self.state[0] += \
            self.state[index+1]*min(abs(action),self.state[index+self.stock_dim+1]) * \
             (1- self.transaction_cost_pct)

            self.state[index+self.stock_dim+1] -= min(abs(action), self.state[index+self.stock_dim+1])
            self.cost +=self.state[index+1]*min(abs(action),self.state[index+self.stock_dim+1]) * \
             self.transaction_cost_pct
            self.trades+=1
        else:
            pass

    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index+1]
        # print('available_amount:{}'.format(available_amount))

        #update balance
        self.state[0] -= self.state[index+1]*min(available_amount, action)* \
                          (1+ self.transaction_cost_pct)

        self.state[index+self.stock_dim+1] += min(available_amount, action)

        self.cost+=self.state[index+1]*min(available_amount, action)* \
                          self.transaction_cost_pct
        self.trades+=1
        
    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique())-1
        # print(actions)

        if self.terminal:
            return self.state, self.reward, self.terminal,{}

        else:
            # print(np.array(self.state[1:29]))

            actions = actions * self.hmax
            #actions = (actions.astype(int))
            
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            #print("begin_total_asset:{}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day,:]         
            #load next state
            # print("stock_shares:{}".format(self.state[29:]))
            self.state =  [self.state[0]] + \
                    self.data.close.values.tolist() + \
                    list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
                    sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
            
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory.append(end_total_asset)
            #print("end_total_asset:{}".format(end_total_asset))
            
            self.reward = end_total_asset - begin_total_asset            
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            
            self.reward = self.reward*self.reward_scaling



        return np.array(self.state), self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        #initiate state
        self.state = [self.initial_amount] + \
                      self.data.close.values.tolist() + \
                      [0]*self.stock_dim + \
                      sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
        # iteration += 1 
        return np.array(self.state)
    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class StockEnvTrade(core.EnvData):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df, 
                stock_dim,
                hmax,                
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold,
                day = 0, iteration=''):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        # action_space normalization and shape is self.self.stock_dim
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.action_space,)) 
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30] 
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.terminal = False     
        self.turbulence_threshold = turbulence_threshold
        # initalize state
        self.state = [self.initial_amount] + \
                      self.data.close.values.tolist() + \
                      [0]*self.stock_dim  + \
                      sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self.data.date.unique()[0]]
        #self.reset()
        self._seed()
        
        self.iteration=iteration


    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.turbulence<self.turbulence_threshold:
            if self.state[index+self.stock_dim+1] > 0:
                #update balance
                self.state[0] += \
                self.state[index+1]*min(abs(action),self.state[index+self.stock_dim+1]) * \
                 (1- self.transaction_cost_pct)
                
                self.state[index+self.stock_dim+1] -= min(abs(action), self.state[index+self.stock_dim+1])
                self.cost +=self.state[index+1]*min(abs(action),self.state[index+self.stock_dim+1]) * \
                 self.transaction_cost_pct
                self.trades+=1
            else:
                pass
        else:
            # if turbulence goes over threshold, just clear out all positions 
            if self.state[index+self.stock_dim+1] > 0:
                #update balance
                self.state[0] += self.state[index+1]*self.state[index+self.stock_dim+1]* \
                              (1- self.transaction_cost_pct)
                self.state[index+self.stock_dim+1] =0
                self.cost += self.state[index+1]*self.state[index+self.stock_dim+1]* \
                              self.transaction_cost_pct
                self.trades+=1
            else:
                pass
    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        if self.turbulence< self.turbulence_threshold:
            available_amount = self.state[0] // self.state[index+1]
            # print('available_amount:{}'.format(available_amount))
            
            #update balance
            self.state[0] -= self.state[index+1]*min(available_amount, action)* \
                              (1+ self.transaction_cost_pct)

            self.state[index+self.stock_dim+1] += min(available_amount, action)
            
            self.cost+=self.state[index+1]*min(available_amount, action)* \
                              self.transaction_cost_pct
            self.trades+=1
        else:
            # if turbulence goes over threshold, just stop buying
            pass
        
    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique())-1
        # print(actions)

        if self.terminal:            
            return self.state, self.reward, self.terminal,{}

        else:
            # print(np.array(self.state[1:29]))
            actions = actions * self.hmax
            self.actions_memory.append(actions)
            #actions = (actions.astype(int))
            if self.turbulence>=self.turbulence_threshold:
                actions=np.array([-self.hmax]*self.stock_dim)
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            #print("begin_total_asset:{}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day,:]         
            self.turbulence = self.data['turbulence'].values[0]
            #print(self.turbulence)
            #load next state
            # print("stock_shares:{}".format(self.state[29:]))
            self.state =  [self.state[0]] + \
                    self.data.close.values.tolist() + \
                    list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
                    sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])

            
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self.data.date.unique()[0])
            #print("end_total_asset:{}".format(end_total_asset))
            
            self.reward = end_total_asset - begin_total_asset            
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            
            self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):  
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        #self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self.data.date.unique()[0]]
        #initiate state
        self.state = [self.initial_amount] + \
                      self.data.close.values.tolist() + \
                      [0]*self.stock_dim + \
                      sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])

            
        return self.state
    
    def render(self, mode='human',close=False):
        return self.state

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory[:-1]
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
