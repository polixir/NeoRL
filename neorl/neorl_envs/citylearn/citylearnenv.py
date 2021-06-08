import os
import random
from neorl.neorl_envs.citylearn.citylearn import CityLearn


def citylearn():
    data_path = os.path.join(os.path.dirname(__file__),"data")
    climate_zone = random.choice([1,2,3,4,5])
    zone_data_path = os.path.join(data_path,"Climate_Zone_"+str(climate_zone))
    building_attributes = os.path.join(zone_data_path, 'building_attributes.json')
    weather_file = os.path.join(zone_data_path, 'weather_data.csv')
    solar_profile = os.path.join(zone_data_path, 'solar_generation_1kW.csv')
    building_state_actions = os.path.join(data_path,'buildings_state_action_space.json')
    building_ids = ['Building_1','Building_2','Building_3','Building_4','Building_5','Building_6','Building_7','Building_8']
    objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','quadratic']
    simulation_period_len = 1000
    env = CityLearn(zone_data_path, 
                    building_attributes, 
                    weather_file, 
                    solar_profile, 
                    building_ids, 
                    buildings_states_actions = building_state_actions, 
                    simulation_period_len = simulation_period_len, 
                    cost_function = objective_function, 
                    central_agent = True, 
                    verbose = 0)
    #print(env.action_space.shape,env.observation_space.shape)

    return env
