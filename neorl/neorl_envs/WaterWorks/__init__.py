from neorl.neorl_envs.WaterWorks.waterworks.env.env import *


def get_env(env_name):
    if env_name == "waterworks_v0":
        env = Waterworks()
    
    return env

waterworks_envs = {
        "ww": "waterworks_v0",
        "ww_v0": "waterworks_v0",
        "waterworks_v0": "waterworks_v0",
}
