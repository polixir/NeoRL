from neorl.neorl_envs.citylearn.citylearnenv import citylearn


def get_env(env_name):
    if env_name == "citylearn_v0":
        env = citylearn()
    
    return env


citylearn_envs = {
        "citylearn": "citylearn_v0",
        "citylearn_v0": "citylearn_v0",
}