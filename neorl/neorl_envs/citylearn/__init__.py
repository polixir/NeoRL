from neorl.neorl_envs.citylearn.citylearnenv import citylearn


def get_env(env_name):
    if env_name == "citylearn_v0":
        env = citylearn()
    if env_name == "citylearn_v1":
        env = citylearn(clip_action=True)
    
    return env


citylearn_envs = {
        "citylearn": "citylearn_v0",
        "citylearn_v0": "citylearn_v0",
        "citylearn-v1": "citylearn_v1",
        "citylearn_v1": "citylearn_v1",
}