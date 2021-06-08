from neorl.neorl_envs.ib.industrial_benchmark_python.IBGym import IBGym

def get_env(env_name):
    if env_name == "ib_continuous_v0":
        env = IBGym(setpoint=70, reward_type='classic', action_type='continuous', observation_type='include_past')
    else:
        env = IBGym(setpoint=70, reward_type='classic', action_type='discrete', observation_type='include_past')
        
    return env

ib_envs = {
        "ib": "ib_continuous_v0",
        "ib_continuous_v0": "ib_continuous_v0",
        "ib_discrete_v0": "ib_discrete_v0",
}
