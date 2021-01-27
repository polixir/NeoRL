from newrl.newrl_envs.ib.industrial_benchmark_python.IBGym import IBGym

ib_continuous_v0 = IBGym(setpoint=70, reward_type='classic', action_type='continuous', observation_type='include_past')

ib_discrete_v0 = IBGym(setpoint=70, reward_type='classic', action_type='discrete', observation_type='include_past')

ib_envs = {
        "ib": ib_continuous_v0,
        "ib_continuous_v0": ib_continuous_v0,
        "ib_discrete_v0": ib_discrete_v0,
}
