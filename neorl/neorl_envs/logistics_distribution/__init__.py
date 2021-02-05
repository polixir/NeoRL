from neorl.neorl_envs.logistics_distribution.ld_env import create_env

ld_v0 = create_env()

logistics_distribution_envs = {
        "logistics_distribution": ld_v0,
        "logistics_distribution_v0": ld_v0,
}
