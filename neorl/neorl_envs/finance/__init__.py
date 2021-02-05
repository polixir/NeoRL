from neorl.neorl_envs.finance.finrl import create_env

def get_env(env_name):
    if env_name == "finance_v0":
        env = create_env()
    
    return env

finance_envs = {
        "finance": "finance_v0",
        "finance_v0": "finance_v0",
}
