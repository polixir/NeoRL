from porl.porl_envs.finance.finrl import create_env

finance_v0 = create_env()

finance_envs = {
        "finance" : finance_v0,
        "finance_v0" : finance_v0, 
        }