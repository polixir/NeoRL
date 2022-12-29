import importlib


def make(task: str, reward_func=None, done_func=None):
    try:
        if task.lower() == "ib" or task.lower() == "industrial-benchmark" or task.lower() == "industrial_benchmark":
            from neorl.neorl_envs.ib import ib_envs, get_env
            task = "ib"
            assert task in ib_envs.keys()
            env = get_env(ib_envs[task])
        elif "citylearn" in task.lower():
            from neorl.neorl_envs.citylearn import citylearn_envs, get_env
            # task = "citylearn"
            assert task in citylearn_envs.keys()
            env = get_env(citylearn_envs[task])
        elif task.lower() == "finance":
            from neorl.neorl_envs.finance import finance_envs, get_env
            task = "finance"
            assert task in finance_envs.keys()
            env = get_env(finance_envs[task])
        elif task == "logistics_distribution":
            from neorl.neorl_envs.logistics_distribution import logistics_distribution_envs
            assert task in logistics_distribution_envs.keys()
            env = logistics_distribution_envs[task]
        elif task in ["HalfCheetah-v3", "Walker2d-v3", "Hopper-v3"]:
            from neorl.neorl_envs import mujoco
            env = mujoco.make_env(task)
        elif task.lower() == 'sp' or task.lower() == 'sp_v0' or task.lower() == 'sp-v0' or task.lower() == 'sales_promotion_v0'or task.lower() == 'sales_promotion-v0':
            from neorl.neorl_envs.SalesPromotion import get_env, sales_promotion_envs
            task = 'sales_promotion_v0'  # In general, task name should match the name in data_map.json, however, the task name for SP env is hardcoded in the environment, so it can be different here
            assert task in sales_promotion_envs.keys()
            env = get_env(sales_promotion_envs[task])
        elif task.lower() =='ww' or task.lower() =='ww_v0' or task.lower() =='ww-v0' or task.lower() == 'waterworks_v0' or task.lower() == 'waterworks-v0':
            from neorl.neorl_envs.WaterWorks import get_env, waterworks_envs
            task = 'ww'
            assert task in waterworks_envs.keys()
            env = get_env(waterworks_envs[task])
            
        #elif task in ['halfcheetah-medium-v0', 'hopper-medium-v0', 'walker2d-medium-v0']:
        #    from neorl.neorl_envs import d4rl
        #    env = d4rl.make_env(task)
        else:
            raise ValueError(f'Env {task} is not supported!')
    except Exception as e:
        print(f'Warning: Env {task} can not be create. Pleace Check!')
        raise e
    env.reset()
    env.set_name(task)

    try:
        default_reward_func = importlib.import_module(f"neorl.neorl_envs.{task}.{task}_reward").get_reward
    except ModuleNotFoundError:
        default_reward_func = None
        
    env.set_reward_func(default_reward_func if reward_func is None else reward_func)
        

    try:
        default_done_func = importlib.import_module(f"neorl.neorl_envs.{task}.{task}_done").get_done
    except ModuleNotFoundError:
        default_done_func = None
    
    env.set_done_func(default_done_func if done_func is None else done_func)

    return env
