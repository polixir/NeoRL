from .core import get_json, OFFLINE_DATA_MAP
import importlib
from porl.porl_envs.finance.finance_reward import get_reward


def make(task : str, reward_func=None):
    try:    
        if task.startswith("ib"):
            from porl.porl_envs.ib import ib_envs
            assert task in ib_envs.keys()
            env = ib_envs[task]
        elif task == 'traffic':
            pass
        elif task == "citylearn":
            from porl.porl_envs.citylearn import citylearn_envs
            assert task in citylearn_envs.keys()
            env = citylearn_envs[task] 
        elif task == 'finance':
            from porl.porl_envs.finance import finance_envs
            assert task in finance_envs.keys()
            env = finance_envs[task]
        elif task in ["HalfCheetah-v3", "Walker2d-v3", "Hopper-v3"]:
            from porl.porl_envs import mujoco
            env = mujoco.make_env(task)
        elif task in ['halfcheetah-medium-v0', 'hopper-medium-v0', 'walker2d-medium-v0']:
            from porl.porl_envs import d4rl
            env = d4rl.make_env(task)
        else:
            raise ValueError(f'Env {task} is not supported!')

        env.set_name(task)
        default_reward_func = importlib.import_module(f"porl.porl_envs.{task}.{task}_reward").get_reward
        env.set_reward_func(default_reward_func if reward_func is None else reward_func)

    except Exception as e:
        print(f'Warning: Env {task} can not be create. Pleace Check!')
        raise e

    return env
