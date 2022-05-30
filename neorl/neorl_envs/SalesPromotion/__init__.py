from neorl.neorl_envs.SalesPromotion.sales_promo.env.env_instance import get_env_instance


def get_env(env_name):
    if env_name == "sales_promotion_v0":
        env = get_env_instance()
    
    return env

sales_promotion_envs = {
        "sp": "sales_promotion_v0",
        "sp_v0": "sales_promotion_v0",
        "sales_promotion_v0": "sales_promotion_v0",
}
