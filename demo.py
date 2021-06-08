import neorl


env = neorl.make("citylearn")
train_data, val_data = env.get_dataset(data_type="medium", train_num=100, need_val=True)
print("citylearn:", train_data, val_data)

reward_func = env.get_reward_func()
print("reward_func:", reward_func)
