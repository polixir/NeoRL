import porl


# TODO: data precision problem (bug) when re-calculating reward
env = porl.make("citylearn")
train_data, val_data = env.get_dataset(train_num=59)
# print("test citylearn", len(train_data["obs"]))

env = porl.make("finance")
train_data, val_data = env.get_dataset()

env = porl.make("ib")
train_data, val_data = env.get_dataset(train_num=99, data_type="M")

env = porl.make("HalfCheetah-v3")
train_data, val_data = env.get_dataset("HalfCheetah-v3")

# env = porl.make("halfcheetah-medium-v0")  # TODO: need to be compatible with d4rl
# train_data, val_data = env.get_dataset()
