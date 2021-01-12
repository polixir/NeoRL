import porl


# TODO: data precision problem (bug) when re-calculating reward
env = porl.make("citylearn")
data = env.get_dataset()
# print("test citylearn", data)

env = porl.make("finance")
data = env.get_dataset()
# print("test finance", data)

env = porl.make("ib")
data = env.get_dataset(noise=False)
# print("test ib", data)
#
# env = porl.make("HalfCheetah-v3")
# data = env.get_dataset("HalfCheetah-v3")
# print("test HalfCheetah-v3", data)
#
# env = porl.make("halfcheetah-medium-v0")  # TODO: need to be compatible with d4rl
# data = env.get_dataset()
# print("test halfcheetah-medium-v0", data)
