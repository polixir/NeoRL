import porl

#env = porl.make("halfcheetah-medium-v0")
#env = porl.make("maze2d-umaze-v1")
#ds = env.get_dataset()
#print(ds)

env = porl.make("citylearn")
data = env.get_dataset("citylearn")
print("test1111111111", data)

env = porl.make("finance")
data = env.get_dataset("finance")
print("test2222222222", data)

env = porl.make("ib")
data = env.get_dataset("ib", noise=False)
print("test3333333333", data)

env = porl.make("HalfCheetah-v3")
data = env.get_dataset("HalfCheetah-v3")
print("test4444444444", data)

env = porl.make("maze2d-open-v0")
data = env.get_dataset()
print("test5555555555", data)

