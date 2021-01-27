import newrl


env = newrl.make("citylearn")
train_data, val_data = env.get_dataset(data_type="medium", train_num=99, need_val=True)
print("citylearn:", train_data, val_data)
