import neorl
import sys
from neorl.core import DATA_PATH
import numpy as np


TEST_DATA_PATH = DATA_PATH


def test_citylearn():
    env = neorl.make("citylearn")
    TEST_NUM = 9
    train_data, val_data = env.get_dataset(train_num=TEST_NUM, path=TEST_DATA_PATH)
    assert len(train_data["obs"]) == TEST_NUM * 1000
    assert len(train_data["next_obs"]) == TEST_NUM * 1000
    assert len(train_data["action"]) == TEST_NUM * 1000
    assert len(train_data["reward"]) == TEST_NUM * 1000
    assert len(train_data["done"]) == TEST_NUM * 1000
    assert len(train_data["index"]) == TEST_NUM

    TEST_NUM = 10
    train_data, val_data = env.get_dataset(train_num=TEST_NUM, path=TEST_DATA_PATH)
    assert len(train_data["obs"]) == TEST_NUM * 1000
    assert len(train_data["next_obs"]) == TEST_NUM * 1000
    assert len(train_data["action"]) == TEST_NUM * 1000
    assert len(train_data["reward"]) == TEST_NUM * 1000
    assert len(train_data["done"]) == TEST_NUM * 1000
    assert len(train_data["index"]) == TEST_NUM

    TEST_NUM = 99
    train_data, val_data = env.get_dataset(train_num=TEST_NUM, path=TEST_DATA_PATH)
    assert len(train_data["obs"]) == TEST_NUM * 1000
    assert len(train_data["next_obs"]) == TEST_NUM * 1000
    assert len(train_data["action"]) == TEST_NUM * 1000
    assert len(train_data["reward"]) == TEST_NUM * 1000
    assert len(train_data["done"]) == TEST_NUM * 1000
    assert len(train_data["index"]) == TEST_NUM

    TEST_NUM = 100
    train_data, val_data = env.get_dataset(train_num=TEST_NUM, path=TEST_DATA_PATH)
    assert len(train_data["obs"]) == TEST_NUM * 1000
    assert len(train_data["next_obs"]) == TEST_NUM * 1000
    assert len(train_data["action"]) == TEST_NUM * 1000
    assert len(train_data["reward"]) == TEST_NUM * 1000
    assert len(train_data["done"]) == TEST_NUM * 1000
    assert len(train_data["index"]) == TEST_NUM

    TEST_NUM = 999
    train_data, val_data = env.get_dataset(train_num=TEST_NUM, path=TEST_DATA_PATH)
    assert len(train_data["obs"]) == TEST_NUM * 1000
    assert len(train_data["next_obs"]) == TEST_NUM * 1000
    assert len(train_data["action"]) == TEST_NUM * 1000
    assert len(train_data["reward"]) == TEST_NUM * 1000
    assert len(train_data["done"]) == TEST_NUM * 1000
    assert len(train_data["index"]) == TEST_NUM

    TEST_NUM = 99
    train_data, val_data = env.get_dataset(train_num=TEST_NUM, data_type="medium", path=TEST_DATA_PATH)
    assert len(train_data["index"]) == TEST_NUM

    train_data, val_data = env.get_dataset(train_num=TEST_NUM, data_type="low", path=TEST_DATA_PATH)
    assert len(train_data["index"]) == TEST_NUM

    train_data, val_data = env.get_dataset(train_num=TEST_NUM, need_val=False, path=TEST_DATA_PATH)
    assert val_data is None

    train_data, val_data = env.get_dataset(train_num=TEST_NUM, val_ratio=0.3, path=TEST_DATA_PATH)
    assert int(len(train_data["index"]) * 0.3) == len(val_data["index"])

    def customized_reward_func(data):
        obs = data["obs"]
        return np.ones((len(obs), 1))

    env = neorl.make("ib", reward_func=customized_reward_func)
    train_data, val_data = env.get_dataset(data_type="high", train_num=50, need_val=False, use_data_reward=False)
    assert len(train_data["index"]) == 50
    assert np.all(train_data["reward"] == np.ones_like(train_data["reward"]))
    assert val_data is None


def test_finance():
    env = neorl.make("finance")
    train_data, val_data = env.get_dataset(train_num=100, data_type="H", path=TEST_DATA_PATH)
    assert int(len(train_data["index"]) * 0.1) == len(val_data["index"])


def test_ib():
    env = neorl.make("ib")
    train_data, val_data = env.get_dataset(train_num=100, data_type="M", path=TEST_DATA_PATH)
    assert int(len(train_data["index"]) * 0.1) == len(val_data["index"])


# def test_logistics_distribution():
#     env = neorl.make("logistics_distribution")
#     train_data, val_data = env.get_dataset(train_num=99, data_type="l", path=TEST_DATA_PATH)
#     assert int(len(train_data["index"]) * 0.1) == len(val_data["index"])


def test_mujoco():
    env = neorl.make("HalfCheetah-v3")
    train_data, val_data = env.get_dataset(train_num=100, data_type="L", path=TEST_DATA_PATH)
    assert int(len(train_data["index"]) * 0.1) == len(val_data["index"])
    env = neorl.make("Walker2d-v3")
    train_data, val_data = env.get_dataset(train_num=10, data_type="m", path=TEST_DATA_PATH)
    assert int(len(train_data["index"]) * 0.1) == len(val_data["index"])
    env = neorl.make("Hopper-v3")
    train_data, val_data = env.get_dataset(train_num=0, data_type="e", path=TEST_DATA_PATH)
    assert int(len(train_data["index"]) * 0.1) == 0 and len(val_data["index"]) == 0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
