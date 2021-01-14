import porl
import sys


TEST_DATA_PATH = "../../data"


def test_citylearn():
    env = porl.make("citylearn")
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


def test_finance():
    env = porl.make("finance")
    train_data, val_data = env.get_dataset(train_num=99, data_type="H", path=TEST_DATA_PATH)


def test_ib():
    env = porl.make("ib")
    train_data, val_data = env.get_dataset(train_num=99, data_type="M", path=TEST_DATA_PATH)


def test_mujoco():
    env = porl.make("HalfCheetah-v3")
    train_data, val_data = env.get_dataset(train_num=99, data_type="L", path=TEST_DATA_PATH)


# def test_d4rl():
#     env = porl.make("halfcheetah-medium-v0")
#     train_data, val_data = env.get_dataset(train_num=99, data_type="L", path=TEST_DATA_PATH)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
