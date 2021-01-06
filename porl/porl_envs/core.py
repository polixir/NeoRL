import gym
import os
import urllib.request
import json
import numpy as np


OFFLINE_DATA_MAP = "https://polixir-ai.oss-cn-shanghai.aliyuncs.com/datasets/offline/data_map.json"


def get_json(url):
    resp = urllib.request.urlopen(url)
    ele_json = json.loads(resp.read())
    return ele_json


def download_dataset_from_url(dataset_url, name):
    DATA_PATH = "./data/"
    # Prevent concurrent FileExistsError
    try:
        if not os.path.exists(DATA_PATH):
            os.mkdir(DATA_PATH)
    except Exception:
        pass

    dataset_filepath = os.path.join(DATA_PATH, name)
    if not os.path.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
        print("finished")
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath


class EnvData(gym.Env):
    @staticmethod
    def get_dataset(task_name_version, data_type="high", train_num=99, val_num=10, download_train=True, download_val=True, noise=True):
        data_json = get_json(OFFLINE_DATA_MAP)

        data_train, data_val = {}, {}
        _data_key = "-".join([task_name_version, data_type])

        if download_train:
            data_key = "-".join([_data_key, str(train_num), "train"])
            if noise:
                data_key = "-".join([data_key, "noise"])
            data_key = data_key + ".npz"
            data_url = data_json[data_key]
            data_train_path = download_dataset_from_url(data_url, name=data_key)
            print("data_train_path:", data_train_path)
            data_train_npz = np.load(data_train_path)
            data_train = dict(data_train_npz)
        if download_val:
            data_key = "-".join([_data_key, str(val_num), "val"])
            if noise:
                data_key = "-".join([data_key, "noise"])
            data_key = data_key + ".npz"
            data_url = data_json[data_key]
            data_val_path = download_dataset_from_url(data_url, name=data_key)
            print("data_val_path:", data_val_path)
            data_val_npz = np.load(data_val_path)
            data_val = dict(data_val_npz)

        return [data_train, data_val]

