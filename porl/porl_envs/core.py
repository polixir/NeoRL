import gym
import os
import urllib.request
import json
import numpy as np
import re


OFFLINE_DATA_MAP = "https://polixir-ai.oss-cn-shanghai.aliyuncs.com/datasets/offline/data_map.json"
DATA_PATH = "./data/"


def get_json(url):
    """
    Get json file from URL.
    """
    resp = urllib.request.urlopen(url)
    ele_json = json.loads(resp.read())
    return ele_json


def download_dataset_from_url(dataset_url, name, to_path=DATA_PATH):
    """
    Download dataset from url to `to_path + name`.
    """
    # Prevent concurrent FileExistsError
    try:
        if not os.path.exists(to_path):
            os.mkdir(to_path)
    except Exception:
        pass

    dataset_filepath = os.path.join(to_path, name)
    if not os.path.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath


def search_local_files(filename, train_or_val="train", path=DATA_PATH):
    """
    Search files containing both `filename` and `train_or_val` in `path`.
    If train_or_val is `train`, only search training data (file with string `train`), and so does `val`.
    """
    files = []

    if os.path.exists(path):
        all_files = os.listdir(path)
        for f in all_files:
            if filename in f and train_or_val in f:
                files.append(f)

    return files


def find_local_file(files, traj_num, train_or_val="train"):
    """
    Find appropriate least dataset in local according to traj_num.
    """
    least_num = np.Inf
    for f in files:
        name_list = re.split("-", f)
        if train_or_val in name_list:
            for tmp in name_list:
                if tmp.isdigit():
                    num = int(tmp)
                    if num >= traj_num and num < least_num:
                        least_num = num
    return least_num


def find_remote_file(data_json, task_name_version, traj_num, train_or_val="train"):
    """
    Find appropriate least dataset in remote (data_json) according to traj_num.
    """
    least_num = np.Inf
    for k, v in data_json.items():
        if task_name_version in k:
            name_list = re.split("-", k)
            if train_or_val in name_list:
                for tmp in name_list:
                    if tmp.isdigit():
                        num = int(tmp)
                        if num >= traj_num and num < least_num:
                            least_num = num
    return least_num


def sample_by_num(data_dict: dict, num: int):
    """
    Sample num trajs from data_dict.
    """
    samples = {}
    for k, v in data_dict.items():
        if k == "index":
            samples[k] = v[0: num]
        else:
            samples[k] = v[0: num * data_dict["index"][0]]
    return samples


class EnvData(gym.Env):
    @staticmethod
    def get_dataset(task_name_version: str, data_type: str = "high", train_num: int = 99, val_num: int = 10,
                    download_train: bool = True, download_val: bool = True, noise: bool = True):
        data_json = get_json(OFFLINE_DATA_MAP)

        data_train, data_val = {}, {}
        _data_key = "-".join([task_name_version, data_type])

        if download_train:
            data_key = "-".join([_data_key, str(train_num), "train"])
            if noise:
                data_key = "-".join([data_key, "noise"])
            data_key = data_key + ".npz"
            try:
                data_url = data_json[data_key]
                data_train_path = download_dataset_from_url(data_url, name=data_key)
                data_train_npz = np.load(data_train_path)
                data_train = dict(data_train_npz)
            except Exception:
                raise Exception(f"Could not find the dataset: {data_key}")

        if download_val:
            data_key = "-".join([_data_key, str(val_num), "val"])
            if noise:
                data_key = "-".join([data_key, "noise"])
            data_key = data_key + ".npz"
            try:
                data_url = data_json[data_key]
                data_val_path = download_dataset_from_url(data_url, name=data_key)
                data_val_npz = np.load(data_val_path)
                data_val = dict(data_val_npz)
            except Exception:
                raise Exception(f"Could not find the dataset: {data_key}")

        return [data_train, data_val]

    @staticmethod
    def get_dataset_by_traj_num(task_name_version: str, traj_num: int, data_type: str = "high",
                                train_or_val: str = "train", noise: bool = True, path: str = DATA_PATH,
                                random: bool = False, seed: int = 123):
        data_json = get_json(OFFLINE_DATA_MAP)

        local_files = search_local_files(task_name_version, train_or_val, path)

        if len(local_files) != 0:  # find dataset in local
            num = find_local_file(local_files, traj_num, train_or_val)
            if num == np.Inf:  # find appropriate least dataset in remote
                num = find_remote_file(data_json, task_name_version, traj_num, train_or_val)
        else:  # find appropriate least dataset in remote
            num = find_remote_file(data_json, task_name_version, traj_num, train_or_val)

        if num == np.Inf:
            raise Exception("Could not find appropriate dataset, please reduce `traj_num`!")

        data_key = "-".join([task_name_version, data_type, str(num), train_or_val])
        if noise:
            data_key = "-".join([data_key, "noise"])
        data_key = data_key + ".npz"
        data_url = data_json[data_key]
        data_train_path = download_dataset_from_url(data_url, name=data_key)
        data_train_npz = np.load(data_train_path)
        data_train = dict(data_train_npz)

        samples = sample_by_num(data_train, num=traj_num) # random=random, seed=seed)

        return samples


# data2 = EnvData.get_dataset("HalfCheetah-v3")
samples = EnvData.get_dataset_by_traj_num("HalfCheetah-v3", traj_num=3, train_or_val="train")
# print(samples)
