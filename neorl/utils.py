import urllib.request
import json
import numpy as np
import re
import os
import neorl
import hashlib


OFFLINE_DATA_MAP = "https://polixir-ai.oss-cn-shanghai.aliyuncs.com/datasets/offline/data_map.json"
DATA_PATH = os.path.abspath(os.path.join(neorl.__file__, "../", "data/"))
LOCAL_JSON_FILE_PATH = os.path.abspath(os.path.join(neorl.__file__, "../", "data_map.json"))


def get_json(file):
    """
    Get json from local file.
    """
    with open(file, 'r', encoding='utf8')as f:
        ele_json = json.load(f)
    return ele_json


def get_file_md5(filename):
    if not os.path.isfile(filename):
        return
    my_hash = hashlib.md5()
    f = open(filename, 'rb')
    while True:
        b = f.read(8096)
        if not b:
            break
        my_hash.update(b)
    f.close()
    return my_hash.hexdigest()


def download_dataset_from_url(dataset_url_md5, name, to_path, verbose=1):
    """
    Download dataset from url to `to_path + name`.
    """
    # Prevent concurrent FileExistsError
    try:
        if not os.path.exists(to_path):
            os.mkdir(to_path)
    except Exception:
        pass

    dataset_url = dataset_url_md5["url"]
    dataset_md5 = dataset_url_md5["md5"]

    dataset_filepath = os.path.join(to_path, name)

    if os.path.exists(dataset_filepath):
        local_file_md5 = get_file_md5(dataset_filepath)
        if local_file_md5 == dataset_md5:
            return dataset_filepath
        else:
            if verbose != 0:
                print(f"Local dataset {name} is broken, ready to re-download.")
    if verbose != 0:
        print(f'Downloading dataset: {dataset_url} to {dataset_filepath}')
    urllib.request.urlretrieve(dataset_url, dataset_filepath)

    if not os.path.exists(dataset_filepath):
        raise IOError(f"Failed to download dataset from {dataset_url}")
    return dataset_filepath


def search_local_files(filename, data_type, train_or_val, path, data_json):
    """
    Search files containing both `filename` and `train_or_val` in `path`.
    If train_or_val is `train`, only search training data (file with string `train`), and so does `val`.
    """
    files = []

    if os.path.exists(path):
        all_files = os.listdir(path)
        for f in all_files:
            if f not in data_json.keys():
                continue
            if filename in f and data_type in f and train_or_val in f:
                dataset_filepath = os.path.join(path, f)
                local_file_md5 = get_file_md5(dataset_filepath)
                dataset_md5 = data_json[f]["md5"]
                if local_file_md5 == dataset_md5:
                    files.append(f)
                else:
                    print(f"{f} is broken so that cannot partition from it.")
    return files


def find_local_file(files, traj_num, train_or_val):
    """
    Find appropriate least dataset in local according to num.
    """
    least_num = np.Inf
    for f in files:
        name_list = re.split("[-.]", f)
        if train_or_val in name_list:
            for tmp in name_list:
                if tmp.isdigit():
                    num = int(tmp)
                    if traj_num <= num < least_num:
                        least_num = num
    return least_num


def find_remote_file(data_json, task_name_version, data_type, traj_num, train_or_val):
    """
    Find appropriate least dataset in remote (data_json) according to num.
    """
    least_num = np.Inf
    for k, v in data_json.items():
        if task_name_version in k and train_or_val in k and data_type in k:
            name_list = re.split("[-.]", k)
            for tmp in name_list:
                if tmp.isdigit():
                    num = int(tmp)
                    if traj_num <= num < least_num:
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
            samples[k] = v[0: int(data_dict["index"][num])]
    return samples


def sample_dataset(task_name_version, path, traj_num, data_json, data_type, use_data_reward, reward_func, train_or_val):
    """
    Warp the procedure of finding appropriate least file, downloading file, sampling num trajs,
    and re-calc reward by customized reward_func (if needed).
    """
    local_files = search_local_files(task_name_version, data_type, train_or_val, path, data_json)

    if len(local_files) != 0:  # find dataset in local
        least_num = find_local_file(local_files, traj_num, train_or_val)
        if least_num == np.Inf:  # find appropriate least dataset in remote
            least_num = find_remote_file(data_json, task_name_version, data_type, traj_num, train_or_val)
    else:  # find appropriate least dataset in remote
        least_num = find_remote_file(data_json, task_name_version, data_type, traj_num, train_or_val)

    if least_num == np.Inf:
        raise Exception("Could not find appropriate dataset, please reduce `num`!")

    data_key = "-".join([task_name_version, data_type, str(least_num), train_or_val])
    all_keys = data_json.keys()
    if_find_dataset = False

    for i in all_keys:
        if data_key in i:
            if_find_dataset = True
            try:
                data_url_md5 = data_json[i]
                data_path = download_dataset_from_url(data_url_md5, name=i, to_path=path)
                data_npz = np.load(data_path)
                data_dict = dict(data_npz)
                data_dict["index"] = np.insert(data_dict["index"], 0, 0)
                break
            except Exception:
                raise Exception(f"Could not download the dataset: {i}")

    if not if_find_dataset:
        raise Exception(f"Could not find the dataset: {data_key}")

    samples = sample_by_num(data_dict, num=traj_num)  # random=random, seed=seed)

    if not use_data_reward:
        if reward_func is None:
            raise Exception("reward_func is None!")
        __reward = reward_func(samples)  # support for calculating reward via customized reward func
        samples["reward"] = __reward

    return samples
