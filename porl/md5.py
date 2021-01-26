import hashlib
import os
from porl.utils import get_json, LOCAL_JSON_FILE_PATH, DATA_PATH, download_dataset_from_url
import json


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


def save_md5_to_data_map(json_file):
    items = json_file.items()
    for key, value in items:
        filename = download_dataset_from_url(value, name=key, to_path=DATA_PATH)
        __md5 = get_file_md5(filename)
        __new_kv = '"' + key + '": {"url": "' + value + '", "md5": "' + __md5 + '"}'
        # __new_json = json.loads(__new_kv)
        print(__new_kv)


if __name__ == '__main__':
    json_file = get_json(LOCAL_JSON_FILE_PATH)
    save_md5_to_data_map(json_file)

# print("MD5:", get_file_md5('./data/citylearn-high-10-val-noise.npz'))
