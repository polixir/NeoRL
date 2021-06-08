from neorl.utils import get_json, LOCAL_JSON_FILE_PATH, DATA_PATH, download_dataset_from_url, get_file_md5


def print_all_kv_with_md5(json_file):
    items = json_file.items()
    for key, value in items:
        filename = download_dataset_from_url(value, name=key, to_path=DATA_PATH, verbose=0)
        __md5 = get_file_md5(filename)
        __new_kv = '"' + key + '": {"url": "' + value["url"] + '", "md5": "' + __md5 + '"}, '
        print(__new_kv)


if __name__ == '__main__':
    json_file = get_json(LOCAL_JSON_FILE_PATH)
    print_all_kv_with_md5(json_file)
