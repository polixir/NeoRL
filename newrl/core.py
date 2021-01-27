import gym
from .utils import get_json, sample_dataset, LOCAL_JSON_FILE_PATH, DATA_PATH


class EnvData(gym.Env):
    def get_dataset(self, task_name_version: str = None, data_type: str = "high", train_num: int = 99,
                    need_val: bool = True, val_ratio: float = 0.1, path: str = DATA_PATH, use_data_reward: bool = True):

        EXPERT = ["expert", "E", "e"]
        HIGH = ["high", "H", "h"]
        MEDIUM = ["medium", "M", "m"]
        LOW = ["low", "L", "l"]

        if data_type in EXPERT:
            data_type = "expert"
        elif data_type in HIGH:
            data_type = "high"
        elif data_type in MEDIUM:
            data_type = "medium"
        elif data_type in LOW:
            data_type = "low"
        else:
            raise Exception(f"Please check `data_type`, {data_type} is not supported!")

        task_name_version = self._name if task_name_version is None else task_name_version

        data_json = get_json(LOCAL_JSON_FILE_PATH)

        train_samples = sample_dataset(task_name_version, path, train_num, data_json, data_type, use_data_reward,
                                       self._reward_func, "train")
        val_samples = None
        if need_val:
            val_samples = sample_dataset(task_name_version, path, int(train_num * val_ratio), data_json, data_type,
                                         use_data_reward, self._reward_func, "val")
        return train_samples, val_samples

    def set_reward_func(self, reward_func):
        self._reward_func = reward_func

    def set_name(self, name):
        self._name = name
