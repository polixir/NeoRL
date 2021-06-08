import gym
from .utils import get_json, sample_dataset, LOCAL_JSON_FILE_PATH, DATA_PATH


class EnvData(gym.Env):
    def get_dataset(self, task_name_version: str = None, data_type: str = "high", train_num: int = 100,
                    need_val: bool = True, val_ratio: float = 0.1, path: str = DATA_PATH, use_data_reward: bool = True):
        """
        Get dataset from given env.

        :param task_name_version: The name and version (if applicable) of the task,
            default is the same as `task` while making env
        :param data_type: Which type of policy is used to collect data. It should
            be one of ["high", "medium", "low"], default to `high`
        :param train_num: The num of trajectory of training data. Note that the num
            should be less than 10,000, default to `100`
        :param need_val: Whether needs to download validation data, default to `True`
        :param val_ratio: The ratio of validation data to training data, default to `0.1`
        :param path: The directory of data to load from or download to `./data/`
        :param use_data_reward: Whether uses default data reward. If false, a customized
            reward function should be provided by users while making env

        :return train_samples, val_samples
        """

        # EXPERT = ["Expert", "expert", "E", "e"]
        HIGH = ["High", "high", "H", "h"]
        MEDIUM = ["Medium", "medium", "M", "m"]
        LOW = ["Low", "low", "L", "l"]

        if data_type in HIGH:
            data_type = "high"
        elif data_type in MEDIUM:
            data_type = "medium"
        elif data_type in LOW:
            data_type = "low"
        # elif data_type in EXPERT:
        #     data_type = "expert"
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
        """
        Users can call this func to set customized reward func.
        """
        self._reward_func = reward_func

    def get_reward_func(self):
        """
        Users can call this func to get customized reward func.
        """
        return self._reward_func

    def set_name(self, name):
        """
        Set name for envs.
        """
        self._name = name
        
    def set_done_func(self, done_func):
        """
        Users can call this func to set done func.
        """
        self._done_func = done_func

    def get_done_func(self):
        """
        Users can call this func to get done func.
        """
        return self._done_func
