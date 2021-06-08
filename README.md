# NeoRL
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License](https://licensebuttons.net/l/by/3.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

This repository is the interface for the offline reinforcement learning benchmark NeoRL: A Near Real-World Benchmark for Offline Reinforcement Learning.

The NeoRL benchmark contains environments, datasets, and reward functions for training and benchmarking offline reinforcement learning algorithms. Current benchmark contains environments of CityLearn, FinRL, IB, and three Gym-MuJoCo tasks.

More about the NeoRL benchmark can be found at http://polixir.ai/research/neorl and an early version of the paper

> Rongjun Qin, Songyi Gao, Xingyuan Zhang, Zhen Xu, Shengkai Huang, Zewen Li, Weinan Zhang, Yang Yu. NeoRL: A Near Real-World Benchmark for Offline Reinforcement Learning. https://arxiv.org/abs/2102.00714

Running the benchmark relies on two additional repositories, i.e. [OfflineRL](https://agit.ai/Polixir/OfflineRL) for training offline RL algorithms and [d3pe](https://agit.ai/Polixir/d3pe) for offline evaluation (also included in this repo under the benchmark directory). Steps for reproducing the benchmark can be found at [here](benchmark/).

## Install NeoRL

NeoRL can be installed as follows:

```
git clone https://github.com/polixir/NeoRL.git
cd neorl
pip install -e .
```

After installation, CityLearn, Finance, and the industrial benchmark will be available. If you want to use MuJoCo in your tasks, it is necessary to obtain a [license](https://www.roboti.us/license.html) and follow the setup instructions, and then run:

```
pip install -e .[mujoco]
```

So far, "HalfCheetah-v3", "Walker2d-v3", and "Hopper-v3" are supported within MuJoCo.

## Using NeoRL

NeoRL uses the [OpenAI Gym](https://github.com/openai/gym) API. Tasks are created via the `neorl.make` function. A full list of all tasks is [available here](https://github.com/polixir/NeoRL/wiki/Tasks).

```
import neorl

# Create an environment
env = neorl.make("citylearn")
env.reset()
env.step(env.action_space.sample())

# Get 100 trajectories of low level policy collection on citylearn task
train_data, val_data = env.get_dataset(data_type = "low", train_num = 100)
```

To facilitate setting different goals, users can provide customized reward function to `neorl.make()` while creating an env. See [usage and examples of `neorl.make()`](https://github.com/polixir/NeoRL/wiki/Parameter-usage#user-content-usage-of-neorl-make) for more details.

In order to test algorithms conveniently and quickly, each task is associated 
with a small training dataset and a validation dataset (also test dataset) by default. They can be obtained by 
`env.get_dataset()`. Extra parameters can be passed into `get_dataset()` 
to get multiple pairs of datasets for benchmarking. Each task collects data using a low, medium, 
or high level policy; for each task, we provide training data with a maximum of 10,000 trajectories. 
See [usage of `get_dataset()`](https://github.com/polixir/NeoRL/wiki/Parameter-usage#user-content-usage-of-get-dataset) for more details about parameter usage.

## Data in NeoRL

In NeoRL, training data and validation data returned by `get_dataset()` function are `dict` with  the same format:

- `obs`: A batch of current observations.

- `next_obs`: A batch of next observations.

- `action`: A batch of actions.

- `reward`: A batch of 1-dimensional array of rewards.

- `done`: A batch of 1-dimensional array of episode termination flags.

- `index`: An trajectory number-dimensional array. 
  The numbers in index indicate the beginning of trajectories.

## Reference


- **CityLearn**: Vázquez-Canteli J R, Kämpf J, Henze G, et al. "CityLearn v1.0: An OpenAI Gym Environment for Demand Response with Deep Reinforcement Learning." Proceedings of the 6th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation, pp. 356-357, 2019. [paper](https://dl.acm.org/doi/10.1145/3360322.3360998) [code](https://github.com/intelligent-environments-lab/CityLearn)
- **FinRL**: Liu X Y, Yang H, Chen Q, et al. "FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance." arXiv preprint arXiv:2011.09607, 2020. [paper](https://arxiv.org/abs/2011.09607) [code](https://github.com/AI4Finance-LLC/FinRL-Library)
- **Industrial Benchmark**: Hein D, Depeweg S, Tokic M, et al. "A Benchmark Environment Motivated by Industrial Control Problems." Proceedings of the 2017 IEEE Symposium Series on Computational Intelligence, pp. 1-8, 2017. [paper](https://arxiv.org/abs/1709.09480) [code](https://github.com/siemens/industrialbenchmark)
- **MuJoCo**: Todorov E, Erez T, Tassa Y. "Mujoco: A Physics Engine for Model-based Control." Proceedings of the 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 5026-5033, 2012. [paper](https://ieeexplore.ieee.org/abstract/document/6386109) [website](https://gym.openai.com/envs/#mujoco)

## Licenses
All datasets are licensed under the [Creative Commons Attribution 4.0 License (CC BY)](https://creativecommons.org/licenses/by/4.0/), and code is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html).