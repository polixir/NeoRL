# NewRL

This repository is the interface for the offline reinforcement learning benchmark NewRL: NEar real-World benchmarks.

The NewRL benchmarks contains environments, datasets, and reward functions for training and benchmarking offline reinforcement learning algorithms. Current benchmarks contains environments of CityLearn, FinRL, IB, and three MuJoCo tasks.

More about the NewRL benchmarks can be found at http://polixir.ai/research/newrl and the following paper
> Rongjun Qin, Songyi Gao, Xingyuan Zhang, Zhen Xu, Shengkai Huang, Zewen Li, Weinan Zhang, Yang Yu. Near Real-World Benchmarks for Offline Reinforcement Learning. https://arxiv.org/

## Install NewRL interface

NewRL interface can be installed as follows:
```
git clone https://agit.ai/Polixir/newrl.git
cd newrl
pip install -e .
```

After installation, CityLearn, Finance, and the industrial benchmark will be available. If you want 
to leverage MuJoCo in your tasks, it is necessary to obtain a [license](https://www.roboti.us/license.html) 
and follow the setup instructions, and then run:
```
pip install -e .[mujoco]
```

So far "HalfCheetah-v3", "Walker2d-v3", and "Hopper-v3" are supported within MuJoCo.

## Example

```
import newrl

env = newrl.make("citylearn")
train_data, val_data = env.get_dataset()  # Use default params here.
```

As a benchmark, in order to test algorithms conveniently and quickly, each task is associated with 
a small training dataset and a validation dataset by default. They can be obtained by 
`env.get_dataset()`. Meanwhile, for flexibility, extra parameters can be passed into `get_dataset()` 
to get multiple pairs of datasets for benchmarking. 
See [wiki](https://agit.ai/Polixir/newrl/wiki) for more details.

## Data in newrl

In newrl, training data and validation data returned by `get_dataset()` function are `dict` with 
the same format:

- `obs`: An <i> N by observation dimensional array </i> of current step's observation.

- `next_obs`: An <i> N by observation dimensional array </i> of next step's observation.

- `action`: An <i> N by action dimensional array </i> of actions.

- `reward`: An <i> N dimensional array of rewards</i>.

- `done`: An <i> N dimensional array of episode termination flags</i>.

- `index`: An <i> trajectory number-dimensional array</i>. 
The numbers in index indicate the beginning of trajectories.

## Reference

- <b>CityLearn</b> <br>
[Vázquez-Canteli, José R., et al. "CityLearn v1.0: An OpenAI Gym Environment for Demand 
Response with Deep Reinforcement Learning", Proceedings of the 6th ACM International Conference, 
ACM New York p. 356-357, New York, 2019](https://dl.acm.org/doi/10.1145/3360322.3360998) <br>
Github: https://github.com/intelligent-environments-lab/CityLearn

- <b>FinRL</b> <br>
[Liu X Y, Yang H, Chen Q, et al. FinRL: A Deep Reinforcement Learning Library for 
Automated Stock Trading in Quantitative Finance. arXiv preprint arXiv:2011.09607, 
2020.](https://arxiv.org/abs/2011.09607) <br>
Github: https://github.com/AI4Finance-LLC/FinRL-Library

- <b>industrial benchmark</b> <br>
[D. Hein, S. Depeweg, et al. "A benchmark environment motivated by industrial control 
problems," in 2017 IEEE Symposium Series on Computational Intelligence (SSCI), 2017, 
pp. 1-8.](https://arxiv.org/abs/1709.09480) <br>
Github: https://github.com/siemens/industrialbenchmark

- <b>MuJoCo</b> <br>
[Todorov E, Erez T, Tassa Y. Mujoco: A physics engine for model-based control. 
2012 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 
2012: 5026-5033.](https://ieeexplore.ieee.org/abstract/document/6386109) <br>
MuJoCo: https://gym.openai.com/envs/#mujoco
