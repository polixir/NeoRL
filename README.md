# porl

PORL is an open-source benchmark for offline reinforcement learning. It provides standardized environments, datasets, and reward function for training and benchmarking algorithms.

Currently, supported environments include citylearn, finance, ib, mujoco, and d4rl.

## Install porl

```
pip install -e .
```

## Example

```
import porl

env = porl.make("citylearn")
train_data, val_data = env.get_dataset()  # Use default args here.
```

## Advanced Usage

See [wiki](https://agit.ai/Polixir_AI/porl/wiki) for more details.

## Reference

- <b>CityLearn</b> <br>
[Vázquez-Canteli, José R., et al. "CityLearn v1.0: An OpenAI Gym Environment for Demand 
Response with Deep Reinforcement Learning", Proceedings of the 6th ACM International Conference, 
ACM New York p. 356-357, New York, 2019](https://dl.acm.org/doi/10.1145/3360322.3360998) <br>
Github: https://github.com/intelligent-environments-lab/CityLearn

- <b>Finance</b> <br>
[Liu X Y, Yang H, Chen Q, et al. FinRL: A Deep Reinforcement Learning Library for 
Automated Stock Trading in Quantitative Finance. arXiv preprint arXiv:2011.09607, 
2020.](https://arxiv.org/abs/2011.09607) <br>
Github: https://github.com/AI4Finance-LLC/FinRL-Library

- <b>industrial benchmark</b> <br>
[D. Hein, S. Depeweg, et al. "A benchmark environment motivated by industrial control 
problems," in 2017 IEEE Symposium Series on Computational Intelligence (SSCI), 2017, 
pp. 1-8.](https://arxiv.org/abs/1709.09480) <br>
Github: https://github.com/siemens/industrialbenchmark

- <b>mujoco</b> <br>

- <b>d4rl</b> <br>
