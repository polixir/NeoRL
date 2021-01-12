# porl

PORL is  an open-source benchmark for offline reinforcement learning. It provides standardized environments , datasets and reward function for training and benchmarking algorithms.

## Install porl

```
pip install -e .
```

## Example

```
import porl
env = porl.make("citylearn")
data = env.get_dataset()

env = porl.make("HalfCheetah-v3")
data = env.get_dataset("HalfCheetah-v3")
```
