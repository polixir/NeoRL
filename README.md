# porl

PORL is  an open-source benchmark for offline reinforcement learning. It provides standardized environments , datasets and reward function for training and benchmarking algorithms.

## Install bactchrl

```
pip install -e .
```

## Example

```
import porl

env = porl.make("halfcheetah-meidum-v3")

env = porl.make("citylearn")
```