## Introduction
D3PE (Deep Data-Driven Policy Evaluation) aims to evaluation a large set of candidate policies by a fix dataset to select best ones.

## Supported Algorithms
- FQE
- MBOPE
- IS (WIS)
- Doubly-Robust

## Installation
```bash
pip install -e .
```

## Usage

```
from d3pe.utils.data import OPEDataset
from d3pe.utils.func import get_evaluator_by_name

policy = ... # load your policy, API in d3pe.evaluator.Policy needs to be supported.
dataset = OPEDataset(...) # load your data, see details in `d3pe.utils.data`

evaluator = get_evaluator_by_name('fqe')() # create an ope evaluator, for example FQE
evaluator.initialize(dataset) # initialize the evaluator
ope_score = evaluator(policy) # evaluate the policy
```