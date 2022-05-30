#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from setuptools import setup
from setuptools import find_packages

setup(
    name='neorl',
    description="NeoRL is NEar real-world benchmarks for Offline Reinforcement Learning",
    url="https://agit.ai/Polixir/neorl.git",
    python_requires=">=3.7",
    version='0.3.0',
    packages=find_packages(),
    install_requires=[
        'gym',
        'numpy',
        'pygame',
        'attrdict',
        'ray[tune]',
        'dm_tree',
        'pandas',
        'opencv_python',
        'torch',
        'tqdm',
    ],
    extras_require={
        'mujoco': ['mujoco-py']
    },
    package_data={'neorl': ['neorl_envs/finance/trade.csv',
                            'neorl_envs/finance/train.csv',
                            'neorl_envs/citylearn/data/*',
                            'neorl_envs/citylearn/data/Climate_Zone_1/*',
                            'neorl_envs/citylearn/data/Climate_Zone_2/*',
                            'neorl_envs/citylearn/data/Climate_Zone_3/*',
                            'neorl_envs/citylearn/data/Climate_Zone_4/*',
                            'neorl_envs/citylearn/data/Climate_Zone_5/*',
                            'data_map.json'
                           ]},
    include_package_data=True,
)
