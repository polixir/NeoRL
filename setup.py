#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from setuptools import setup

setup(
    name='neorl',
    description="NeoRL is NEar real-world benchmarks for Offline Reinforcement Learning",
    url="https://agit.ai/Polixir/neorl.git",
    python_requires=">=3.7",
    version='0.3.0',
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
    }
)
