#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from setuptools import setup

setup(
    name='newrl',
    description="newrl is NEar real-World benchmarks for offline Reinforcement Learning",
    url="https://agit.ai/Polixir/newrl.git",
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
