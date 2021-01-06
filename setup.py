#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from setuptools import setup

setup(
    name='porl',
    description="PORL is an open-source benchmark for offline reinforcement learning",
    url="https://agit.ai/Polixir_AI/porl.git",
    python_requires=">=3.7",
    version='0.1.0',
    install_requires=[
        'gym',
        'numpy',
        'pygame',
        'attrdict',
        'mujoco-py',
        'ray[tune]',
        'dm_tree',
        'pandas',
        'opencv_python',
        'torch',
        'tqdm'
    ]
    
)
