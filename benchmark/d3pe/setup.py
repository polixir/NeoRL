import os
from setuptools import setup
from setuptools import find_packages
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'd3pe'))
from version import __version__

assert sys.version_info.major == 3, \
    "This repo is designed to work with Python 3." \
    + "Please install it before proceeding."

setup(
    name='d3pe',
    author='Icarus',
    author_email='wizardicarus@gmail.com',
    packages=find_packages(),
    version=__version__,
    install_requires=[
        'torch',
        'torchvision',
        'ray',
        'gym',
        'numpy',
        'scipy',
        'matplotlib',
        'tensorboard',
        'tqdm',
    ],
    url="https://agit.ai/Polixir_AI/d3pe"
)
