from os.path import join, dirname, realpath
from setuptools import setup
import sys

setup(
    name='fuzzy_rl',
    py_modules=['fuzzy_rl'],
    version='0.1',
    install_requires=[
        'gymnasium[classical_control]',
        'numpy',
        'tensorflow',
        'tqdm'
    ],
    description="deep RL actor critic methods with differentiable fuzzy logic",
    author="Bassel El Mabsout"
)
