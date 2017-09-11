#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages

setup(
    name='gphypo',
    version='0.0.1',
    description='Hyper-Prameter Optimization Library using GP-UCB',
    long_description='README.md',
    license='MIT',
    author='Kohei Watanabe',
    author_email='humilitiny@gmail.com',
    url='https://github.com/LittleWat/gp_ucb_python_lib',
    keywords='gp_ucb hyper_parameter_optimization python',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['scikit-learn>=0.18', 'tqdm', 'scikit-sparse'],
)
