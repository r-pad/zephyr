#!/usr/bin/env python
from distutils.core import setup, Extension
from setuptools import find_packages

setup(
    name='zephyr',
    version='0.1dev',
    author='Brian Okorn and Qiao Gu',
    packages=find_packages('python'),
    package_dir={'': 'python'},
    description= 'Tools for oriented features',
    long_description='',
    package_data={'': ['*.so']}
)

