#!/usr/bin/env python

from setuptools import setup

setup(
    name='tinysql',
    version='0.2.1',
    description='A minimalistic object-relational mapper',
    author='Nicolai Waniek',
    author_email='rochus@rochus.net',
    py_modules=["tinysql"],
    install_requires=["numpy"],
)
