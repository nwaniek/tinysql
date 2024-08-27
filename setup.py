#!/usr/bin/env python

from setuptools import setup

setup(
    name='tinysql',
    version='0.2.7',
    description='A minimalistic object-relational mapper',
    author='Nicolai Waniek',
    author_email='n@rochus.net',
    py_modules=["tinysql"],
    install_requires=["numpy"],
    license='MIT'
)
