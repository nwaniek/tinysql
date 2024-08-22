#!/usr/bin/env python

from setuptools import setup

setup(
    name='tinysql',
    version='0.2.1',
    description='A minimalistic ORM for sqlite',
    author='Nicolai Waniek',
    author_email='rochus@rochus.net',
    py_modules=["tinysql"],
    install_requires=["numpy"],
)
