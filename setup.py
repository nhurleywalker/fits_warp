#! /usr/bin/env python
"""
Setup for fits_warp
"""
import os
import sys
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    """Read a file"""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_version():
    """Get the version number of AegeanTools"""
    import fits_warp
    return fits_warp.__version__


reqs = ['numpy>=1.12',
        'scipy>=1.0',
        'astropy>=2.0']

setup(
    name="FitsWarp",
    version=get_version(),
    author=["Natasha Hurley-Walker", "Paul Hancock"],
    author_email="nhw@icrar.org",
    description="FitsWarp",
    url="https://github.com/nhurleywalker/fits_warp",
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    install_requires=reqs,
    scripts=['fits_warp.py']
)
