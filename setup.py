#!/usr/bin/env python
# coding: utf-8

from setuptools import Extension, setup
#from Cython.Build import cythonize
import numpy 
from distutils.core import setup

extensions = [
    Extension(
        "src/cutils", 
        ["src/cutils.pyx"],
        include_dirs=[numpy.get_include()], 
    ),
]

setup(
    name = 'tuotuo',
    packages = ['src'],
    version = '0.0.',  
    description = 'LDA & Neura based topic modelling library',
    author = 'TuoTuo Superman',
    author_email = 'tuotuo@superman.com',
    url = 'https://github.com/RobbenRibery/TuoTuo/tree/main',
    download_url = 'TuoTuo-ReleaseV0.tar.gz',
    keywords = ['Latent Dirichlet Allocation', 'Topic Modelling'],
    #ext_modules=cythonize(extensions, annotate=True),
)