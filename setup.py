#!/usr/bin/env python
# coding: utf-8

from setuptools import Extension, setup
#from Cython.Build import cythonize
#import numpy 
from distutils.core import setup
from pathlib import Path
this_directory = Path(__file__).parent

# extensions = [
#     Extension(
#         "src/cutils", 
#         ["src/cutils.pyx"],
#         include_dirs=[numpy.get_include()], 
#     ),
# ]

long_description = (this_directory / "README.md").read_text()

setup(
    name = 'tuotuo',
    packages = ['src'],
    version = '0.0.3',  
    license='MIT',
    description = 'LDA & Neura based topic modelling library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author = 'tuotuo Superman',
    author_email = 'tuotuo@HanwellSquare.BigForce.com',
    url = 'https://github.com/RobbenRibery/TuoTuo',
    download_url = 'https://github.com/RobbenRibery/TuoTuo/archive/refs/tags/Pypi-0.03.tar.gz',
    keywords = ['Generative Topic Modelling','Latent Dirichlet Allocation'],
    install_requires=[            
        'numpy',
        'torch',
        'scipy',
        'pyro',
        'nltk',
    ],
    #ext_modules=cythonize(extensions, annotate=True),
    classifiers=[
        'Development Status :: 3 - Alpha', 
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
  ],
)