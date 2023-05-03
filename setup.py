# coding: utf-8
from setuptools import Extension, setup
from setuptools import dist
from distutils.core import setup
from pathlib import Path
from setuptools.command.build_ext import build_ext

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

class NumpyExtension(Extension):
    # setuptools calls this function after installing dependencies
    def _convert_pyx_sources_to_lang(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        super()._convert_pyx_sources_to_lang()

extensions = NumpyExtension(
    name = "tuotuo/cutils", 
    sources = ["tuotuo/cutils.pyx"],
    libraries= ['Cython']
)

def cythonise_extensions(extensions): 
    from Cython.Build import cythonize
    return cythonize(
        extensions, 
        annotate = True,

    )

setup(
    name = 'TuoTuo',
    packages = ['tuotuo'],
    version = '0.2.0',  
    license='MIT',
    description = 'LDA & Neura based topic modelling library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author = 'tuotuo Superman',
    author_email = 'tuotuo@HanwellSquare.BigForce.com',
    url = 'https://github.com/RobbenRibery/TuoTuo',
    download_url = 'https://github.com/RobbenRibery/TuoTuo/archive/refs/tags/pypi-test-0.01.tar.gz',
    keywords = ['Generative Topic Modelling','Latent Dirichlet Allocation'],
    install_requires=[            
        'numpy',
        'torch',
        'scipy',
        'pyro-ppl',
        'pandas',
        'nltk',
        'spacy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha', 
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    ext_modules=cythonise_extensions(
        extensions, 
    ),
)