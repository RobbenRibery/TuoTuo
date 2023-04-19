
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy 

extensions = [
    Extension(
        "cutils", 
        ["cutils.pyx"],
        include_dirs=[numpy.get_include()], 
    ),
]
setup(
    ext_modules=cythonize(extensions, annotate=True),
)