"""
Cython setup and compilation file
---------------------------------
Compiles the demand and supply functions

To run the program use the shell command: python setup.py build_ext --inplace
"""

__author__ = "Karl Naumann-Woleske"
__version__ = "0.0.1"
__license__ = "MIT"

from setuptools import setup
from setuptools.extension import Extension

import numpy

try:
    from Cython.Build import cythonize
except ImportError:
    # create closure for deferred import
    def cythonize (*args, ** kwargs ):
        from Cython.Build import cythonize
        return cythonize(*args, ** kwargs)

extensions = [Extension(
    name="step_functions",
    sources=["step_functions.pyx"],
    include_dirs=[numpy.get_include()],
    )
]


ext_options = {"compiler_directives": {"profile": False}, "annotate": True}
setup(
    ext_modules = cythonize(extensions),#'step_functions.pyx', **ext_options),
    include_dirs=[numpy.get_include()]
)
