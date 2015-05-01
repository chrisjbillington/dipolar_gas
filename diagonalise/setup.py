# Run this setup script like so:
# python setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# import Cython.Compiler.Options
# Cython.Compiler.Options.annotate = True

import numpy

ext_modules = [Extension("diagonalise",
                         ["diagonalise.pyx"],
                         include_dirs = [numpy.get_include()])]
setup(
    name = "diagonalise",
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules
)
