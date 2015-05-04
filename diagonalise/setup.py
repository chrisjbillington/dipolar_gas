# Run this setup script like so:
# python setup.py build_ext --inplace

# To produce html annotation for a cython file, instead run:
# cython -a myfile.pyx

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("diagonalise", ["diagonalise.pyx"])]
setup(
    name = "diagonalise",
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules
)
