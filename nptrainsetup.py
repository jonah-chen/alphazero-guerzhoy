from distutils.core import setup, Extension
import numpy as np

module = Extension("nptrain", sources=["nptrain.c"], include_dirs=[np.get_include()])

setup(name="nptrain",
description="Training module for Gomoku using numpy arrays",
ext_modules=[module])