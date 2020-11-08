from distutils.core import setup, Extension

module = Extension("trainGomoku", sources=["trainmodule.c"])

setup(name="trainGomoku", version=0, 
description="A faster implementation of Gomoku game written in C used for training. C do be faster than python tho :((",
ext_modules=[module], author="Jonah Chen, Ahsan Kaleem")