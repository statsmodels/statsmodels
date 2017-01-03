#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_module = Extension(
    "Weibull_AS_cytools",
    ["Weibull_AS_cytools.pyx"],
    extra_compile_args=['-fopenmp', '-O3'],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()]
)

setup(
    name = "Weibull app",
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module]
)
