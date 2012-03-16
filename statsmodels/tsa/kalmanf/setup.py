#! /usr/bin/env python

import os.path

cur_dir = os.path.abspath(os.path.dirname(__file__))

import sys
sys.path.insert(0, os.path.normpath(os.path.join(cur_dir,
                                        '..', '..', '..', 'tools')))
from _build import cython, has_c_compiler
sys.path.pop(0)
del sys

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import (Configuration,
                                           get_numpy_include_dirs)

    config = Configuration('kalmanf', parent_package, top_path)

    # This function tries to create C files from the given .pyx files.  If
    # it fails, we build the checked-in .c files.
    if has_c_compiler():
        cython(['kalman_loglike.pyx'], working_path=cur_dir)

        config.add_extension('kalman_loglike',
                         sources=['kalman_loglike.c'],
                         include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**(configuration(top_path='').todict()))
