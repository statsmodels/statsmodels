#! /usr/bin/env python

import os.path

cur_dir = os.path.abspath(os.path.dirname(__file__))

import sys
sys.path.insert(0, os.path.normpath(os.path.join(cur_dir,
                                        '..', '..', 'tools')))
from _build import cython, has_c_compiler
sys.path.pop(0)
del sys

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import (Configuration,
                                        get_numpy_include_dirs)
    config = Configuration('nonparametric', parent_package, top_path)

    #config.add_subpackage('tests')
    #config.add_subpackage('tests/results')
    config.add_data_dir('tests')
    config.add_data_dir('tests/results')
    config.add_data_files('tests/results/*.csv')
    #config.add_data_files('tests/Xi_test_data.csv')
    #config.add_data_files('tests/results/results_kde.csv')
    #config.add_data_files('tests/results/results_kde_fft.csv')
    #config.add_data_files('tests/results/results_kde_weights.csv')
    if has_c_compiler():
        cython(['fast_linbin.pyx'], working_path=cur_dir)

        config.add_extension('fast_linbin',
                         sources=['fast_linbin.c'],
                         include_dirs=[get_numpy_include_dirs()])


    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
