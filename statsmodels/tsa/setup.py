#! /usr/bin/env python

import os.path

base_path = os.path.abspath(os.path.dirname(__file__))

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import (Configuration,
                            get_numpy_include_dirs)
    config = Configuration('tsa', parent_package, top_path)

    config.add_subpackage('kalmanf')

    config.add_data_files('vector_ar/data/*.dat')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
