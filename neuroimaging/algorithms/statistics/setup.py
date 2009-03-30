import os.path
import numpy.core

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('statistics', parent_package, top_path)

    config.add_data_dir('tests')
    config.add_extension('intvol', ['intvol.c'], include_dirs = [os.path.abspath(os.path.join(os.path.dirname(numpy.core.__file__), 'include'))])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
