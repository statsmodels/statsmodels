#!/usr/bin/env python
from os.path import join
import sys

from nipy import  __version__, __doc__

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    # The quiet=True option will silence all of the name setting warnings:
    # Ignoring attempt to set 'name' (from 'neuroimaging.core' to
    #    'neuroimaging.core.image')
    # Robert Kern recommends setting quiet=True on the numpy list, stating
    # these messages are probably only used in debugging numpy distutils.

    config.get_version('neuroimaging/version.py') # sets config.version

    config.add_subpackage('neuroimaging', 'neuroimaging')

    return config


def main():
    from numpy.distutils.core import setup

    setup( name = 'neuroimaging',
           description = 'This is a neuroimaging python package',
           author = 'Various',
           author_email = 'nipy-devel@neuroimaging.scipy.org',
           url = 'http://neuroimaging.scipy.org',
           long_description = __doc__,
           configuration = configuration)


if __name__ == "__main__":
    main()
