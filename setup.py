"""
setuptools must be installed first. If you do not have setuptools installed
please download and install it from http://pypi.python.org/pypi/setuptools
"""

descr = """
Statmodels (?) is a python package that provides an interface to SciPy for
statistical computations including descriptive statistics and
fitting statistical models.

Brief history of the major codebase...

LICENSE: TBD
"""
import os
import sys

import setuptools
from numpy.distutils.core import setup

DISTNAME = 'scikits.statsmodels'
DESCRIPTION = 'Statistical computations and models for use with SciPy'
LONG_DESCRIPTION = descr
MAINTAINER = ''
MAINAINER_EMAIL =''
URL = ''
LICENSE = ''
DOWNLOAD_URL = ''

MAJ = 0
MIN = 1
REV = 1
DEV = True

classifiers = [ 'Development Status :: 3 - Alpha',
              'Environment :: Console',
              'Programming Language :: Python :: 2.4',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: TBD',
              'Topic :: Scientific/Engineering']

def build_ver_str():
    return '%d.%d.%d' % (MAJ,MIN,REV)

def fbuild_fver_str():
    if DEV:
        return build_ver_str() +'dev'
    else:
        return build_ver_str()

VERSION = build_ver_str()

def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path,
                           namespace_packages = ['scikits'],
                           name = DISTNAME,
                           version = fbuild_fver_str(),
                           maintainer  = MAINTAINER,
                           #maintainer_email = MAINTAINER_EMAIL,
                           description = DESCRIPTION,
                           license = LICENSE,
                           url = URL,
                           download_url = DOWNLOAD_URL,
                           long_description = LONG_DESCRIPTION)

    config.add_data_files('scikits/__init__.py')
    config.add_data_dir('scikits/statsmodels/tests')
    config.add_data_dir('scikits/statsmodels/examples')

    config.set_options(
            ignore_setup_xxx_py = True,
            assume_default_configuration = True,
            delegate_options_to_subpackages = True,
            quiet = False,
            )

    return config

if __name__ == "__main__":

    setup(configuration = configuration,
        install_requires = 'numpy',
        namespace_packages = ['scikits'],
        packages = setuptools.find_packages(),
        include_package_data = True,
        test_suite="tester",
        zip_safe = True, # the package can run out of an .egg file
        classifiers = classifiers)
