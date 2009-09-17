"""
setuptools must be installed first. If you do not have setuptools installed
please download and install it from http://pypi.python.org/pypi/setuptools
"""

descr = """
Statsmodels is a python package that provides an interface to scipy for
statistical computations including descriptive statistics and
fitting statistical models.

LICENSE: Simplified BSD
"""
import os
import sys

import setuptools
from numpy.distutils.core import setup

DISTNAME = 'scikits.statsmodels'
DESCRIPTION = 'Statistical computations and models for use with SciPy'
LONG_DESCRIPTION = descr
MAINTAINER = 'Skipper Seabold, Josef Perktold'
MAINTAINER_EMAIL =''
URL = ''
LICENSE = 'BSD'
DOWNLOAD_URL = ''

MAJ = 0
MIN = 1
REV = 0
DEV = False #True
QUALIFIER = 'b1'

classifiers = [ 'Development Status :: 3 - Alpha',
              'Environment :: Console',
              'Programming Language :: Python :: 2.4',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              #'License :: BSD',
              'Topic :: Scientific/Engineering']

def build_ver_str():
    return '%d.%d.%d' % (MAJ,MIN,REV)

def fbuild_fver_str():
    if DEV:
        return build_ver_str() + 'dev'
    else:
        return build_ver_str() + QUALIFIER

VERSION = build_ver_str()

def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    #if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path,
                           namespace_packages = ['scikits'],
                           name = DISTNAME,
                           version = fbuild_fver_str(),
                           maintainer  = MAINTAINER,
                           maintainer_email = MAINTAINER_EMAIL,
                           description = DESCRIPTION,
                           license = LICENSE,
                           url = URL,
                           download_url = DOWNLOAD_URL,
                           long_description = LONG_DESCRIPTION)
    config.add_subpackage('scikits')
    config.add_data_files('scikits/__init__.py')
    config.add_data_dir('scikits/statsmodels/tests')
    config.add_data_dir('scikits/statsmodels/examples')
    config.add_data_dir('scikits/statsmodels/docs')
    extradatafiles = [os.path.join(r,d) for r,ds,f in os.walk('scikits/statsmodels/datasets')
                      for d in f if not os.path.splitext(d)[1] in
                      ['.py', '.pyc']]
    for f in extradatafiles:
        config.add_data_files(f)

    #config.add_subpackage(DISTNAME)
    #config.add_subpackage('scikits/statsmodels/examples')
    #config.add_subpackage('scikits/statsmodels/tests')


    config.set_options(
            ignore_setup_xxx_py = True,
            assume_default_configuration = True,
            delegate_options_to_subpackages = True,
            quiet = False,
            )

    return config

if __name__ == "__main__":

    setup(configuration = configuration,
        #name = DISTNAME,
        install_requires = 'numpy',
        namespace_packages = ['scikits'],
        packages = setuptools.find_packages(),
        include_package_data = True,
        test_suite="nose.collector",
        zip_safe = False, # the package can not run out of an .egg file bc of
                          # nose tests
        classifiers = classifiers)
