"""
setuptools must be installed first. If you do not have setuptools installed
please download and install it from http://pypi.python.org/pypi/setuptools
"""

descr = """
Statsmodels is a python package that provides a complement to scipy for
statistical computations including descriptive statistics and
estimation of statistical models.

scikits.statsmodels provides classes and functions for the estimation of
several categories of statistical models. These currently include linear
regression models, OLS, GLS, WLS and GLS with AR(p) errors, generalized
linear models for six distribution families, M-estimators for robust
linear models, and regression with discrete dependent variables, Logit,
Probit, MNLogit, Poisson, based on maximum likelihood estimators.
An extensive list of result statistics are avalable for each estimation
problem.

We welcome feedback:
mailing list at http://groups.google.com/group/pystatsmodels?hl=en  or
our bug tracker at https://bugs.launchpad.net/statsmodels

For updated versions between releases, we recommend our repository at
http://code.launchpad.net/statsmodels

Main Changes in 0.2.0
---------------------

* Improved documentation and expanded and more examples
* Added four discrete choice models: Poisson, Probit, Logit, and Multinomial Logit.
* Added PyDTA. Tools for reading Stata binary datasets (*.dta) and putting
  them into numpy arrays.
* Added four new datasets for examples and tests.
* Results classes have been refactored to use lazy evaluation.
* Improved support for maximum likelihood estimation.
* bugfixes
* renames for more consistency
  RLM.fitted_values -> RLM.fittedvalues
  GLMResults.resid_dev -> GLMResults.resid_deviance

Sandbox
-------

We are continuing to work on support for systems of equations models, panel data
models, time series analysis, and information and entropy econometrics in the
sandbox. This code is often merged into trunk as it becomes more robust.


"""
import os
import sys

import setuptools
from numpy.distutils.core import setup
import numpy

compile_cython = 0
if "--with-cython" in sys.argv:
    compile_cython = 1
    sys.argv.remove('--with-cython')


DISTNAME = 'scikits.statsmodels'
DESCRIPTION = 'Statistical computations and models for use with SciPy'
LONG_DESCRIPTION = descr
MAINTAINER = 'Skipper Seabold, Josef Perktold'
MAINTAINER_EMAIL =''
URL = ''
LICENSE = 'BSD License'
DOWNLOAD_URL = ''

MAJ = 0
MIN = 3
REV = 0
DEV = True #False
QUALIFIER = '' #'b2dev'

classifiers = [ 'Development Status :: 4 - Beta',
              'Environment :: Console',
              'Programming Language :: Python :: 2.4',
              'Operating System :: OS Independent',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
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
    config.add_data_dir('scikits/statsmodels/iolib/tests')
    config.add_data_dir('scikits/statsmodels/discrete/tests')
    config.add_data_dir('scikits/statsmodels/glm/tests')
    config.add_data_dir('scikits/statsmodels/regression/tests')
    config.add_data_dir('scikits/statsmodels/robust/tests')
    extradatafiles = [os.path.join(r,d) for r,ds,f in \
                      os.walk('scikits/statsmodels/datasets')
                      for d in f if not os.path.splitext(d)[1] in
                      ['.py', '.pyc']]
    for f in extradatafiles:
        config.add_data_files(f)
    tsaresultsfiles = [os.path.join(r,d) for r,ds,f in \
                       os.walk('scikits/statsmodels/tsa/tests/results') for \
                       d in f if not os.path.splitext(d)[1] in ['.py',
                           '.do', '.pyc', '.swp']]
    for f in tsaresultsfiles:
        config.add_data_files(f)

    if compile_cython:
        config.add_extension('tsa/kalmanf/kalman_loglike',
                sources = ['scikits/statsmodels/tsa/kalmanf/kalman_loglike.c'],
                include_dirs=[numpy.get_include()])

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
        #install_requires = 'numpy',
        namespace_packages = ['scikits'],
        packages = setuptools.find_packages(),
        include_package_data = True,
        test_suite="nose.collector",
        zip_safe = False, # the package can not run out of an .egg file bc of
                          # nose tests
        classifiers = classifiers)
