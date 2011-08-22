"""
setuptools must be installed first. If you do not have setuptools installed
please download and install it from http://pypi.python.org/pypi/setuptools
"""

descr = """
Statsmodels is a Python package that provides a complement to scipy for
statistical computations including descriptive statistics and
estimation of statistical models.

scikits.statsmodels provides classes and functions for the estimation of
several categories of statistical models. These currently include linear
regression models, OLS, GLS, WLS and GLS with AR(p) errors, generalized
linear models for six distribution families, M-estimators for robust
linear models, and regression with discrete dependent variables, Logit,
Probit, MNLogit, Poisson, based on maximum likelihood estimators,
timeseries models, ARMA, AR and VAR. An extensive list of result statistics
are available for each estimation problem. Statsmodels also contains
descriptive statistics, a wide range of statistical tests, tools for density
estimation and more.

We welcome feedback on our mailing list http://groups.google.com/group/pystatsmodels.
Report problems on our bug tracker https://github.com/statsmodels/statsmodels/issues.

For updated versions between releases, we recommend our repository on github
https://github.com/statsmodels/statsmodels.

Main changes for 0.3.0
----------------------

*Changes that break backwards compatibility*

Added api.py for importing. So the new convention for importing is ::

import scikits.statsmodels.api as sm

Importing from modules directly now avoids unnecessary imports and increases
the import speed if a library or user only needs specific functions.

* sandbox/output.py -> iolib/table.py
* lib/io.py -> iolib/foreign.py (Now contains Stata .dta format reader)
* family -> families
* families.links.inverse -> families.links.inverse_power
* Datasets' Load class is now load function.
* regression.py -> regression/linear_model.py
* discretemod.py -> discrete/discrete_model.py
* rlm.py -> robust/robust_linear_model.py
* glm.py -> genmod/generalized_linear_model.py
* model.py -> base/model.py
* t() method -> tvalues attribute (t() still exists but raises a warning)

*main changes and additions*

* Numerous bugfixes.
* Time Series Analysis model (tsa)
  - Vector Autoregression Models VAR (tsa.VAR)
  - Autogressive Models AR (tsa.AR)
  - Autoregressive Moving Average Models ARMA (tsa.ARMA) :
      optionally uses Cython for Kalman Filtering
      use setup.py install with option --with-cython
  - Baxter-King band-pass filter (tsa.filters.baxter_king)
  - Hodrick-Prescott filter (tsa.filters.hpfilter)
  - Christiano-Fitzgerald filter (tsa.filters.cffilter)

* Improved maximum likelihood framework uses all available scipy.optimize solvers
* Refactor of the datasets sub-package.
* Added more datasets for examples.
* Removed RPy dependency for running the test suite.
* Refactored the test suite.
* Refactored codebase/directory structure.
* Support for offset and exposure in GLM.
* Removed data_weights argument to GLM.fit for Binomial models.
* New statistical tests, especially diagnostic and specification tests
* Multiple test correction
* General Method of Moment framework in sandbox
* Improved documentation
* and other additions


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
  -RLM.fitted_values -> RLM.fittedvalues
  -GLMResults.resid_dev -> GLMResults.resid_deviance


Python 3
--------

scikits.statsmodels has been ported and tested for Python 3.2. Python 3
version of the code can be obtained by running 2to3.py over the entire
statsmodels source. The numerical core of statsmodels worked almost without
changes, however there can be problems with data input and plotting.
The STATA file reader and writer in iolib.foreign has not been ported yet.
And there are still some problems with the matplotlib version for Python 3
that was used in testing. Running the test suite with Python 3.2 shows some
errors related to foreign and matplotlib.


Sandbox
-------

We are continuing to work on support for systems of equations models, panel data
models, time series analysis, and information and entropy econometrics in the
sandbox. This code is often merged into trunk as it becomes more robust.


Windows Help
------------
The source distribution for Windows includes a htmlhelp file (statsmodels.chm).
This can be opened from the python interpreter ::

>>> import scikits.statsmodels.api as sm
>>> sm.open_help()
"""
import os
import sys
import subprocess
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
MAINTAINER_EMAIL ='pystatsmodels@googlegroups.com'
URL = 'http://statsmodels.sourceforge.net/'
LICENSE = 'BSD License'
DOWNLOAD_URL = ''

MAJ = 0
MIN = 3
REV = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJ,MIN,REV)

classifiers = [ 'Development Status :: 4 - Beta',
              'Environment :: Console',
              'Programming Language :: Python :: 2.4',
              'Operating System :: OS Independent',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Topic :: Scientific/Engineering']

# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(" ".join(cmd), stdout = subprocess.PIPE, env=env,
                               shell=True).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

def write_version_py(filename='scikits/statsmodels/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    dowrite = True
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists(filename):
        # must be a source distribution, use existing version file
        try:
            from scikits.statsmodels.version import git_revision as GIT_REVISION
            print "debug import success GIT_REVISION", GIT_REVISION
        except ImportError:
            dowrite = False
            #changed: if we are not in a git repository then don't update version.py
##            raise ImportError("Unable to import git_revision. Try removing " \
##                              "scikits/statsmodels/version.py and the build directory " \
##                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]


    if dowrite:
        a = open(filename, 'w')
        try:
            a.write(cnt % {'version': VERSION,
                           'full_version' : FULLVERSION,
                           'git_revision' : GIT_REVISION,
                           'isrelease': str(ISRELEASED)})
        finally:
            a.close()

def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    #if os.path.fexists('MANIFEST'): os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path,
                           namespace_packages = ['scikits'],
                           name = DISTNAME,
                           version = VERSION,
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
    config.add_data_dir('scikits/statsmodels/sandbox/examples')
    config.add_data_dir('scikits/statsmodels/docs')
    config.add_data_dir('scikits/statsmodels/iolib/tests')
    config.add_data_dir('scikits/statsmodels/discrete/tests')
    config.add_data_dir('scikits/statsmodels/genmod/tests')
    config.add_data_dir('scikits/statsmodels/regression/tests')
    config.add_data_dir('scikits/statsmodels/robust/tests')
    config.add_data_dir('scikits/statsmodels/tsa/vector_ar/tests')
    config.add_data_dir('scikits/statsmodels/tsa/filters/tests')
    config.add_data_files('scikits/statsmodels/docs/build/htmlhelp/statsmodelsdoc.chm')
    config.add_data_files('scikits/statsmodels/iolib/tests/results/macrodata.npy')
    config.add_data_dir('scikits/statsmodels/nonparametric/tests')
    vardatafiles = [os.path.join(r,d) for r,ds,f in \
                    os.walk('scikits/statsmodels/tsa/vector_ar/data')
                    for d in f if not os.path.splitext(d)[1] in ['.py',
                    '.pyc']]
    for f in vardatafiles:
        config.add_data_files(f)
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
    kderesultsfiles = [os.path.join(r,d) for r,ds,f in \
                os.walk('scikits/statsmodels/nonparametric/tests/results') for \
                       d in f if not os.path.splitext(d)[1] in ['.py',
                           '.do', '.pyc', '.swp']]
    for f in kderesultsfiles:
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
    write_version_py()
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
