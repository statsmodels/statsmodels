"""
To build with coverage of Cython files
export SM_CYTHON_COVERAGE=1
python setup.py develop
pytest --cov=statsmodels statsmodels
"""
import fnmatch
import os
import sys
import shutil
from collections import defaultdict
from os.path import relpath, abspath, split, join as pjoin

import pkg_resources

try:
    # SM_FORCE_C is a testing shim to force setup to use C source files
    FORCE_C = int(os.environ.get('SM_FORCE_C', 0))
    if FORCE_C:
        raise ImportError('Force import error for testing')
    from Cython import Tempita
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext

    HAS_CYTHON = True
except ImportError:
    from setuptools.command.build_ext import build_ext

    HAS_CYTHON = False
from setuptools import Extension, find_packages, setup
from distutils.command.clean import clean
from setuptools.dist import Distribution

import versioneer

###############################################################################
# Key Values that Change Each Release
###############################################################################
SETUP_REQUIREMENTS = {'numpy': '1.11',  # released March 2016
                      'scipy': '0.18',  # released July 2016
                      }

REQ_NOT_MET_MSG = """
{0} is installed but older ({1}) than required ({2}). You must manually
upgrade {0} before installing or install into a fresh virtualenv.
"""
for key in SETUP_REQUIREMENTS:
    import importlib
    from distutils.version import LooseVersion
    req_ver = LooseVersion(SETUP_REQUIREMENTS[key])
    try:
        mod = importlib.import_module(key)
        ver = LooseVersion(mod.__version__)
        if ver < req_ver:
            raise RuntimeError(REQ_NOT_MET_MSG.format(key, ver, req_ver))
    except ImportError:
        pass
    except AttributeError:
        raise RuntimeError(REQ_NOT_MET_MSG.format(key, ver, req_ver))

INSTALL_REQUIREMENTS = SETUP_REQUIREMENTS.copy()
INSTALL_REQUIREMENTS.update({'pandas': '0.19',  # released October 2016
                             'patsy': '0.4.0',  # released July 2015
                             })

CYTHON_MIN_VER = '0.24'  # released Apr 2016

SETUP_REQUIRES = [k + '>=' + v for k, v in SETUP_REQUIREMENTS.items()]
INSTALL_REQUIRES = [k + '>=' + v for k, v in INSTALL_REQUIREMENTS.items()]

EXTRAS_REQUIRE = {'build': ['cython>=' + CYTHON_MIN_VER],
                  'develop': ['cython>=' + CYTHON_MIN_VER],
                  'docs': ['sphinx', 'nbconvert', 'jupyter_client',
                           'ipykernel', 'matplotlib', 'nbformat', 'numpydoc',
                           'pandas-datareader']}

###############################################################################
# Values that rarely change
###############################################################################
DISTNAME = 'statsmodels'
DESCRIPTION = 'Statistical computations and models for Python'
SETUP_DIR = split(abspath(__file__))[0]
with open(pjoin(SETUP_DIR, 'README.rst')) as readme:
    README = readme.read()
LONG_DESCRIPTION = README
MAINTAINER = 'Josef Perktold, Chad Fulton, Kerby Shedden'
MAINTAINER_EMAIL = 'pystatsmodels@googlegroups.com'
URL = 'https://www.statsmodels.org/'
LICENSE = 'BSD License'
DOWNLOAD_URL = ''
PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/statsmodels/statsmodels/issues',
    'Documentation': 'https://www.statsmodels.org/stable/index.html',
    'Source Code': 'https://github.com/statsmodels/statsmodels'
}

CLASSIFIERS = ['Development Status :: 4 - Beta',
               'Environment :: Console',
               'Programming Language :: Cython',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.4',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Operating System :: OS Independent',
               'Intended Audience :: End Users/Desktop',
               'Intended Audience :: Developers',
               'Intended Audience :: Science/Research',
               'Natural Language :: English',
               'License :: OSI Approved :: BSD License',
               'Topic :: Office/Business :: Financial',
               'Topic :: Scientific/Engineering']

FILES_TO_INCLUDE_IN_PACKAGE = ['LICENSE.txt', 'setup.cfg']

FILES_COPIED_TO_PACKAGE = []
for filename in FILES_TO_INCLUDE_IN_PACKAGE:
    if os.path.exists(filename):
        dest = os.path.join('statsmodels', filename)
        shutil.copy2(filename, dest)
        FILES_COPIED_TO_PACKAGE.append(dest)

ADDITIONAL_PACKAGE_DATA = {
    'statsmodels': FILES_TO_INCLUDE_IN_PACKAGE,
    'statsmodels.datasets.tests': ['*.zip'],
    'statsmodels.iolib.tests.results': ['*.dta'],
    'statsmodels.stats.tests.results': ['*.json'],
    'statsmodels.tsa.vector_ar.tests.results': ['*.npz', '*.dat'],
    'statsmodels.stats.tests': ['*.txt'],
    'statsmodels.stats.libqsturng': ['*.r', '*.txt', '*.dat'],
    'statsmodels.stats.libqsturng.tests': ['*.csv', '*.dat'],
    'statsmodels.sandbox.regression.tests': ['*.dta', '*.csv']
}

##############################################################################
# Extension Building
##############################################################################
CYTHON_COVERAGE = os.environ.get('SM_CYTHON_COVERAGE', False)
CYTHON_COVERAGE = CYTHON_COVERAGE in ('1', 'true', '"true"')
CYTHON_TRACE_NOGIL = str(int(CYTHON_COVERAGE))
if CYTHON_COVERAGE:
    print('Building with coverage for Cython code')
COMPILER_DIRECTIVES = {'linetrace': CYTHON_COVERAGE}
DEFINE_MACROS = [('CYTHON_TRACE_NOGIL', CYTHON_TRACE_NOGIL)]


exts = dict(
    _exponential_smoothers={'source': 'statsmodels/tsa/_exponential_smoothers.pyx'},  # noqa: E501
    _hamilton_filter={'source': 'statsmodels/tsa/regime_switching/_hamilton_filter.pyx.in'},  # noqa: E501
    _kim_smoother={'source': 'statsmodels/tsa/regime_switching/_kim_smoother.pyx.in'},  # noqa: E501
    _innovations={'source': 'statsmodels/tsa/innovations/_arma_innovations.pyx.in'},  # noqa: E501
    linbin={'source': 'statsmodels/nonparametric/linbin.pyx'},
    _smoothers_lowess={'source': 'statsmodels/nonparametric/_smoothers_lowess.pyx'},  # noqa: E501
    kalman_loglike={'source': 'statsmodels/tsa/kalmanf/kalman_loglike.pyx',
                    'include_dirs': ['statsmodels/src'],
                    'depends': ['statsmodels/src/capsule.h']})

statespace_exts = [
    'statsmodels/tsa/statespace/_initialization.pyx.in',
    'statsmodels/tsa/statespace/_representation.pyx.in',
    'statsmodels/tsa/statespace/_kalman_filter.pyx.in',
    'statsmodels/tsa/statespace/_filters/_conventional.pyx.in',
    'statsmodels/tsa/statespace/_filters/_inversions.pyx.in',
    'statsmodels/tsa/statespace/_filters/_univariate.pyx.in',
    'statsmodels/tsa/statespace/_filters/_univariate_diffuse.pyx.in',
    'statsmodels/tsa/statespace/_kalman_smoother.pyx.in',
    'statsmodels/tsa/statespace/_smoothers/_alternative.pyx.in',
    'statsmodels/tsa/statespace/_smoothers/_classical.pyx.in',
    'statsmodels/tsa/statespace/_smoothers/_conventional.pyx.in',
    'statsmodels/tsa/statespace/_smoothers/_univariate.pyx.in',
    'statsmodels/tsa/statespace/_smoothers/_univariate_diffuse.pyx.in',
    'statsmodels/tsa/statespace/_simulation_smoother.pyx.in',
    'statsmodels/tsa/statespace/_tools.pyx.in',
]


class CleanCommand(clean):
    def run(self):
        msg = """

python setup.py clean is not supported.

Use one of:

* `git clean -xdf` to clean all untracked files
* `git clean -Xdf` to clean untracked files ignored by .gitignore
"""
        print(msg)
        sys.exit(1)


class DeferredBuildExt(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def build_extensions(self):
        self._update_extensions()
        build_ext.build_extensions(self)

    def _update_extensions(self):
        import numpy
        from numpy.distutils.misc_util import get_info

        numpy_includes = [numpy.get_include()]
        extra_incl = pkg_resources.resource_filename('numpy', 'core/include')
        numpy_includes += [extra_incl]
        numpy_includes = list(set(numpy_includes))
        numpy_math_libs = get_info('npymath')

        for extension in self.extensions:
            if not hasattr(extension, 'include_dirs'):
                continue
            extension.include_dirs = list(set(extension.include_dirs +
                                              numpy_includes))
            if extension.name in EXT_REQUIRES_NUMPY_MATH_LIBS:
                extension.include_dirs += numpy_math_libs['include_dirs']
                extension.libraries += numpy_math_libs['libraries']
                extension.library_dirs += numpy_math_libs['library_dirs']


cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = DeferredBuildExt
cmdclass['clean'] = CleanCommand


def check_source(source_name):
    """Chooses C or pyx source files, and raises if C is needed but missing"""
    source_ext = '.pyx'
    if not HAS_CYTHON:
        source_name = source_name.replace('.pyx.in', '.c')
        source_name = source_name.replace('.pyx', '.c')
        source_ext = '.c'
        if not os.path.exists(source_name):
            msg = 'C source not found.  You must have Cython installed to ' \
                  'build if the C source files have not been generated.'
            raise IOError(msg)
    return source_name, source_ext


def process_tempita(source_name):
    """Runs pyx.in files through tempita is needed"""
    if source_name.endswith('pyx.in'):
        with open(source_name, 'r') as templated:
            pyx_template = templated.read()
        pyx = Tempita.sub(pyx_template)
        pyx_filename = source_name[:-3]
        with open(pyx_filename, 'w') as pyx_file:
            pyx_file.write(pyx)
        file_stats = os.stat(source_name)
        try:
            os.utime(pyx_filename, ns=(file_stats.st_atime_ns,
                                       file_stats.st_mtime_ns))
        except AttributeError:
            os.utime(pyx_filename, (file_stats.st_atime, file_stats.st_mtime))
        source_name = pyx_filename
    return source_name


EXT_REQUIRES_NUMPY_MATH_LIBS = []
extensions = []
for config in exts.values():
    uses_blas = True
    source, ext = check_source(config['source'])
    source = process_tempita(source)
    name = source.replace('/', '.').replace(ext, '')
    include_dirs = config.get('include_dirs', [])
    depends = config.get('depends', [])
    libraries = config.get('libraries', [])
    library_dirs = config.get('library_dirs', [])

    uses_numpy_libraries = config.get('numpy_libraries', False)
    if uses_blas or uses_numpy_libraries:
        EXT_REQUIRES_NUMPY_MATH_LIBS.append(name)

    ext = Extension(name, [source],
                    include_dirs=include_dirs, depends=depends,
                    libraries=libraries, library_dirs=library_dirs,
                    define_macros=DEFINE_MACROS)
    extensions.append(ext)

for source in statespace_exts:
    source, ext = check_source(source)
    source = process_tempita(source)
    name = source.replace('/', '.').replace(ext, '')

    EXT_REQUIRES_NUMPY_MATH_LIBS.append(name)
    ext = Extension(name, [source],
                    include_dirs=['statsmodels/src'], depends=[],
                    libraries=[], library_dirs=[],
                    define_macros=DEFINE_MACROS)
    extensions.append(ext)

if HAS_CYTHON:
    extensions = cythonize(extensions, compiler_directives=COMPILER_DIRECTIVES,
                           language_level=3)

##############################################################################
# Construct package data
##############################################################################
package_data = defaultdict(list)
filetypes = ['*.csv', '*.txt', '*.dta']
for root, _, filenames in os.walk(pjoin(os.getcwd(), 'statsmodels', 'datasets')):  # noqa: E501
    matches = []
    for filetype in filetypes:
        for filename in fnmatch.filter(filenames, filetype):
            matches.append(filename)
    if matches:
        package_data['.'.join(relpath(root).split(os.path.sep))] = filetypes
for root, _, _ in os.walk(pjoin(os.getcwd(), 'statsmodels')):
    if root.endswith('results'):
        package_data['.'.join(relpath(root).split(os.path.sep))] = filetypes

for path, filetypes in ADDITIONAL_PACKAGE_DATA.items():
    package_data[path].extend(filetypes)

if os.path.exists('MANIFEST'):
    os.unlink('MANIFEST')


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


setup(name=DISTNAME,
      version=versioneer.get_version(),
      maintainer=MAINTAINER,
      ext_modules=extensions,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      download_url=DOWNLOAD_URL,
      project_urls=PROJECT_URLS,
      long_description=LONG_DESCRIPTION,
      classifiers=CLASSIFIERS,
      platforms='any',
      cmdclass=cmdclass,
      packages=find_packages(),
      package_data=package_data,
      distclass=BinaryDistribution,
      include_package_data=False,  # True will install all files in repo
      setup_requires=SETUP_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      zip_safe=False,
      data_files=[('', ['LICENSE.txt', 'setup.cfg'])]
      )

# Clean-up copied files
for copy in FILES_COPIED_TO_PACKAGE:
    os.unlink(copy)
