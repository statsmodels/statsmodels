import os
import fnmatch
from os.path import relpath, join as pjoin
import pkg_resources
from distutils.version import LooseVersion
from collections import defaultdict

from setuptools import Extension, find_packages, setup
from setuptools.dist import Distribution

from Cython.Distutils import build_ext
from Cython.Build import cythonize
from Cython import Tempita as tempita

import numpy
from numpy.distutils.misc_util import get_info
import importlib

import versioneer

REQUIREMENTS = {'numpy': '1.9',
                'scipy': '0.16',
                'pandas': '0.16.2',
                'patsy': '0.4.0'}

EXTRAS = {'build': ['cython>=0.24'],
          'install': ['cython>=0.24'],
          'develop': ['cython>=0.24'],
          'docs': ['sphinx>=1.3.5',
                   'nbconvert>=4.2.0',
                   'jupyter_client',
                   'ipykernel',
                   'matplotlib',
                   'nbformat>=4.0.1',
                   'numpydoc>=0.6.0',
                   'pandas-datareader']}

DISTNAME = 'statsmodels'
DESCRIPTION = 'Statistical computations and models for Python'
MAINTAINER = 'Skipper Seabold, Josef Perktold'
MAINTAINER_EMAIL = 'pystatsmodels@googlegroups.com'
URL = 'http://www.statsmodels.org/'
LICENSE = 'BSD License'
README = open(pjoin(os.getcwd(), 'README.rst')).read()
LONG_DESCRIPTION = README
DOWNLOAD_URL = ''

CLASSIFIERS = ['Development Status :: 4 - Beta',
               'Environment :: Console',
               'Programming Language :: Cython',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.3',
               'Programming Language :: Python :: 3.4',
               'Programming Language :: Python :: 3.5',
               'Operating System :: OS Independent',
               'Intended Audience :: End Users/Desktop',
               'Intended Audience :: Developers',
               'Intended Audience :: Science/Research',
               'Natural Language :: English',
               'License :: OSI Approved :: BSD License',
               'Topic :: Scientific/Engineering']

NUMPY_INCLUDES = [numpy.get_include()]
NUMPY_INCLUDES += [pkg_resources.resource_filename('numpy', 'core/include')]
NUMPY_MATH_LIBS = get_info('npymath')

# Determine whether to build the cython extensions with coverage
# measurement enabled.
CYTHON_COVERAGE = bool(os.environ.get('CYTHON_COVERAGE', True))
CYTHON_TRACE_NOGIL = str(int(CYTHON_COVERAGE))
if CYTHON_COVERAGE:
    print('Building with coverage for Cython code')
COMPILER_DIRECTIVES = {'linetrace': CYTHON_COVERAGE}

ADDITIONAL_PACKAGE_DATA = {
    'statsmodels.datasets.tests': ['*.zip'],
    'statsmodels.iolib.tests.results': ['*.dta'],
    'statsmodels.stats.tests.results': ['*.json'],
    'statsmodels.tsa.vector_ar.tests.results': ['*.npz'],
    'statsmodels.stats.tests': ['*.txt'],
    'statsmodels.stats.libqsturng': ['*.r', '*.txt', '*.dat'],
    'statsmodels.stats.libqsturng.tests': ['*.csv', '*.dat'],
    'statsmodels.tsa.vector_ar.data': ['*.dat'],
    'statsmodels.sandbox.regression.tests': ['*.dta', '*.csv']
}

problems = []
for pkg, req_ver in REQUIREMENTS.items():
    try:
        mod = importlib.import_module(pkg)
        version = LooseVersion(mod.__version__)
        if version < LooseVersion(req_ver):
            problems.append('{pkg}: {version}'.format(pkg=pkg,
                                                      version=version))
    except ImportError:
        pass
    if problems:
        raise RuntimeError('''
Some packages are older then the minimum required. setup will not force 
upgradecore PyData packages.  Please update or install in an environment
without old packages.  The identified packages were:
{problems}
'''.format(problems='\n'.join(problems)))

cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext

##############################################################################
# Extension Building
##############################################################################
ext_data = dict(
    _hamilton_filter={'source': 'statsmodels/tsa/regime_switching/_hamilton_filter.pyx.in',
                      'depends': [],
                      'include_dirs': [],
                      'sources': []},
    _kim_smoother={'source': 'statsmodels/tsa/regime_switching/_kim_smoother.pyx.in',
                   'depends': [],
                   'include_dirs': [],
                   'sources': []},
    _statespace={'source': 'statsmodels/tsa/statespace/_statespace.pyx.in',
                 'depends': ['statsmodels/src/capsule.h'],
                 'include_dirs': ['statsmodels/src'] + NUMPY_MATH_LIBS['include_dirs'],
                 'libraries': NUMPY_MATH_LIBS['libraries'],
                 'library_dirs': NUMPY_MATH_LIBS['library_dirs'],
                 'sources': []},
    linbin={'source': 'statsmodels/nonparametric/linbin.pyx'},
    _smoothers_lowess={'source': 'statsmodels/nonparametric/_smoothers_lowess.pyx'},
    kalman_loglike={'source': 'statsmodels/tsa/kalmanf/kalman_loglike.pyx',
                    'include_dirs': ['statsmodels/src'],
                    'depends': ['statsmodels/src/capsule.h']},
    _representation={'source': 'statsmodels/tsa/statespace/_representation.pyx.in',
                     'include_dirs': ['statsmodels/src'],
                     'libraries': NUMPY_MATH_LIBS['libraries'],
                     'library_dirs': NUMPY_MATH_LIBS['library_dirs'],
                     'sources': [],
                     'blas': True},
    _kalman_filter={'source': 'statsmodels/tsa/statespace/_kalman_filter.pyx.in',
                    'include_dirs': ['statsmodels/src'],
                    'libraries': NUMPY_MATH_LIBS['libraries'],
                    'library_dirs': NUMPY_MATH_LIBS['library_dirs'],
                    'sources': [],
                    'blas': True},
    _kalman_filter_conventional={'source': 'statsmodels/tsa/statespace/_filters/_conventional.pyx.in',
                                 'filename': '_conventional',
                                 'include_dirs': ['statsmodels/src'],
                                 'libraries': NUMPY_MATH_LIBS['libraries'],
                                 'library_dirs': NUMPY_MATH_LIBS['library_dirs'],
                                 'sources': [],
                                 'blas': True},
    _kalman_filter_inversions={'source': 'statsmodels/tsa/statespace/_filters/_inversions.pyx.in',
                               'filename': '_inversions',
                               'include_dirs': ['statsmodels/src'],
                               'libraries': NUMPY_MATH_LIBS['libraries'],
                               'library_dirs': NUMPY_MATH_LIBS['library_dirs'],
                               'sources': [],
                               'blas': True},
    _kalman_filter_univariate={'source': 'statsmodels/tsa/statespace/_filters/_univariate.pyx.in',
                               'filename': '_univariate',
                               'include_dirs': ['statsmodels/src'],
                               'libraries': NUMPY_MATH_LIBS['libraries'],
                               'library_dirs': NUMPY_MATH_LIBS['library_dirs'],
                               'sources': [],
                               'blas': True},
    _kalman_smoother={'source': 'statsmodels/tsa/statespace/_kalman_smoother.pyx.in',
                      'include_dirs': ['statsmodels/src'],
                      'libraries': NUMPY_MATH_LIBS['libraries'],
                      'library_dirs': NUMPY_MATH_LIBS['library_dirs'],
                      'sources': [],
                      'blas': True},
    _kalman_smoother_alternative={'source': 'statsmodels/tsa/statespace/_smoothers/_alternative.pyx.in',
                                  'filename': '_alternative',
                                  'include_dirs': ['statsmodels/src'],
                                  'libraries': NUMPY_MATH_LIBS['libraries'],
                                  'library_dirs': NUMPY_MATH_LIBS['library_dirs'],
                                  'sources': [],
                                  'blas': True},
    _kalman_smoother_classical={'source': 'statsmodels/tsa/statespace/_smoothers/_classical.pyx.in',
                                'filename': '_classical',
                                'include_dirs': ['statsmodels/src'],
                                'libraries': NUMPY_MATH_LIBS['libraries'],
                                'library_dirs': NUMPY_MATH_LIBS['library_dirs'],
                                'sources': [],
                                'blas': True},
    _kalman_smoother_conventional={'source': 'statsmodels/tsa/statespace/_smoothers/_conventional.pyx.in',
                                   'filename': '_conventional',
                                   'include_dirs': ['statsmodels/src'],
                                   'libraries': NUMPY_MATH_LIBS['libraries'],
                                   'library_dirs': NUMPY_MATH_LIBS['library_dirs'],
                                   'sources': [],
                                   'blas': True},
    _kalman_smoother_univariate={'source': 'statsmodels/tsa/statespace/_smoothers/_univariate.pyx.in',
                                 'filename': '_univariate',
                                 'include_dirs': ['statsmodels/src'],
                                 'libraries': NUMPY_MATH_LIBS['libraries'],
                                 'library_dirs': NUMPY_MATH_LIBS['library_dirs'],
                                 'sources': [],
                                 'blas': True},
    _kalman_simulation_smoother={'source': 'statsmodels/tsa/statespace/_simulation_smoother.pyx.in',
                                 'filename': '_simulation_smoother',
                                 'include_dirs': ['statsmodels/src'],
                                 'libraries': NUMPY_MATH_LIBS['libraries'],
                                 'library_dirs': NUMPY_MATH_LIBS['library_dirs'],
                                 'sources': [],
                                 'blas': True},
    _kalman_tools={'source': 'statsmodels/tsa/statespace/_tools.pyx.in',
                   'filename': '_tools',
                   'sources': [],
                   'blas': True},
)

define_macros = [('CYTHON_TRACE_NOGIL', CYTHON_TRACE_NOGIL)]
extensions = []
has_scipy_blas = False
try:
    from scipy.linalg import cython_blas

    has_scipy_blas = True
except ImportError:
    pass

for config in ext_data.values():
    blas_required = config.get('blas', False)
    if blas_required and not has_scipy_blas:
        continue
    source = config['source']
    if source.endswith('pyx.in'):
        with open(source, 'r') as templated:
            pyx_template = templated.read()
        pyx = tempita.sub(pyx_template)
        pyx_filename = source[:-3]
        with open(pyx_filename, 'w') as pyx_file:
            pyx_file.write(pyx)
        file_stats = os.stat(source)
        os.utime(pyx_filename, ns=(file_stats.st_atime_ns, file_stats.st_mtime_ns))
        source = pyx_filename

    name = source.replace('/', '.').replace('.pyx', '')
    include_dirs = config.get('include_dirs', []) + NUMPY_INCLUDES
    depends = config.get('depends', [])
    libraries = config.get('libraries', [])
    library_dirs = config.get('library_dirs', [])

    ext = Extension(name, [source],
                    include_dirs=include_dirs, depends=depends,
                    libraries=libraries, library_dirs=library_dirs,
                    define_macros=define_macros)
    extensions.append(ext)

extensions = cythonize(extensions, compiler_directives=COMPILER_DIRECTIVES)


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


data_files = defaultdict(list)
filetypes = ['*.csv', '*.txt', '*.dta']
for root, dirnames, filenames in os.walk(pjoin(os.getcwd(), 'statsmodels', 'datasets')):
    matches = []
    for filetype in filetypes:
        for filename in fnmatch.filter(filenames, filetype):
            matches.append(filename)
    if matches:
        data_files['.'.join(relpath(root).split(os.path.sep))] = filetypes
for root, dirnames, filenames in os.walk(pjoin(os.getcwd(), 'statsmodels')):
    if root.endswith('results'):
        data_files['.'.join(relpath(root).split(os.path.sep))] = filetypes

if os.path.exists('MANIFEST'):
    os.unlink('MANIFEST')

package_data = data_files
for path, filetypes in ADDITIONAL_PACKAGE_DATA.items():
    package_data[path].extend(filetypes)

packages = find_packages()
packages.append('statsmodels.tsa.vector_ar.data')

setup(name=DISTNAME,
      version=versioneer.get_version(),
      maintainer=MAINTAINER,
      ext_modules=extensions,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      classifiers=CLASSIFIERS,
      cmdclass=cmdclass,
      packages=packages,
      package_data=package_data,
      distclass=BinaryDistribution,
      include_package_data=False,  # True will install all files in repo
      extras_require=EXTRAS,
      zip_safe=False)
