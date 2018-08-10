import fnmatch
import os
from collections import defaultdict
from os.path import relpath, join as pjoin

import numpy
import pkg_resources
from Cython import Tempita as tempita
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import get_info
from setuptools import Extension, find_packages, setup
from setuptools.dist import Distribution

import versioneer

###############################################################################
# Key Values that Change Each Release
###############################################################################
SETUP_REQUIREMENTS = {'numpy': '1.11',  # released March 2016
                      'scipy': '0.18',  # released July 2016
                      }

INSTALL_REQUIREMENTS = SETUP_REQUIREMENTS.copy()
INSTALL_REQUIREMENTS.update({'pandas': '0.19',  # released October 2016
                             'patsy': '0.4.0',  # released July 2015
                             })

EXTRAS = {'build': ['cython>=0.24'],
          'install': ['cython>=0.24'],
          'develop': ['cython>=0.24'],
          'docs': ['sphinx',
                   'nbconvert',
                   'jupyter_client',
                   'ipykernel',
                   'matplotlib',
                   'nbformat',
                   'numpydoc',
                   'pandas-datareader']}

###############################################################################
# Key Values that Change Each Release
###############################################################################

DISTNAME = 'statsmodels'
DESCRIPTION = 'Statistical computations and models for Python'
README = open(pjoin(os.getcwd(), 'README.rst')).read()
LONG_DESCRIPTION = README
MAINTAINER = 'Josef Perktold, Chad Fulton, Kerby Shedden'
MAINTAINER_EMAIL = 'pystatsmodels@googlegroups.com'
URL = 'https://www.statsmodels.org/'
LICENSE = 'BSD License'
DOWNLOAD_URL = ''

CLASSIFIERS = ['Development Status :: 4 - Beta',
               'Environment :: Console',
               'Programming Language :: Cython',
               'Programming Language :: Python :: 2.7',
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

NUMPY_INCLUDES = [numpy.get_include()]
NUMPY_INCLUDES += [pkg_resources.resource_filename('numpy', 'core/include')]
NUMPY_MATH_LIBS = get_info('npymath')

# Determine whether to build the cython extensions with coverage
# measurement enabled.
ADDITIONAL_PACKAGE_DATA = {
    'statsmodels.datasets.tests': ['*.zip'],
    'statsmodels.iolib.tests.results': ['*.dta'],
    'statsmodels.stats.tests.results': ['*.json'],
    'statsmodels.tsa.vector_ar.tests.results': ['*.npz', '*.dat'],
    'statsmodels.stats.tests': ['*.txt'],
    'statsmodels.stats.libqsturng': ['*.r', '*.txt', '*.dat'],
    'statsmodels.stats.libqsturng.tests': ['*.csv', '*.dat'],
    'statsmodels.sandbox.regression.tests': ['*.dta', '*.csv']
}

cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext

##############################################################################
# Extension Building
##############################################################################
COMPILER_DIRECTIVES = {}

ext_data = dict(
    _exponential_smoothers={'source': 'statsmodels/tsa/_exponential_smoothers.pyx'},
    _hamilton_filter={'source': 'statsmodels/tsa/regime_switching/_hamilton_filter.pyx.in'},
    _kim_smoother={'source': 'statsmodels/tsa/regime_switching/_kim_smoother.pyx.in'},
    linbin={'source': 'statsmodels/nonparametric/linbin.pyx'},
    _smoothers_lowess={'source': 'statsmodels/nonparametric/_smoothers_lowess.pyx'},
    kalman_loglike={'source': 'statsmodels/tsa/kalmanf/kalman_loglike.pyx',
                    'include_dirs': ['statsmodels/src'],
                    'depends': ['statsmodels/src/capsule.h']},
    _initialization={'source': 'statsmodels/tsa/statespace/_initialization.pyx.in',
                     'include_dirs': ['statsmodels/src'],
                     'blas': True},
    _representation={'source': 'statsmodels/tsa/statespace/_representation.pyx.in',
                     'include_dirs': ['statsmodels/src'],
                     'blas': True},
    _kalman_filter={'source': 'statsmodels/tsa/statespace/_kalman_filter.pyx.in',
                    'filename': '_kalman_filter',
                    'include_dirs': ['statsmodels/src'],
                    'blas': True},
    _kalman_filter_conventional={
        'source': 'statsmodels/tsa/statespace/_filters/_conventional.pyx.in',
        'filename': '_conventional',
        'include_dirs': ['statsmodels/src'],
        'blas': True},
    _kalman_filter_inversions={'source': 'statsmodels/tsa/statespace/_filters/_inversions.pyx.in',
                               'filename': '_inversions',
                               'include_dirs': ['statsmodels/src'],
                               'blas': True},
    _kalman_filter_univariate={'source': 'statsmodels/tsa/statespace/_filters/_univariate.pyx.in',
                               'filename': '_univariate',
                               'include_dirs': ['statsmodels/src'],
                               'blas': True},
    _kalman_filter_univariate_diffuse={
        'source': 'statsmodels/tsa/statespace/_filters/_univariate_diffuse.pyx.in',
        'filename': '_univariate_diffuse',
        'include_dirs': ['statsmodels/src'],
        'blas': True},
    _kalman_smoother={'source': 'statsmodels/tsa/statespace/_kalman_smoother.pyx.in',
                      'include_dirs': ['statsmodels/src'],
                      'blas': True},
    _kalman_smoother_alternative={
        'source': 'statsmodels/tsa/statespace/_smoothers/_alternative.pyx.in',
        'filename': '_alternative',
        'include_dirs': ['statsmodels/src'],
        'blas': True},
    _kalman_smoother_classical={
        'source': 'statsmodels/tsa/statespace/_smoothers/_classical.pyx.in',
        'include_dirs': ['statsmodels/src'],
        'blas': True},
    _kalman_smoother_conventional={
        'source': 'statsmodels/tsa/statespace/_smoothers/_conventional.pyx.in',
        'include_dirs': ['statsmodels/src'],
        'blas': True},
    _kalman_smoother_univariate={
        'source': 'statsmodels/tsa/statespace/_smoothers/_univariate.pyx.in',
        'filename': '_univariate',
        'include_dirs': ['statsmodels/src'],
        'blas': True},
    _kalman_smoother_univariate_diffuse={
        'source': 'statsmodels/tsa/statespace/_smoothers/_univariate_diffuse.pyx.in',
        'filename': '_univariate',
        'include_dirs': ['statsmodels/src'],
        'blas': True},
    _kalman_simulation_smoother={
        'source': 'statsmodels/tsa/statespace/_simulation_smoother.pyx.in',
        'filename': '_simulation_smoother',
        'include_dirs': ['statsmodels/src'],
        'blas': True},
    _kalman_tools={'source': 'statsmodels/tsa/statespace/_tools.pyx.in',
                   'filename': '_tools',
                   'blas': True},
)

define_macros = []
extensions = []

for config in ext_data.values():
    uses_blas = True
    source = config['source']
    if source.endswith('pyx.in'):
        with open(source, 'r') as templated:
            pyx_template = templated.read()
        pyx = tempita.sub(pyx_template)
        pyx_filename = source[:-3]
        with open(pyx_filename, 'w') as pyx_file:
            pyx_file.write(pyx)
        file_stats = os.stat(source)
        try:
            os.utime(pyx_filename, ns=(file_stats.st_atime_ns, file_stats.st_mtime_ns))
        except AttributeError:
            os.utime(pyx_filename, (file_stats.st_atime, file_stats.st_mtime))
        source = pyx_filename

    name = source.replace('/', '.').replace('.pyx', '')
    include_dirs = config.get('include_dirs', []) + NUMPY_INCLUDES
    depends = config.get('depends', [])
    libraries = config.get('libraries', [])
    library_dirs = config.get('library_dirs', [])

    uses_numpy_libraries = config.get('numpy_libraries', False)
    if uses_blas or uses_numpy_libraries:
        libraries.extend(NUMPY_MATH_LIBS['libraries'])
        library_dirs.extend(NUMPY_MATH_LIBS['library_dirs'])

    ext = Extension(name, [source],
                    include_dirs=include_dirs, depends=depends,
                    libraries=libraries, library_dirs=library_dirs,
                    define_macros=define_macros)
    extensions.append(ext)

extensions = cythonize(extensions, compiler_directives=COMPILER_DIRECTIVES)


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


##############################################################################
# Construct package data
##############################################################################
package_data = defaultdict(list)
filetypes = ['*.csv', '*.txt', '*.dta']
for root, dirnames, filenames in os.walk(pjoin(os.getcwd(), 'statsmodels', 'datasets')):
    matches = []
    for filetype in filetypes:
        for filename in fnmatch.filter(filenames, filetype):
            matches.append(filename)
    if matches:
        package_data['.'.join(relpath(root).split(os.path.sep))] = filetypes
for root, dirnames, filenames in os.walk(pjoin(os.getcwd(), 'statsmodels')):
    if root.endswith('results'):
        package_data['.'.join(relpath(root).split(os.path.sep))] = filetypes

for path, filetypes in ADDITIONAL_PACKAGE_DATA.items():
    package_data[path].extend(filetypes)

if os.path.exists('MANIFEST'):
    os.unlink('MANIFEST')

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
      platforms='any',
      cmdclass=cmdclass,
      packages=find_packages(),
      package_data=package_data,
      distclass=BinaryDistribution,
      include_package_data=False,  # True will install all files in repo
      setup_requires=[k + '>=' + v for k, v in SETUP_REQUIREMENTS.items()],
      install_requires=[k + '>=' + v for k, v in INSTALL_REQUIREMENTS.items()],
      extras_require=EXTRAS,
      zip_safe=False,
      )
