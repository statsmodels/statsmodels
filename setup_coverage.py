import os
from os.path import relpath, join as pjoin
import sys
import subprocess
import re
import pkg_resources
from distutils.version import LooseVersion

from setuptools import Command, Extension, find_packages, setup
from setuptools.dist import Distribution

from Cython.Distutils import build_ext
from Cython.Build import cythonize
from Cython import Tempita as tempita

import numpy

import versioneer

NUMPY_INCLUDES = [numpy.get_include()]
NUMPY_INCLUDES += [pkg_resources.resource_filename('numpy', 'core/include')]

# Determine whether to build the cython extensions with coverage
# measurement enabled.
CYTHON_COVERAGE = bool(os.environ.get('CYTHON_COVERAGE', True))
CYTHON_TRACE_NOGIL = str(int(CYTHON_COVERAGE))
if CYTHON_COVERAGE:
    print('Building with coverage for Cython code')

README = open(pjoin(os.getcwd(), "README.rst")).read()
DISTNAME = 'statsmodels'
DESCRIPTION = 'Statistical computations and models for Python'
LONG_DESCRIPTION = README
MAINTAINER = 'Skipper Seabold, Josef Perktold'
MAINTAINER_EMAIL = 'pystatsmodels@googlegroups.com'
URL = 'http://www.statsmodels.org/'
LICENSE = 'BSD License'
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

cmdclass=versioneer.get_cmdclass()
cmdclass = {'build_ext': build_ext}

from numpy.distutils.misc_util import get_info

npymath_info = get_info('npymath')

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
                 'include_dirs': ['statsmodels/src'] + npymath_info['include_dirs'],
                 'libraries': npymath_info['libraries'],
                 'library_dirs': npymath_info['library_dirs'],
                 'sources': []},
    linbin={'source': 'statsmodels/nonparametric/linbin.pyx'},
    _smoothers_lowess={'source': 'statsmodels/nonparametric/_smoothers_lowess.pyx'},
    kalman_loglike={'source': 'statsmodels/tsa/kalmanf/kalman_loglike.pyx',
                    'include_dirs': ['statsmodels/src'],
                    'depends': ['statsmodels/src/capsule.h']}
)

statespace_ext_data = dict(
    _representation={'source': 'statsmodels/tsa/statespace/_representation.pyx.in',
                     'include_dirs': ['statsmodels/src'],
                     'libraries': npymath_info['libraries'],
                     'library_dirs': npymath_info['library_dirs'],
                     'sources': []},
    _kalman_filter={'source': 'statsmodels/tsa/statespace/_kalman_filter.pyx.in',
                    'include_dirs': ['statsmodels/src'],
                    'libraries': npymath_info['libraries'],
                    'library_dirs': npymath_info['library_dirs'],
                    'sources': []},
    _kalman_filter_conventional={'source': 'statsmodels/tsa/statespace/_filters/_conventional.pyx.in',
                                 'filename': '_conventional',
                                 'include_dirs': ['statsmodels/src'],
                                 'libraries': npymath_info['libraries'],
                                 'library_dirs': npymath_info['library_dirs'],
                                 'sources': []},
    _kalman_filter_inversions={'source': 'statsmodels/tsa/statespace/_filters/_inversions.pyx.in',
                               'filename': '_inversions',
                               'include_dirs': ['statsmodels/src'],
                               'libraries': npymath_info['libraries'],
                               'library_dirs': npymath_info['library_dirs'],
                               'sources': []},
    _kalman_filter_univariate={'source': 'statsmodels/tsa/statespace/_filters/_univariate.pyx.in',
                               'filename': '_univariate',
                               'include_dirs': ['statsmodels/src'],
                               'libraries': npymath_info['libraries'],
                               'library_dirs': npymath_info['library_dirs'],
                               'sources': []},
    _kalman_smoother={'source': 'statsmodels/tsa/statespace/_kalman_smoother.pyx.in',
                      'include_dirs': ['statsmodels/src'],
                      'libraries': npymath_info['libraries'],
                      'library_dirs': npymath_info['library_dirs'],
                      'sources': []},
    _kalman_smoother_alternative={'source': 'statsmodels/tsa/statespace/_smoothers/_alternative.pyx.in',
                                  'filename': '_alternative',
                                  'include_dirs': ['statsmodels/src'],
                                  'libraries': npymath_info['libraries'],
                                  'library_dirs': npymath_info['library_dirs'],
                                  'sources': []},
    _kalman_smoother_classical={'source': 'statsmodels/tsa/statespace/_smoothers/_classical.pyx.in',
                                'filename': '_classical',
                                'include_dirs': ['statsmodels/src'],
                                'libraries': npymath_info['libraries'],
                                'library_dirs': npymath_info['library_dirs'],
                                'sources': []},
    _kalman_smoother_conventional={'source': 'statsmodels/tsa/statespace/_smoothers/_conventional.pyx.in',
                                   'filename': '_conventional',
                                   'include_dirs': ['statsmodels/src'],
                                   'libraries': npymath_info['libraries'],
                                   'library_dirs': npymath_info['library_dirs'],
                                   'sources': []},
    _kalman_smoother_univariate={'source': 'statsmodels/tsa/statespace/_smoothers/_univariate.pyx.in',
                                 'filename': '_univariate',
                                 'include_dirs': ['statsmodels/src'],
                                 'libraries': npymath_info['libraries'],
                                 'library_dirs': npymath_info['library_dirs'],
                                 'sources': []},
    _kalman_simulation_smoother={'source': 'statsmodels/tsa/statespace/_simulation_smoother.pyx.in',
                                 'filename': '_simulation_smoother',
                                 'include_dirs': ['statsmodels/src'],
                                 'libraries': npymath_info['libraries'],
                                 'library_dirs': npymath_info['library_dirs'],
                                 'sources': []},
    _kalman_tools={'source': 'statsmodels/tsa/statespace/_tools.pyx.in',
                   'filename': '_tools',
                   'sources': []},
)

macros = [('CYTHON_TRACE_NOGIL', CYTHON_TRACE_NOGIL)]

extensions = []

ext_data.update(statespace_ext_data)
for config in ext_data.values():
    source = config['source']
    if source.endswith('pyx.in'):
        with open(source, 'r') as templated:
            pyx_template = templated.read()
        pyx = tempita.sub(pyx_template)
        pyx_filename = source[:-3]
        with open(pyx_filename, "w") as pyx_file:
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
                    define_macros=macros)
    extensions.append(ext)

directives={'linetrace': CYTHON_COVERAGE}
extensions = cythonize(extensions, compiler_directives=directives)


# ext = Extension('statsmodels.tsa.kalmanf.kalman_loglike',
#                define_macros=macros)

class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


def get_data_files():
    sep = os.path.sep
    # install the datasets
    data_files = {}
    root = pjoin(os.getcwd(), "statsmodels", "datasets")
    for i in os.listdir(root):
        if i is "tests":
            continue
        path = pjoin(root, i)
        if os.path.isdir(path):
            data_files.update({relpath(path, start=os.getcwd()).replace(sep, "."): ["*.csv",
                                                                                    "*.dta"]})
    # add all the tests and results files
    for r, ds, fs in os.walk(pjoin(os.getcwd(), "statsmodels")):
        r_ = relpath(r, start=os.getcwd())
        if r_.endswith('results'):
            data_files.update({r_.replace(sep, "."): ["*.csv",
                                                      "*.txt",
                                                      "*.dta"]})

    return data_files


if os.path.exists('MANIFEST'):
    os.unlink('MANIFEST')

min_versions = {
    'numpy': '1.8',
    'scipy': '0.16',
    'pandas': '0.18',
    'patsy': '0.4',
}
if sys.version_info[0] == 3 and sys.version_info[1] >= 3:
    # 3.3 needs numpy 1.7+
    min_versions.update({"numpy": "1.9.0"})

# this adds *.csv and *.dta files in datasets folders
# and *.csv and *.txt files in test/results folders
package_data = get_data_files()
packages = find_packages()
packages.append("statsmodels.tsa.vector_ar.data")

package_data["statsmodels.datasets.tests"].append("*.zip")
package_data["statsmodels.iolib.tests.results"].append("*.dta")
package_data["statsmodels.stats.tests.results"].append("*.json")
package_data["statsmodels.tsa.vector_ar.tests.results"].append("*.npz")
# data files that don't follow the tests/results pattern. should fix.
package_data.update({"statsmodels.stats.tests": ["*.txt"]})

package_data.update({"statsmodels.stats.libqsturng":
                         ["*.r", "*.txt", "*.dat"]})
package_data.update({"statsmodels.stats.libqsturng.tests":
                         ["*.csv", "*.dat"]})
package_data.update({"statsmodels.tsa.vector_ar.data": ["*.dat"]})
package_data.update({"statsmodels.tsa.vector_ar.data": ["*.dat"]})
# temporary, until moved:
package_data.update({"statsmodels.sandbox.regression.tests":
                         ["*.dta", "*.csv"]})

extras = {'docs': ['sphinx>=1.3.5',
                   'nbconvert>=4.2.0',
                   'jupyter_client',
                   'ipykernel',
                   'matplotlib',
                   'nbformat>=4.0.1',
                   'numpydoc>=0.6.0',
                   'pandas-datareader']}

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
      packages=packages,
      package_data=package_data,
      distclass=BinaryDistribution,
      include_package_data=True,  # True will install all files in repo
      extras_require=extras,
      zip_safe=False)
