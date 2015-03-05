import sys
import os.path

from setuptools import setup, Command, find_packages
from setuptools.extension import Extension
from setuptools.command import build_ext as _build_ext

from Cython.Build import cythonize
from numpy.distutils.misc_util import get_info


npymath_info = get_info("npymath")


setup(
    name='statsmodels',
    version='0.7.0',

    maintainer='Skipper Seabold, Josef Perktold',
    maintainer_email='pystatsmodels@googlegroups.com',
    url='http://statsmodels.sourceforge.net/',
    description='Statistical computations and models for use with SciPy',
    license='BSD License',
    download_url='',
    platforms='any',

    long_description=open(os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "README.rst")
    ).read(),

    packages=find_packages() + ["statsmodels.tsa.vector_ar.data"],

    install_requires=[
        'numpy>=1.7.0',
        'scipy>=0.7.0',
        'pandas>=0.7.1',
        'patsy>=0.1.0',
    ],
    setup_requires=[
        'numpy>=1.7.0',
        'cython',
    ],
    tests_require=[
        'nose',
    ],
    test_suite='nose.collector',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.2',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering'
    ],

    ext_modules=cythonize([
        Extension(
            'statsmodels.tsa.kalmanf.kalman_loglike',
            depends=['statsmodels/src/capsule.h'],
            include_dirs=['statsmodels/src'] + npymath_info['include_dirs'],
            sources=['statsmodels/tsa/kalmanf/kalman_loglike.pyx']
        ),
        Extension(
            'statsmodels.nonparametric._smoothers_lowess',
            depends=[],
            include_dirs=['statsmodels/src'] + npymath_info['include_dirs'],
            sources=['statsmodels/nonparametric/_smoothers_lowess.pyx']
        ),
        Extension(
            'statsmodels.nonparametric.linbin',
            depends=[],
            include_dirs=['statsmodels/src'] + npymath_info['include_dirs'],
            sources=['statsmodels/nonparametric/linbin.pyx']
        ),
        Extension(
            'statsmodels.tsa.statespace._statespace',
            depends=['statsmodels/src/capsule.h'],
            include_dirs=['statsmodels/src'] + npymath_info['include_dirs'],
            libraries=npymath_info['libraries'],
            library_dirs=npymath_info['library_dirs'],
            sources=['statsmodels/tsa/statespace/_statespace.pyx']
        ),
    ]),

    package_data={
        'statsmodels.datasets.anes96': ['*.csv', '*.dta'],
        'statsmodels.datasets.cancer': ['*.csv', '*.dta'],
        'statsmodels.datasets.ccard': ['*.csv', '*.dta'],
        'statsmodels.datasets.co2': ['*.csv', '*.dta'],
        'statsmodels.datasets.committee': ['*.csv', '*.dta'],
        'statsmodels.datasets.copper': ['*.csv', '*.dta'],
        'statsmodels.datasets.cpunish': ['*.csv', '*.dta'],
        'statsmodels.datasets.elnino': ['*.csv', '*.dta'],
        'statsmodels.datasets.engel': ['*.csv', '*.dta'],
        'statsmodels.datasets.fair': ['*.csv', '*.dta'],
        'statsmodels.datasets.fertility': ['*.csv', '*.dta'],
        'statsmodels.datasets.grunfeld': ['*.csv', '*.dta'],
        'statsmodels.datasets.heart': ['*.csv', '*.dta'],
        'statsmodels.datasets.longley': ['*.csv', '*.dta'],
        'statsmodels.datasets.macrodata': ['*.csv', '*.dta'],
        'statsmodels.datasets.modechoice': ['*.csv', '*.dta'],
        'statsmodels.datasets.nile': ['*.csv', '*.dta'],
        'statsmodels.datasets.randhie': ['*.csv', '*.dta'],
        'statsmodels.datasets.scotland': ['*.csv', '*.dta'],
        'statsmodels.datasets.spector': ['*.csv', '*.dta'],
        'statsmodels.datasets.stackloss': ['*.csv', '*.dta'],
        'statsmodels.datasets.star98': ['*.csv', '*.dta'],
        'statsmodels.datasets.statecrime': ['*.csv', '*.dta'],
        'statsmodels.datasets.strikes': ['*.csv', '*.dta'],
        'statsmodels.datasets.sunspots': ['*.csv', '*.dta'],
        'statsmodels.datasets.tests': ['*.csv', '*.dta', '*.zip'],
        'statsmodels.discrete.tests.results': ['*.csv', '*.txt'],
        'statsmodels.duration.tests.results': ['*.csv', '*.txt'],
        'statsmodels.emplike.tests.results': ['*.csv', '*.txt'],
        'statsmodels.genmod.tests.results': ['*.csv', '*.txt'],
        'statsmodels.iolib.tests.results': ['*.csv', '*.txt', '*.dta'],
        'statsmodels.nonparametric.tests.results': ['*.csv', '*.txt'],
        'statsmodels.regression.tests.results': ['*.csv', '*.txt'],
        'statsmodels.robust.tests.results': ['*.csv', '*.txt'],
        'statsmodels.sandbox.regression.tests': ['*.dta', '*.csv'],
        'statsmodels.stats.libqsturng': ['*.r', '*.txt', '*.dat'],
        'statsmodels.stats.libqsturng.tests': ['*.csv', '*.dat'],
        'statsmodels.stats.tests': ['*.txt'],
        'statsmodels.stats.tests.results': ['*.csv', '*.txt', '*.json'],
        'statsmodels.tests.results': ['*.csv', '*.txt'],
        'statsmodels.tools.tests.results': ['*.csv', '*.txt'],
        'statsmodels.tsa.filters.tests.results': ['*.csv', '*.txt'],
        'statsmodels.tsa.statespace.tests.results': ['*.csv', '*.txt'],
        'statsmodels.tsa.tests.results': ['*.csv', '*.txt'],
        'statsmodels.tsa.vector_ar.data': ['*.dat'],
        'statsmodels.tsa.vector_ar.tests.results': ['*.csv', '*.txt', '*.npz']
    },
    include_package_data=False,
    zip_safe=False,
)
