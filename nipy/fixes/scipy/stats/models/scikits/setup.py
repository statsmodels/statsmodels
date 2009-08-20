#! /usr/bin/env python
# Last Change: Wed Nov 05 11:00 AM 2008 J

# Copyright (C) 2008 Cournapeau David <cournape@gmail.com>

descr   = """Example package.

This is a do nothing package, to show how to organize a scikit.
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

DISTNAME = 'scikits.statmodels'
DESCRIPTION = 'Statistical computations and models for use with SciPy'
LONG_DESCRIPTION = desc
MAINTAINER = ''
MAINAINER_EMAIL =''
URL = ''
LICENSE = ''
DOWNLOAD_URL = ''

MAJ = 0
MIN = 1
REV = 1
DEV = True

def build_ver_str():
    return '%d.%d.%d' % (MAJ,MIN,REV)

def fbuild_fver_str():
    if DEV:
        return build_ver_str() +'dev'
    else:
        return build_ver_str()

VERSION = build_ver_str()

def write_version():
    f = open(fname, "w")
    f.writelines("version = '%s'\n" % build_ver_str())
    f.writelines("dev = %s\n" % DEV)
    f.writelines("full_version = '%s'\n" % build_fver_str())
    f.close()

def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    write_version(os.path.join("scikits","models", "version.py"))
    if os.path.exists(os.path.join("docs", "src")):
        write_version(os.path.join("docs". "src", "models_version.py"))
    pkg_prefix_dir = os.path.join('scikits', 'statsmodels')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path,
                           namespace_packages = ['scikits'],
                           version = build_fver_str(),
                           maintainer  = MAINTAINER,
                           maintainer_email = MAINTAINER_EMAIL,
                           description = DESCRIPTION,
                           license = LICENSE,
                           url = URL,
                           download_url = DOWNLOAD_URL,
                           long_description = LONG_DESCRIPTION)

    config.options(
            ignore_setup_xxx_py = True,
            assume_default_configuration = True,
            delegate_optsion_to_subpackages = True,
            quiet = True,
            )

    return config

if __name__ == "__main__":

    setup(configuration = configuration,
        install_requires = 'numpy',
        namespace_packages = ['scikits'],
        packages = setuptools.find_packages(),
        include_package_data = True,
#        test_suite="tester", # what should this point to for models?  the tests folder?
# or should tester.py import the tests?
        zip_safe = True, # the package can run out of an .egg file
#FIXME: is the below correct?
        classifiers =
            [ 'Development Status :: 3 - Alpha',
              'Environment :: Console',
              'Programming Language :: Python :: 2.4',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: TBD',
              'Topic :: Scientific/Engineering'])
