"""
setuptools must be installed first. If you do not have setuptools installed
please download and install it from http://pypi.python.org/pypi/setuptools
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

curdir = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(curdir, "README.txt")).read()
CHANGES = open(os.path.join(curdir, "CHANGES.txt")).read()

DISTNAME = 'statsmodels'
DESCRIPTION = 'Statistical computations and models for use with SciPy'
LONG_DESCRIPTION = README + '\n\n' + CHANGES
MAINTAINER = 'Skipper Seabold, Josef Perktold'
MAINTAINER_EMAIL ='pystatsmodels@googlegroups.com'
URL = 'http://statsmodels.sourceforge.net/'
LICENSE = 'BSD License'
DOWNLOAD_URL = ''

MAJ = 0
MIN = 4
REV = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJ,MIN,REV)

classifiers = [ 'Development Status :: 4 - Beta',
              'Environment :: Console',
              'Programming Language :: Python :: 2.5',
              'Programming Language :: Python :: 2.6',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3.2',
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

def write_version_py(filename='statsmodels/version.py'):
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
            from statsmodels.version import git_revision as GIT_REVISION
            #print "debug import success GIT_REVISION", GIT_REVISION
        except ImportError:
            dowrite = False
            #changed: if we are not in a git repository then don't update version.py
##            raise ImportError("Unable to import git_revision. Try removing " \
##                              "statsmodels/version.py and the build directory " \
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

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    # 2.x
    from distutils.command.build_py import build_py


def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path,
                           namespace_packages = ['scikits'])

    config.add_subpackage('scikits')
    config.add_subpackage(DISTNAME)
    config.add_subpackage('scikits.statsmodels')
    config.add_data_files('scikits/__init__.py')
    config.add_data_files('docs/build/htmlhelp/statsmodelsdoc.chm',
                          'statsmodels/statsmodelsdoc.chm')

    config.set_options(
            ignore_setup_xxx_py = True,
            assume_default_configuration = True,
            delegate_options_to_subpackages = True,
            quiet = False,
            )

    return config

if __name__ == "__main__":
    write_version_py()
    setup(
          name = DISTNAME,
          version = VERSION,
          maintainer  = MAINTAINER,
          maintainer_email = MAINTAINER_EMAIL,
          description = DESCRIPTION,
          license = LICENSE,
          url = URL,
          download_url = DOWNLOAD_URL,
          long_description = LONG_DESCRIPTION,
          configuration = configuration,
          install_requires = ['pandas >= 0.7.0'],
          namespace_packages = ['scikits'],
          packages = setuptools.find_packages(),
          include_package_data = True,
          test_suite="nose.collector",
          zip_safe = False, # the package can not run out of an .egg file bc of
          # nose tests
          classifiers = classifiers,
          cmdclass={'build_py': build_py})
