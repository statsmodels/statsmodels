"""
Much of the build system code was adapted from work done by the pandas
developers [1], which was in turn based on work done in pyzmq [2] and lxml [3].

[1] http://pandas.pydata.org
[2] http://zeromq.github.io/pyzmq/
[3] http://lxml.de/
"""

import os
from os.path import relpath, join as pjoin
import sys
import subprocess
import re
from distutils.version import StrictVersion, LooseVersion


# temporarily redirect config directory to prevent matplotlib importing
# testing that for writeable directory which results in sandbox error in
# certain easy_install versions
os.environ["MPLCONFIGDIR"] = "."

no_frills = (len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                                     sys.argv[1] in ('--help-commands',
                                                     'egg_info', '--version',
                                                     'clean')))

# try bootstrapping setuptools if it doesn't exist
try:
    import pkg_resources
    try:
        pkg_resources.require("setuptools>=0.6c5")
    except pkg_resources.VersionConflict:
        from ez_setup import use_setuptools
        use_setuptools(version="0.6c5")
    from setuptools import setup, Command, find_packages
    _have_setuptools = True
except ImportError:
    # no setuptools installed
    from distutils.core import setup, Command
    _have_setuptools = False

if _have_setuptools:
    setuptools_kwargs = {"zip_safe": False,
                         "test_suite": "nose.collector"}
else:
    setuptools_kwargs = {}
    if sys.version_info[0] >= 3:
        sys.exit("Need setuptools to install statsmodels for Python 3.x")


curdir = os.path.abspath(os.path.dirname(__file__))
README = open(pjoin(curdir, "README.rst")).read()
CYTHON_EXCLUSION_FILE = 'cythonize_exclusions.dat'

DISTNAME = 'statsmodels'
DESCRIPTION = 'Statistical computations and models for Python'
LONG_DESCRIPTION = README
MAINTAINER = 'Skipper Seabold, Josef Perktold'
MAINTAINER_EMAIL ='pystatsmodels@googlegroups.com'
URL = 'http://www.statsmodels.org/'
LICENSE = 'BSD License'
DOWNLOAD_URL = ''

# These imports need to be here; setuptools needs to be imported first.
from distutils.extension import Extension
from distutils.command.build import build
from distutils.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def build_extensions(self):
        numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')

        for ext in self.extensions:
            if (hasattr(ext, 'include_dirs') and
                    not numpy_incl in ext.include_dirs):
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)


def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                          os.path.join(cwd, 'tools', 'cythonize.py'),
                          'statsmodels'],
                         cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")


def init_cython_exclusion(filename):
    with open(filename, 'w') as f:
        pass


def append_cython_exclusion(path, filename):
    with open(filename, 'a') as f:
        f.write(path + "\n")


def strip_rc(version):
    return re.sub(r"rc\d+$", "", version)


def check_dependency_versions(min_versions):
    """
    Don't let pip/setuptools do this all by itself.  It's rude.

    For all dependencies, try to import them and check if the versions of
    installed dependencies match the minimum version requirements.  If
    installed but version too low, raise an error.  If not installed at all,
    return the correct ``setup_requires`` and ``install_requires`` arguments to
    be added to the setuptools kwargs.  This prevents upgrading installed
    dependencies like numpy (that should be an explicit choice by the user and
    never happen automatically), but make things work when installing into an
    empty virtualenv for example.

    """
    setup_requires = []
    install_requires = []

    try:
        from numpy.version import short_version as npversion
    except ImportError:
        setup_requires.append('numpy')
        install_requires.append('numpy')
    else:
        if not (LooseVersion(npversion) >= min_versions['numpy']):
            raise ImportError("Numpy version is %s. Requires >= %s" %
                              (npversion, min_versions['numpy']))

    try:
        import scipy
    except ImportError:
        install_requires.append('scipy')
    else:
        try:
            from scipy.version import short_version as spversion
        except ImportError:
            from scipy.version import version as spversion  # scipy 0.7.0
        if not (LooseVersion(spversion) >= min_versions['scipy']):
            raise ImportError("Scipy version is %s. Requires >= %s" %
                              (spversion, min_versions['scipy']))

    try:
        from pandas import __version__ as pversion
    except ImportError:
        install_requires.append('pandas')
    else:
        if not (LooseVersion(pversion) >= min_versions['pandas']):
            ImportError("Pandas version is %s. Requires >= %s" %
                        (pversion, min_versions['pandas']))

    try:
        from patsy import __version__ as patsy_version
    except ImportError:
        install_requires.append('patsy')
    else:
        # patsy dev looks like 0.1.0+dev
        pversion = re.match("\d*\.\d*\.\d*", patsy_version).group()
        if not (LooseVersion(pversion) >= min_versions['patsy']):
            raise ImportError("Patsy version is %s. Requires >= %s" %
                              (pversion, min_versions["patsy"]))

    return setup_requires, install_requires


MAJ = 0
MIN = 8
REV = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJ,MIN,REV)

classifiers = ['Development Status :: 4 - Beta',
               'Environment :: Console',
               'Programming Language :: Cython',
               'Programming Language :: Python :: 2.6',
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

def write_version_py(filename=pjoin(curdir, 'statsmodels/version.py')):
    cnt = "\n".join(["",
                    "# THIS FILE IS GENERATED FROM SETUP.PY",
                    "short_version = '%(version)s'",
                    "version = '%(version)s'",
                    "full_version = '%(full_version)s'",
                    "git_revision = '%(git_revision)s'",
                    "release = %(isrelease)s", "",
                    "if not release:",
                    "    version = full_version"])
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
        except ImportError:
            dowrite = False
            GIT_REVISION = "Unknown"
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]


    if dowrite:
        try:
            a = open(filename, 'w')
            a.write(cnt % {'version': VERSION,
                           'full_version' : FULLVERSION,
                           'git_revision' : GIT_REVISION,
                           'isrelease': str(ISRELEASED)})
        finally:
            a.close()


class CleanCommand(Command):
    """Custom distutils command to clean the .so and .pyc files."""

    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self._clean_me = []
        self._clean_trees = []
        self._clean_exclude = ["bspline_ext.c",
                               "bspline_impl.c"]

        for root, dirs, files in list(os.walk('statsmodels')):
            for f in files:
                if f in self._clean_exclude:
                    continue
                if os.path.splitext(f)[-1] in ('.pyc', '.so', '.o',
                                               '.pyo',
                                               '.pyd', '.c', '.orig'):
                    self._clean_me.append(pjoin(root, f))
            for d in dirs:
                if d == '__pycache__':
                    self._clean_trees.append(pjoin(root, d))

        for d in ('build',):
            if os.path.exists(d):
                self._clean_trees.append(d)

    def finalize_options(self):
        pass

    def run(self):
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except Exception:
                pass
        for clean_tree in self._clean_trees:
            try:
                import shutil
                shutil.rmtree(clean_tree)
            except Exception:
                pass


class CheckingBuildExt(build_ext):
    """Subclass build_ext to get clearer report if Cython is necessary."""

    def check_cython_extensions(self, extensions):
        for ext in extensions:
            for src in ext.sources:
                if not os.path.exists(src):
                    raise Exception("""Cython-generated file '%s' not found.
        Cython is required to compile statsmodels from a development branch.
        Please install Cython or download a source release of statsmodels.
                """ % src)

    def build_extensions(self):
        self.check_cython_extensions(self.extensions)
        build_ext.build_extensions(self)


class DummyBuildSrc(Command):
    """ numpy's build_src command interferes with Cython's build_ext.
    """
    user_options = []

    def initialize_options(self):
        self.py_modules_dict = {}

    def finalize_options(self):
        pass

    def run(self):
        pass


cmdclass = {'clean': CleanCommand,
            'build': build}

cmdclass["build_src"] = DummyBuildSrc
cmdclass["build_ext"] = CheckingBuildExt


# some linux distros require it
#NOTE: we are not currently using this but add it to Extension, if needed.
# libraries = ['m'] if 'win32' not in sys.platform else []

from numpy.distutils.misc_util import get_info

# Reset the cython exclusions file
init_cython_exclusion(CYTHON_EXCLUSION_FILE)

npymath_info = get_info("npymath")
ext_data = dict(
    kalman_loglike = {"name" : "statsmodels/tsa/kalmanf/kalman_loglike.c",
              "depends" : ["statsmodels/src/capsule.h"],
              "include_dirs": ["statsmodels/src"],
              "sources" : []},
    _hamilton_filter = {"name" : "statsmodels/tsa/regime_switching/_hamilton_filter.c",
              "depends" : [],
              "include_dirs": [],
              "sources" : []},
    _kim_smoother = {"name" : "statsmodels/tsa/regime_switching/_kim_smoother.c",
              "depends" : [],
              "include_dirs": [],
              "sources" : []},
    _statespace = {"name" : "statsmodels/tsa/statespace/_statespace.c",
              "depends" : ["statsmodels/src/capsule.h"],
              "include_dirs": ["statsmodels/src"] + npymath_info['include_dirs'],
              "libraries": npymath_info['libraries'],
              "library_dirs": npymath_info['library_dirs'],
              "sources" : []},
    linbin = {"name" : "statsmodels/nonparametric/linbin.c",
             "depends" : [],
             "sources" : []},
    _smoothers_lowess = {"name" : "statsmodels/nonparametric/_smoothers_lowess.c",
             "depends" : [],
             "sources" : []}
    )

statespace_ext_data = dict(
    _representation = {"name" : "statsmodels/tsa/statespace/_representation.c",
              "include_dirs": ['statsmodels/src'] + npymath_info['include_dirs'],
              "libraries": npymath_info['libraries'],
              "library_dirs": npymath_info['library_dirs'],
              "sources": []},
    _kalman_filter = {"name" : "statsmodels/tsa/statespace/_kalman_filter.c",
              "include_dirs": ['statsmodels/src'] + npymath_info['include_dirs'],
              "libraries": npymath_info['libraries'],
              "library_dirs": npymath_info['library_dirs'],
              "sources": []},
    _kalman_filter_conventional = {"name" : "statsmodels/tsa/statespace/_filters/_conventional.c",
              "filename": "_conventional",
              "include_dirs": ['statsmodels/src'] + npymath_info['include_dirs'],
              "libraries": npymath_info['libraries'],
              "library_dirs": npymath_info['library_dirs'],
              "sources": []},
    _kalman_filter_inversions = {"name" : "statsmodels/tsa/statespace/_filters/_inversions.c",
              "filename": "_inversions",
              "include_dirs": ['statsmodels/src'] + npymath_info['include_dirs'],
              "libraries": npymath_info['libraries'],
              "library_dirs": npymath_info['library_dirs'],
              "sources": []},
    _kalman_filter_univariate = {"name" : "statsmodels/tsa/statespace/_filters/_univariate.c",
              "filename": "_univariate",
              "include_dirs": ['statsmodels/src'] + npymath_info['include_dirs'],
              "libraries": npymath_info['libraries'],
              "library_dirs": npymath_info['library_dirs'],
              "sources": []},
    _kalman_smoother = {"name" : "statsmodels/tsa/statespace/_kalman_smoother.c",
              "include_dirs": ['statsmodels/src'] + npymath_info['include_dirs'],
              "libraries": npymath_info['libraries'],
              "library_dirs": npymath_info['library_dirs'],
              "sources": []},
    _kalman_smoother_alternative = {"name" : "statsmodels/tsa/statespace/_smoothers/_alternative.c",
              "filename": "_alternative",
              "include_dirs": ['statsmodels/src'] + npymath_info['include_dirs'],
              "libraries": npymath_info['libraries'],
              "library_dirs": npymath_info['library_dirs'],
              "sources": []},
    _kalman_smoother_classical = {"name" : "statsmodels/tsa/statespace/_smoothers/_classical.c",
              "filename": "_classical",
              "include_dirs": ['statsmodels/src'] + npymath_info['include_dirs'],
              "libraries": npymath_info['libraries'],
              "library_dirs": npymath_info['library_dirs'],
              "sources": []},
    _kalman_smoother_conventional = {"name" : "statsmodels/tsa/statespace/_smoothers/_conventional.c",
              "filename": "_conventional",
              "include_dirs": ['statsmodels/src'] + npymath_info['include_dirs'],
              "libraries": npymath_info['libraries'],
              "library_dirs": npymath_info['library_dirs'],
              "sources": []},
    _kalman_smoother_univariate = {"name" : "statsmodels/tsa/statespace/_smoothers/_univariate.c",
              "filename": "_univariate",
              "include_dirs": ['statsmodels/src'] + npymath_info['include_dirs'],
              "libraries": npymath_info['libraries'],
              "library_dirs": npymath_info['library_dirs'],
              "sources": []},
    _kalman_simulation_smoother = {"name" : "statsmodels/tsa/statespace/_simulation_smoother.c",
              "filename": "_simulation_smoother",
              "include_dirs": ['statsmodels/src'] + npymath_info['include_dirs'],
              "libraries": npymath_info['libraries'],
              "library_dirs": npymath_info['library_dirs'],
              "sources": []},
    _kalman_tools = {"name" : "statsmodels/tsa/statespace/_tools.c",
              "filename": "_tools",
              "sources": []},
)
try:
    from scipy.linalg import cython_blas
    ext_data.update(statespace_ext_data)
except ImportError:
    for name, data in statespace_ext_data.items():
        path = '.'.join([data["name"].split('.')[0], 'pyx.in'])
        append_cython_exclusion(path.replace('/', os.path.sep),
                                CYTHON_EXCLUSION_FILE)

extensions = []
for name, data in ext_data.items():
    data['sources'] = data.get('sources', []) + [data['name']]

    destdir = ".".join(os.path.dirname(data["name"]).split("/"))
    data.pop('name')

    filename = data.pop('filename', name)

    obj = Extension('%s.%s' % (destdir, filename), **data)

    extensions.append(obj)


def get_data_files():
    sep = os.path.sep
    # install the datasets
    data_files = {}
    root = pjoin(curdir, "statsmodels", "datasets")
    for i in os.listdir(root):
        if i is "tests":
            continue
        path = pjoin(root, i)
        if os.path.isdir(path):
            data_files.update({relpath(path, start=curdir).replace(sep, ".") : ["*.csv",
                                                                  "*.dta"]})
    # add all the tests and results files
    for r, ds, fs in os.walk(pjoin(curdir, "statsmodels")):
        r_ = relpath(r, start=curdir)
        if r_.endswith('results'):
            data_files.update({r_.replace(sep, ".") : ["*.csv",
                                                       "*.txt",
                                                       "*.dta"]})

    return data_files


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.unlink('MANIFEST')

    min_versions = {
        'numpy' : '1.6.2',
        'scipy' : '0.11',
        'pandas' : '0.13',
        'patsy' : '0.2.1',
                   }
    if sys.version_info[0] == 3 and sys.version_info[1] >= 3:
        # 3.3 needs numpy 1.7+
        min_versions.update({"numpy" : "1.7.0"})

    (setup_requires,
     install_requires) = check_dependency_versions(min_versions)

    if _have_setuptools:
        setuptools_kwargs['setup_requires'] = setup_requires
        setuptools_kwargs['install_requires'] = install_requires

        write_version_py()

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
    package_data.update({"statsmodels.stats.tests" : ["*.txt"]})

    package_data.update({"statsmodels.stats.libqsturng" :
                         ["*.r", "*.txt", "*.dat"]})
    package_data.update({"statsmodels.stats.libqsturng.tests" :
                         ["*.csv", "*.dat"]})
    package_data.update({"statsmodels.tsa.vector_ar.data" : ["*.dat"]})
    package_data.update({"statsmodels.tsa.vector_ar.data" : ["*.dat"]})
    # temporary, until moved:
    package_data.update({"statsmodels.sandbox.regression.tests" :
                         ["*.dta", "*.csv"]})

    #TODO: deal with this. Not sure if it ever worked for bdists
    #('docs/build/htmlhelp/statsmodelsdoc.chm',
    # 'statsmodels/statsmodelsdoc.chm')

    cwd = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(os.path.join(cwd, 'PKG-INFO')) and not no_frills:
        # Generate Cython sources, unless building from source release
        generate_cython()
    extras = {'docs': ['sphinx>=1.3.5',
                       'nbconvert>=4.2.0',
                       'jupyter_client',
                       'ipykernel',
                       'matplotlib',
                       'nbformat>=4.0.1',
                       'numpydoc>=0.6.0']}

    setup(name = DISTNAME,
          version = VERSION,
          maintainer = MAINTAINER,
          ext_modules = extensions,
          maintainer_email = MAINTAINER_EMAIL,
          description = DESCRIPTION,
          license = LICENSE,
          url = URL,
          download_url = DOWNLOAD_URL,
          long_description = LONG_DESCRIPTION,
          classifiers = classifiers,
          platforms = 'any',
          cmdclass = cmdclass,
          packages = packages,
          package_data = package_data,
          include_package_data=False,  # True will install all files in repo
          extras_require=extras,
          **setuptools_kwargs)
