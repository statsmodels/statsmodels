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

# temporarily redirect config directory to prevent matplotlib importing
# testing that for writeable directory which results in sandbox error in
# certain easy_install versions
os.environ["MPLCONFIGDIR"] = "."


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

setuptools_kwargs = {}
if sys.version_info[0] >= 3:
    setuptools_kwargs = {'zip_safe': False}

    if not _have_setuptools:
        sys.exit("need setuptools/distribute for Py3k"
                 "\n$ pip install distribute")
else:
    setuptools_kwargs = {
        'install_requires': [],
        'zip_safe': False,
    }

if not _have_setuptools:
    setuptools_kwargs = {}

curdir = os.path.abspath(os.path.dirname(__file__))
README = open(pjoin(curdir, "README.txt")).read()

DISTNAME = 'statsmodels'
DESCRIPTION = 'Statistical computations and models for use with SciPy'
LONG_DESCRIPTION = README
MAINTAINER = 'Skipper Seabold, Josef Perktold'
MAINTAINER_EMAIL ='pystatsmodels@googlegroups.com'
URL = 'http://statsmodels.sourceforge.net/'
LICENSE = 'BSD License'
DOWNLOAD_URL = ''

from distutils.extension import Extension
from distutils.command.build import build
from distutils.command.sdist import sdist
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


def strip_rc(version):
    return re.sub(r"rc\d+$", "", version)


def check_dependency_versions(min_versions):
    """
    Don't let setuptools do this. It's rude.

    Just makes sure it can import the packages and if not, stops the build
    process.
    """
    from distutils.version import StrictVersion
    try:
        from numpy.version import short_version as npversion
    except ImportError:
        raise ImportError("statsmodels requires numpy")
    try:
        from scipy.version import short_version as spversion
    except ImportError:
        try: # scipy 0.7.0
            from scipy.version import version as spversion
        except ImportError:
            raise ImportError("statsmodels requires scipy")
    try:
        from pandas.version import short_version as pversion
    except ImportError:
        raise ImportError("statsmodels requires pandas")
    try:
        from patsy import __version__ as patsy_version
    except ImportError:
        raise ImportError("statsmodels requires patsy. http://patsy.readthedocs.org")

    try:
        assert StrictVersion(strip_rc(npversion)) >= min_versions['numpy']
    except AssertionError:
        raise ImportError("Numpy version is %s. Requires >= %s" %
                (npversion, min_versions['numpy']))
    try:
        assert StrictVersion(strip_rc(spversion)) >= min_versions['scipy']
    except AssertionError:
        raise ImportError("Scipy version is %s. Requires >= %s" %
                (spversion, min_versions['scipy']))
    try:
        assert StrictVersion(strip_rc(pversion)) >= min_versions['pandas']
    except AssertionError:
        raise ImportError("Pandas version is %s. Requires >= %s" %
                (pversion, min_versions['pandas']))

    try: # patsy dev looks like 0.1.0+dev
        pversion = re.match("\d*\.\d*\.\d*", patsy_version).group()
        assert StrictVersion(pversion) >= min_versions['patsy']
    except AssertionError:
        raise ImportError("Patsy version is %s. Requires >= %s" %
                (pversion, min_versions["patsy"]))


MAJ = 0
MIN = 6
REV = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJ,MIN,REV)

classifiers = [ 'Development Status :: 4 - Beta',
              'Environment :: Console',
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
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]


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

ext_data = dict(
        kalman_loglike = {"name" : "statsmodels/tsa/kalmanf/kalman_loglike.c",
                  "depends" : ["statsmodels/tsa/kalmanf/capsule.h"],
                  "sources" : []},

        linbin = {"name" : "statsmodels/nonparametric/linbin.c",
                 "depends" : [],
                 "sources" : []},
        _smoothers_lowess = {"name" : "statsmodels/nonparametric/_smoothers_lowess.c",
                 "depends" : [],
                 "sources" : []}
        )


extensions = []
for name, data in ext_data.items():
    data['sources'] = data.get('sources', []) + [data['name']]

    destdir = ".".join(os.path.dirname(data["name"]).split("/"))
    data.pop('name')

    obj = Extension('%s.%s' % (destdir, name), **data)

    extensions.append(obj)


if _have_setuptools:
    setuptools_kwargs["test_suite"] = "nose.collector"


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
                                                       "*.txt"]})

    return data_files


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.unlink('MANIFEST')

    min_versions = {
        'numpy' : '1.4.0',
        'scipy' : '0.7.0',
        'pandas' : '0.7.1',
        'patsy' : '0.1.0',
                   }
    if sys.version_info[0] == 3 and sys.version_info[1] >= 3:
        # 3.3 needs numpy 1.7+
        min_versions.update({"numpy" : "1.7.0b2"})

    if not (len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
            sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                            'clean'))):
        check_dependency_versions(min_versions)
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
    if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
        # Generate Cython sources, unless building from source release
        generate_cython()

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
          **setuptools_kwargs)
