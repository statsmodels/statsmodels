"""
Much of the build system code was adapted from work done by the pandas
developers [1], which was in turn based on work done in pyzmq [2] and lxml [3].

[1] http://pandas.pydata.org
[2] http://zeromq.github.io/pyzmq/
[3] http://lxml.de/
"""

import os
from os.path import splitext, basename, join as pjoin
import sys
import subprocess
import re

# may need to work around setuptools bug by providing a fake Pyrex
try:
    import Cython
    sys.path.insert(0, pjoin(os.path.dirname(__file__), "fake_pyrex"))
except ImportError:
    pass

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
    setuptools_kwargs = {'use_2to3': True,
                         'zip_safe': False,
                         #'use_2to3_exclude_fixers': [],
                         }
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

try:
    from Cython.Distutils import build_ext as _build_ext
    # from Cython.Distutils import Extension # to get pyrex debugging symbols
    cython = True
except ImportError:
    cython = False


class build_ext(_build_ext):
    def build_extensions(self):
        numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')

        for ext in self.extensions:
            if (hasattr(ext, 'include_dirs') and
                    not numpy_incl in ext.include_dirs):
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)


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
        from pandas.version import version as pversion
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
        #NOTE: not sure how robust this regex is but it at least allows
        # double digit version numbering
        pversion = re.match("\d*\.\d*\.\d*", pversion).group()
        assert StrictVersion(pversion) >= min_versions['pandas']
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
MIN = 5
REV = 0
ISRELEASED = True
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

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    # 2.x
    from distutils.command.build_py import build_py


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


class CheckSDist(sdist):
    """Custom sdist that ensures Cython has compiled all pyx files to c."""

    _pyxfiles = ['statsmodels/nonparametric/linbin.pyx',
                 'statsmodels/nonparametric/_smoothers_lowess.pyx',
                 'statsmodels/tsa/kalmanf/kalman_loglike.pyx']

    def initialize_options(self):
        sdist.initialize_options(self)

        '''
        self._pyxfiles = []
        for root, dirs, files in os.walk('statsmodels'):
            for f in files:
                if f.endswith('.pyx'):
                    self._pyxfiles.append(pjoin(root, f))
        '''

    def run(self):
        if 'cython' in cmdclass:
            self.run_command('cython')
        else:
            for pyxfile in self._pyxfiles:
                cfile = pyxfile[:-3] + 'c'
                msg = "C-source file '%s' not found." % (cfile) +\
                    " Run 'setup.py cython' before sdist."
                assert os.path.isfile(cfile), msg
        sdist.run(self)


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


class CythonCommand(build_ext):
    """Custom distutils command subclassed from Cython.Distutils.build_ext
    to compile pyx->c, and stop there. All this does is override the
    C-compile method build_extension() with a no-op."""
    def build_extension(self, ext):
        pass


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
            'build': build,
            'sdist': CheckSDist}

if cython:
    suffix = ".pyx"
    cmdclass["build_ext"] = CheckingBuildExt
    cmdclass["cython"] = CythonCommand
else:
    suffix = ".c"
    cmdclass["build_src"] = DummyBuildSrc
    cmdclass["build_ext"] = CheckingBuildExt

lib_depends = []

def srcpath(name=None, suffix='.pyx', subdir='src'):
    return pjoin('statsmodels', subdir, name + suffix)

if suffix == ".pyx":
    lib_depends = [srcpath(f, suffix=".pyx") for f in lib_depends]
else:
    lib_depends = []

common_include = []

# some linux distros require it
libraries = ['m'] if 'win32' not in sys.platform else []

ext_data = dict(
        kalman_loglike = {"pyxfile" : "tsa/kalmanf/kalman_loglike",
                  "depends" : [],
                  "sources" : []},

        linbin = {"pyxfile" : "nonparametric/linbin",
                 "depends" : [],
                 "sources" : []},
        _smoothers_lowess = {"pyxfile" : "nonparametric/_smoothers_lowess",
                 "depends" : [],
                 "sources" : []}
        )

def pxd(name):
    return os.path.abspath(pjoin('pandas', name + '.pxd'))

extensions = []
for name, data in ext_data.items():
    sources = [srcpath(data['pyxfile'], suffix=suffix, subdir='')]
    pxds = [pxd(x) for x in data.get('pxdfiles', [])]
    destdir = ".".join(os.path.dirname(data["pyxfile"]).split("/"))
    if suffix == '.pyx' and pxds:
        sources.extend(pxds)

    sources.extend(data.get('sources', []))

    include = data.get('include', common_include)

    obj = Extension('statsmodels.%s.%s' % (destdir, name),
                    sources=sources,
                    depends=data.get('depends', []),
                    include_dirs=include)

    extensions.append(obj)

if suffix == '.pyx' and 'setuptools' in sys.modules:
    # undo dumb setuptools bug clobbering .pyx sources back to .c
    for ext in extensions:
        if ext.sources[0].endswith('.c'):
            root, _ = os.path.splitext(ext.sources[0])
            ext.sources[0] = root + suffix

if _have_setuptools:
    setuptools_kwargs["test_suite"] = "nose.collector"

from os.path import relpath

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
            data_files.update({relpath(path).replace(sep, ".") : ["*.csv",
                                                                  "*.dta"]})
    # add all the tests and results files
    for r, ds, fs in os.walk(pjoin(curdir, "statsmodels")):
        if r.endswith('results') and 'sandbox' not in r:
            data_files.update({relpath(r).replace(sep, ".") : ["*.csv",
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
    # the next two are in the sdist, but I don't manage to get them installed
    package_data.update({"statsmodels.stats.libqstrung" :
                         ["*.r", "*.txt", "*.dat"]})
    package_data.update({"statsmodels.stats.libqstrung.tests" :
                         ["*.csv", "*.dat"]})
    package_data.update({"statsmodels.tsa.vector_ar.data" : ["*.dat"]})
    package_data.update({"statsmodels.tsa.vector_ar.data" : ["*.dat"]})
    # Why are we installing this stuff?

    #TODO: deal with this. Not sure if it ever worked for bdists
    #('docs/build/htmlhelp/statsmodelsdoc.chm',
    # 'statsmodels/statsmodelsdoc.chm')

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
          include_package_data=True,
          **setuptools_kwargs)
