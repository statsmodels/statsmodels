#!/usr/bin/env python
import sys
from os.path import dirname

def show_versions():
    print("\nINSTALLED VERSIONS")
    print("------------------")
    print("Python: %d.%d.%d.%s.%s" % sys.version_info[:])
    try:
        import os
        (sysname, nodename, release, version, machine) = os.uname()
        print("OS: %s %s %s %s" % (sysname, release, version,machine))
        print("byteorder: %s" % sys.byteorder)
        print("LC_ALL: %s" % os.environ.get('LC_ALL',"None"))
        print("LANG: %s" % os.environ.get('LANG',"None"))
    except:
        pass

    try:
        import statsmodels
        from statsmodels import version
        has_sm = True
    except ImportError:
        has_sm = False

    print('\nStatsmodels\n===========\n')
    if has_sm:
        print('Installed: %s (%s)' % (version.full_version,
                                      dirname(statsmodels.__file__)))
    else:
        print('Not installed')

    print("\nRequired Dependencies\n=====================\n")
    try:
        import Cython
        print("cython: %s (%s)" % (Cython.__version__,
                                   dirname(Cython.__file__)))
    except ImportError:
        print("cython: Not installed")

    try:
        import numpy
        print("numpy: %s (%s)" % (numpy.version.version,
                                  dirname(numpy.__file__)))
    except ImportError:
        print("numpy: Not installed")

    try:
        import scipy
        print("scipy: %s (%s)" % (scipy.version.version,
                                  dirname(scipy.__file__)))
    except ImportError:
        print("scipy: Not installed")

    try:
        import pandas
        print("pandas: %s (%s)" % (pandas.version.version,
                                   dirname(pandas.__file__)))
    except ImportError:
        print("pandas: Not installed")

    try:
        import patsy
        print("patsy: %s (%s)" % (patsy.__version__,
                                  dirname(patsy.__file__)))
    except ImportError:
        print("patsy: Not installed")

    print("\nOptional Dependencies\n=====================\n")

    try:
        import matplotlib as mpl
        print("matplotlib: %s (%s)" % (mpl.__version__, dirname(mpl.__file__)))
    except ImportError:
        print("matplotlib: Not installed")

    try:
        from cvxopt import info
        print("cvxopt: %s (%s)" % (info.version, dirname(info.__file__)))
    except ImportError:
        print("cvxopt: Not installed")

    print("\nDeveloper Tools\n================\n")

    try:
        import IPython
        print("IPython: %s (%s)" % (IPython.__version__,
                                    dirname(IPython.__file__)))
    except ImportError:
        print("IPython: Not installed")
    try:
        import jinja2
        print("    jinja2: %s (%s)" % (jinja2.__version__,
                                       dirname(jinja2.__file__)))
    except ImportError:
        print("    jinja2: Not installed")

    try:
        import sphinx
        print("sphinx: %s (%s)" % (sphinx.__version__,
                                   dirname(sphinx.__file__)))
    except ImportError:
        print("sphinx: Not installed")

    try:
        import pygments
        print("    pygments: %s (%s)" % (pygments.__version__,
                                         dirname(pygments.__file__)))
    except ImportError:
        print("    pygments: Not installed")

    try:
        import nose
        print("nose: %s (%s)" % (nose.__version__, dirname(nose.__file__)))
    except ImportError:
        print("nose: Not installed")

    try:
        import virtualenv
        print("virtualenv: %s (%s)" % (virtualenv.__version__,
                                       dirname(virtualenv.__file__)))
    except ImportError:
        print("virtualenv: Not installed")

    print("\n")

if __name__ == "__main__":
    show_versions()
