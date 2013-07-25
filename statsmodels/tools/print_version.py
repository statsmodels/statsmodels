#!/usr/bin/env python
import sys

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
        from statsmodels import version
        has_sm = True
    except ImportError:
        has_sm = False

    print('\nStatsmodels\n===========')
    if has_sm:
        print('\nInstalled: %s' % version.full_version)
    else:
        print('Not installed')

    print("\nRequired Dependencies\n=====================\n")
    try:
        import Cython
        print("cython: %s" % Cython.__version__)
    except ImportError:
        print("cython: Not installed")

    try:
        import numpy
        print("numpy: %s" % numpy.version.version)
    except ImportError:
        print("numpy: Not installed")

    try:
        import scipy
        print("scipy: %s" % scipy.version.version)
    except ImportError:
        print("scipy: Not installed")

    try:
        import pandas
        print("pandas: %s" % pandas.version.version)
    except ImportError:
        print("pandas: Not installed")

    try:
        import patsy
        print("patsy: %s" % patsy.__version__)
    except ImportError:
        print("patsy: Not installed")

    print("\nOptional Dependencies\n=====================\n")

    try:
        import matplotlib as mpl
        print("matplotlib: %s" % mpl.__version__)
    except ImportError:
        print("matplotlib: Not installed")

    try:
        from cvxopt import info
        print("cvxopt: %s" % info.version)
    except ImportError:
        print("cvxopt: Not installed")

    print("\nDeveloper Tools\n================\n")

    try:
        import IPython
        print("IPython: %s" % IPython.__version__)
    except ImportError:
        print("IPython: Not installed")
    try:
        import jinja2
        print("    jinja2: %s" % jinja2.__version__)
    except ImportError:
        print("    jinja2: Not installed")

    try:
        import sphinx
        print("sphinx: %s" % sphinx.__version__)
    except ImportError:
        print("sphinx: Not installed")

    try:
        import pygments
        print("    pygments: %s" % pygments.__version__)
    except ImportError:
        print("    pygments: Not installed")

    try:
        import nose
        print("nose: %s" % nose.__version__)
    except ImportError:
        print("nose: Not installed")

    try:
        import virtualenv
        print("virtualenv: %s" % virtualenv.__version__)
    except ImportError:
        print("virtualenv: Not installed")

    print("\n")

if __name__ == "__main__":
    show_versions()
