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

    print ("\nOptional Dependencies\n=====================\n")

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


    print("\n")

if __name__ == "__main__":
    show_versions()
