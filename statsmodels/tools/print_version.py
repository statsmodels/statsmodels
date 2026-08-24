#!/usr/bin/env python
"""Print installed dependency versions for diagnostics"""

from functools import reduce
import os
from pathlib import Path
import platform
import sys


def safe_version(module, attr="__version__", *others):
    """
    Return a version attribute from a module, or a fallback message

    Parameters
    ----------
    module : module
        The imported module to inspect.
    attr : str or list of str, optional
        Name of the attribute (or chain of nested attribute names) to
        look up on ``module``, e.g., ``"__version__"`` or
        ``["version", "version"]``. Default is ``"__version__"``.
    *others : str
        Additional attribute names to try, in order, if ``attr`` is not
        found on ``module``.

    Returns
    -------
    str
        The value of the located version attribute, or
        ``"Cannot detect version"`` if none of the attributes exist.
    """
    if not isinstance(attr, list):
        attr = [attr]
    try:
        return reduce(getattr, [module, *attr])
    except AttributeError:
        if others:
            return safe_version(module, others[0], *others[1:])
        return "Cannot detect version"


def _show_versions_only():
    print("\nINSTALLED VERSIONS")
    print("------------------")
    print("Python: {:d}.{:d}.{:d}.{}.{}".format(*sys.version_info[:]))

    uname = platform.uname()
    sysname = uname.system
    release = uname.release
    version = uname.version
    machine = uname.machine
    print(f"OS: {sysname} {release} {version} {machine}")
    print(f"byteorder: {sys.byteorder}")
    print("LC_ALL: {}".format(os.environ.get("LC_ALL", "None")))
    print("LANG: {}".format(os.environ.get("LANG", "None")))
    try:
        import statsmodels

        has_sm = True
    except ImportError:
        has_sm = False

    print("\nstatsmodels\n===========\n")
    if has_sm:
        print(f"Installed: {safe_version(statsmodels)}")
    else:
        print("Not installed")

    print("\nRequired Dependencies\n=====================\n")
    try:
        import Cython

        print(f"cython: {safe_version(Cython)}")
    except ImportError:
        print("cython: Not installed")

    try:
        import numpy as np

        print(f"numpy: {safe_version(np)}")
    except ImportError:
        print("numpy: Not installed")

    try:
        import scipy

        print(f"scipy: {safe_version(scipy)}")
    except ImportError:
        print("scipy: Not installed")

    try:
        import pandas as pd

        print(f"pandas: {safe_version(pd)}")
    except ImportError:
        print("pandas: Not installed")

    try:
        import dateutil

        print(f"    dateutil: {safe_version(dateutil)}")
    except ImportError:
        print("    dateutil: not installed")

    try:
        import patsy

        print(f"patsy: {safe_version(patsy)}")
    except ImportError:
        print("patsy: Not installed")

    try:
        import formulaic

        print(f"formulaic: {safe_version(formulaic)}")
    except ImportError:
        print("formulaic: Not installed")

    print("\nOptional Dependencies\n=====================\n")

    try:
        import matplotlib as mpl

        print(f"matplotlib: {safe_version(mpl)}")
    except ImportError:
        print("matplotlib: Not installed")

    try:
        from cvxopt import info

        print("cvxopt: {}".format(safe_version(info, "version")))
    except ImportError:
        print("cvxopt: Not installed")

    try:
        import joblib

        print(f"joblib: {safe_version(joblib)} ")
    except ImportError:
        print("joblib: Not installed")

    print("\nDeveloper Tools\n================\n")

    try:
        import IPython

        print(f"IPython: {safe_version(IPython)}")
    except ImportError:
        print("IPython: Not installed")
    try:
        import jinja2

        print(f"    jinja2: {safe_version(jinja2)}")
    except ImportError:
        print("    jinja2: Not installed")

    try:
        import sphinx

        print(f"sphinx: {safe_version(sphinx)}")
    except ImportError:
        print("sphinx: Not installed")

    try:
        import pygments

        print(f"    pygments: {safe_version(pygments)}")
    except ImportError:
        print("    pygments: Not installed")

    try:
        import pytest

        print(f"pytest: {safe_version(pytest)} ({Path(pytest.__file__).parent})")
    except ImportError:
        print("pytest: Not installed")

    try:
        import virtualenv

        print(f"virtualenv: {safe_version(virtualenv)}")
    except ImportError:
        print("virtualenv: Not installed")

    print("\n")


def show_versions(show_dirs=True):
    """
    List the versions of statsmodels and any installed dependencies

    Parameters
    ----------
    show_dirs : bool, optional
        Flag indicating to show module locations. Default is True.

    """
    if not show_dirs:
        _show_versions_only()
        return
    print("\nINSTALLED VERSIONS")
    print("------------------")
    print("Python: {:d}.{:d}.{:d}.{}.{}".format(*sys.version_info[:]))
    uname = platform.uname()
    sysname = uname.system
    release = uname.release
    version = uname.version
    machine = uname.machine
    print(f"OS: {sysname} {release} {version} {machine}")
    print(f"byteorder: {sys.byteorder}")
    print("LC_ALL: {}".format(os.environ.get("LC_ALL", "None")))
    print("LANG: {}".format(os.environ.get("LANG", "None")))

    try:
        import statsmodels

        has_sm = True
    except ImportError:
        has_sm = False

    print("\nstatsmodels\n===========\n")
    if has_sm:
        print(
            f"Installed: {safe_version(statsmodels)} ({Path(statsmodels.__file__).parent})"
            )
    else:
        print("Not installed")

    print("\nRequired Dependencies\n=====================\n")
    try:
        import Cython

        print(f"cython: {safe_version(Cython)} ({Path(Cython.__file__).parent})")
    except ImportError:
        print("cython: Not installed")

    try:
        import numpy as np

        print(f"numpy: {safe_version(np)} ({Path(np.__file__).parent})")
    except ImportError:
        print("numpy: Not installed")

    try:
        import scipy

        print(f"scipy: {safe_version(scipy)} ({Path(scipy.__file__).parent})")
    except ImportError:
        print("scipy: Not installed")

    try:
        import pandas as pd

        print(
            "pandas: {} ({})".format(
                safe_version(pd, "__version__"),
                Path(pd.__file__).parent,
            )
        )
    except ImportError:
        print("pandas: Not installed")

    try:
        import dateutil

        print(
            f"    dateutil: {safe_version(dateutil)} ({Path(dateutil.__file__).parent})"
            )
    except ImportError:
        print("    dateutil: not installed")

    try:
        import patsy

        print(f"patsy: {safe_version(patsy)} ({Path(patsy.__file__).parent})")
    except ImportError:
        print("patsy: Not installed")

    print("\nOptional Dependencies\n=====================\n")

    try:
        import matplotlib as mpl

        print(f"matplotlib: {safe_version(mpl)} ({Path(mpl.__file__).parent})")
        print("    backend: {} ".format(mpl.rcParams["backend"]))
    except ImportError:
        print("matplotlib: Not installed")

    try:
        from cvxopt import info

        print(
            "cvxopt: {} ({})".format(
                safe_version(info, "version"), Path(info.__file__).parent
            )
        )
    except ImportError:
        print("cvxopt: Not installed")

    try:
        import joblib

        print(f"joblib: {safe_version(joblib)} ({Path(joblib.__file__).parent})")
    except ImportError:
        print("joblib: Not installed")

    print("\nDeveloper Tools\n================\n")

    try:
        import IPython

        print(f"IPython: {safe_version(IPython)} ({Path(IPython.__file__).parent})")
    except ImportError:
        print("IPython: Not installed")
    try:
        import jinja2

        print(f"    jinja2: {safe_version(jinja2)} ({Path(jinja2.__file__).parent})")
    except ImportError:
        print("    jinja2: Not installed")

    try:
        import sphinx

        print(f"sphinx: {safe_version(sphinx)} ({Path(sphinx.__file__).parent})")
    except ImportError:
        print("sphinx: Not installed")

    try:
        import pygments

        print(
            f"    pygments: {safe_version(pygments)} ({Path(pygments.__file__).parent})"
        )
    except ImportError:
        print("    pygments: Not installed")

    try:
        import pytest

        print(f"pytest: {safe_version(pytest)} ({Path(pytest.__file__).parent})")
    except ImportError:
        print("pytest: Not installed")

    try:
        import virtualenv

        print(f"virtualenv: {safe_version(virtualenv)} ({Path(virtualenv.__file__).parent})")
    except ImportError:
        print("virtualenv: Not installed")

    print("\n")


if __name__ == "__main__":
    show_versions()
