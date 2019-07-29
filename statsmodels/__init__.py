
from statsmodels._version import get_versions

debug_warnings = False

if debug_warnings:
    import warnings
    from .compat import PY3

    warnings.simplefilter("default")
    # use the following to raise an exception for debugging specific warnings
    # warnings.filterwarnings("error", message=".*integer.*")
    if PY3:
        # ResourceWarning doesn't exist in python 2
        # we have currently many ResourceWarnings in the datasets on python 3.4
        warnings.simplefilter("ignore", ResourceWarning)  # noqa:F821


def test(*args, **kwargs):
    from .tools._testing import PytestTester
    tst = PytestTester(package_path=__file__)
    return tst(*args, **kwargs)


__version__ = get_versions()['version']
del get_versions
