
from ._version import get_versions

debug_warnings = False

if debug_warnings:
    import warnings

    warnings.simplefilter("default")
    # use the following to raise an exception for debugging specific warnings
    # warnings.filterwarnings("error", message=".*integer.*")


def test(*args, **kwargs):
    from .tools._testing import PytestTester
    tst = PytestTester(package_path=__file__)
    return tst(*args, **kwargs)


__version__ = get_versions()['version']
del get_versions
