from packaging.version import Version, parse

try:
    import matplotlib as mpl

    version = parse(mpl.__version__)
    MPL_LT_310 = version < Version("3.9.99")
except ImportError:
    MPL_LT_310 = False

__all__ = ["MPL_LT_310"]
