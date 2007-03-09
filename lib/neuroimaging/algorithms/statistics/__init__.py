"""
TODO
"""
__docformat__ = 'restructuredtext'


def test(level=1, verbosity=1, flags=[]):
    from neuroimaging.utils.testutils import set_flags
    set_flags(flags)
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
