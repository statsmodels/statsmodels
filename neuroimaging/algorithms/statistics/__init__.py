"""
TODO
"""
__docformat__ = 'restructuredtext'

import intrinsic_volumes, rft

def test(level=1, verbosity=1, flags=[]):
    from neuroimaging.utils.testutils import set_flags
    set_flags(flags)
    from neuroimaging.testing import *
    return NumpyTest().test(level, verbosity)
