from statsmodels.compat import iteritems
from statsmodels.compat.collections import Counter
from numpy.testing import assert_


def test_counter():
    #just check a basic example
    c = Counter('gallahad')
    res = [('a', 3), ('d', 1), ('g', 1), ('h', 1), ('l', 2)]
    msg = 'gallahad fails\n'+repr(sorted(iteritems(c)))
    assert_(sorted(iteritems(c)) == res, msg=msg)
