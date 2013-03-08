
from numpy.testing import assert_
from statsmodels.compatnp.collections import Counter


def test_counter():
    #just check a basic example
    c = Counter('gallahad')
    res = [('a', 3), ('d', 1), ('g', 1), ('h', 1), ('l', 2)]
    msg = 'gallahad fails\n'+repr(sorted(c.items()))
    assert_(sorted(c.items()) == res, msg=msg)
