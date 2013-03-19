# -*- coding: utf-8 -*-
"""

Created on Mon Mar 18 15:48:23 2013
Author: Josef Perktold

"""

import numpy as np
from scipy import optimize

# based on scipy.stats.distributions._ppf_single_call
def brentq_expanding(func, low=None, upp=None, args=(), xtol=1e-5,
                     start_low=None, start_upp=None, increasing=None,
                     max_it=100, maxiter_bq=100):
    #assumes monotonically increasing ``func``

    left, right = low, upp  #alias

    # start_upp first because of possible sl = -1 > upp
    if upp is not None:
        su = upp
    elif start_upp is not None:
        su = start_upp
        if start_upp < 0:
            print "raise ValueError('start_upp needs to be positive')"
    else:
        su = 1
        start_upp = 1


    if low is not None:
        sl = low
    elif start_low is not None:
        sl = start_low
        if start_low > 0:
            print "raise ValueError('start_low needs to be negative')"
    else:
        sl = min(-1, su - 1)
        start_low = sl

    # need sl < su
    if upp is None:
        su = max(su, sl + 1)


    # increasing or not ?
    if ((low is None) or (upp is None)) and increasing is None:
        assert sl < su
        f_low = func(sl, *args)
        f_upp = func(su, *args)
        increasing = (f_low < f_upp)

    print 'low, upp', low, upp
    print 'increasing', increasing
    print 'sl, su', sl, su


    start_low, start_upp = sl, su
    if not increasing:
        start_low, start_upp =  start_upp, start_low
        left, right = right, left

    #max_it = 200
    n_it = 0
    factor = 10.
    if left is None: # i.e. self.a = -inf
        left = start_low #* factor
        while func(left, *args) > 0:
            right = left
            left *= factor
            if n_it >= max_it:
                break
            n_it += 1
        # left is now such that cdf(left) < q
    if right is None: # i.e. self.b = inf
        right = start_upp #* factor
        while func(right, *args) < 0:
            left = right
            right *= factor
            if n_it >= max_it:
                break
            n_it += 1
        # right is now such that cdf(right) > q

    if n_it >= max_it:
        print 'Warning: max_it reached'

#    if left > right:
#        left, right = right, left #swap
    return optimize.brentq(func, \
                           left, right, args=args,
                           xtol=xtol, maxiter=maxiter_bq)


def func(x, a):
    f = (x - a)**3
    print 'evaluating at %f, fval = %f' % (x, f)
    return f



def funcn(x, a):
    f = -(x - a)**3
    print 'evaluating at %g, fval = %g' % (x, f)
    return f

def func2(x, a):
    f = (x - a)**3
    print 'evaluating at %g, fval = %f' % (x, f)
    return f

if __name__ == '__main__':
    print brentq_expanding(func, args=(0,), increasing=True)

    print brentq_expanding(funcn, args=(0,), increasing=False)
    print brentq_expanding(funcn, args=(-50,), increasing=False)

    print brentq_expanding(func, args=(20,))
    print brentq_expanding(funcn, args=(20,))
    print brentq_expanding(func, args=(500000,))

    # one bound
    print brentq_expanding(func, args=(500000,), low=10000)
    print brentq_expanding(func, args=(-50000,), upp=-1000)

    print brentq_expanding(funcn, args=(500000,), low=10000)
    print brentq_expanding(funcn, args=(-50000,), upp=-1000)

    # both bounds
    # hits maxiter in brentq if bounds too wide
    print brentq_expanding(func, args=(500000,), low=300000, upp=700000)
    print brentq_expanding(func, args=(-50000,), low= -70000, upp=-1000)
    print brentq_expanding(funcn, args=(500000,), low=300000, upp=700000)
    print brentq_expanding(funcn, args=(-50000,), low= -70000, upp=-10000)

    print brentq_expanding(func, args=(1.234e30,), xtol=1e10, increasing=True, maxiter_bq=200)


    #
    from numpy.testing import assert_allclose
    cases = [
        (0, {}),
        (50, {}),
        (-50, {}),
        (500000, dict(low=10000)),
        (-50000, dict(upp=-1000)),
        (500000, dict(low=300000, upp=700000)),
        (-50000, dict(low= -70000, upp=-1000))
        ]

    funcs = [(func, None),
             (func, True),
             (funcn, None),
             (funcn, False)]

    for func, inc in funcs:
        for a, kwds in cases:
            kw = {'increasing':inc}
            kw.update(kwds)
            res = brentq_expanding(func, args=(a,), **kwds)
            assert_allclose(res, a, rtol=1e-5)
