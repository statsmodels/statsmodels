# -*- coding: utf-8 -*-
"""

Created on Mon Mar 18 15:48:23 2013
Author: Josef Perktold

"""

import numpy as np
from scipy import optimize

DEBUG = False

# based on scipy.stats.distributions._ppf_single_call
def brentq_expanding(func, low=None, upp=None, args=(), xtol=1e-5,
                     start_low=None, start_upp=None, increasing=None,
                     max_it=100, maxiter_bq=100):
    '''find the root of a function in one variable by expanding and brentq

    Assumes function ``func`` is monotonic.

    Parameter
    ---------
    func : callable
        function for which we find the root ``x`` such that ``func(x) = 0``
    low : float or None
        lower bound for brentq
    upp : float or None
        upper bound for brentq
    args : tuple
        optional additional arguments for ``func``
    xtol : float
        parameter x tolerance given to brentq
    start_low : float (positive) or None
        starting bound for expansion with increasing ``x``. It needs to be
        positive. If None, then it is set to 1.
    start_upp : float (negative) or None
        starting bound for expansion with decreasing ``x``. It needs to be
        negative. If None, then it is set to -1.
    increasing : bool or None
        If None, then the function is evaluated at the initial bounds to
        determine wether the function is increasing or not. If increasing is
        True (False), then it is assumed that the function is monotonically
        increasing (decreasing).
    max_it : int
        maximum number of expansion steps
    maxiter_bq : int
        maximum number of iterations of brentq.

    Returns
    -------
    x : float
        root of the function, value at which ``func(x) = 0``.

    '''


    left, right = low, upp  #alias

    # start_upp first because of possible sl = -1 > upp
    if upp is not None:
        su = upp
    elif start_upp is not None:
        if start_upp < 0:
            print "raise ValueError('start_upp needs to be positive')"
        su = start_upp
    else:
        su = 1


    if low is not None:
        sl = low
    elif start_low is not None:
        if start_low > 0:
            print "raise ValueError('start_low needs to be negative')"
        sl = start_low
    else:
        sl = min(-1, su - 1)

    # need sl < su
    if upp is None:
        su = max(su, sl + 1)


    # increasing or not ?
    if ((low is None) or (upp is None)) and increasing is None:
        assert sl < su  # check during developement
        f_low = func(sl, *args)
        f_upp = func(su, *args)
        increasing = (f_low < f_upp)

    if DEBUG:
        print 'low, upp', low, upp
        print 'increasing', increasing
        print 'sl, su', sl, su

    if not increasing:
        sl, su = su, sl
        left, right = right, left

    n_it = 0
    factor = 10.
    if left is None:
        left = sl
        while func(left, *args) > 0:
            right = left
            left *= factor
            if n_it >= max_it:
                break
            n_it += 1
        # left is now such that func(left) < q
    if right is None:
        right = su
        while func(right, *args) < 0:
            left = right
            right *= factor
            if n_it >= max_it:
                break
            n_it += 1
        # right is now such that func(right) > q

    if n_it >= max_it:
        print 'Warning: max_it reached'
        #TODO: use Warnings

    return optimize.brentq(func, \
                           left, right, args=args,
                           xtol=xtol, maxiter=maxiter_bq)


def func(x, a):
    f = (x - a)**3
    if DEBUG: print 'evaluating at %f, fval = %f' % (x, f)
    return f



def funcn(x, a):
    f = -(x - a)**3
    if DEBUG: print 'evaluating at %g, fval = %g' % (x, f)
    return f

def func2(x, a):
    f = (x - a)**3
    print 'evaluating at %g, fval = %f' % (x, f)
    return f

if __name__ == '__main__':
    run_all = False
    if run_all:
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

        print brentq_expanding(func, args=(1.234e30,), xtol=1e10,
                               increasing=True, maxiter_bq=200)


    print brentq_expanding(func, args=(-50000,), start_low=-10000)
    print brentq_expanding(func, args=(-500,), start_upp=-100)
    ''' it still works
    raise ValueError('start_upp needs to be positive')
    -499.999996336
    '''
    ''' this doesn't work
    >>> print brentq_expanding(func, args=(-500,), start_upp=-1000)
    raise ValueError('start_upp needs to be positive')
    OverflowError: (34, 'Result too large')
    '''

    try:
        print brentq_expanding(funcn, args=(-50000,), low= -40000, upp=-10000)
    except Exception, e:
        print e

    #
    from numpy.testing import assert_allclose, assert_raises

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

    for f, inc in funcs:
        for a, kwds in cases:
            kw = {'increasing':inc}
            kw.update(kwds)
            res = brentq_expanding(f, args=(a,), **kwds)
            print '%10d'%a, ['dec', 'inc'][f is func], res - a
            assert_allclose(res, a, rtol=1e-5)

    # wrong sign for start bounds
    # doesn't raise yet during development TODO: activate this
    # it kind of works in some cases, but not correctly or in a useful way
    #assert_raises(ValueError, brentq_expanding, func, args=(-500,), start_upp=-1000)
    #assert_raises(ValueError, brentq_expanding, func, args=(500,), start_low=1000)

    # low upp given, but doesn't bound root, leave brentq exception
    # ValueError: f(a) and f(b) must have different signs
    assert_raises(ValueError, brentq_expanding, funcn, args=(-50000,), low= -40000, upp=-10000)

    # max_it too low to find root bounds
    # ValueError: f(a) and f(b) must have different signs
    assert_raises(ValueError, brentq_expanding, func, args=(-50000,), max_it=2)

    # maxiter_bq too low
    # RuntimeError: Failed to converge after 3 iterations.
    assert_raises(RuntimeError, brentq_expanding, func, args=(-50000,), maxiter_bq=3)
