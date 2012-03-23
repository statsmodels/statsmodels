# -*- coding: utf-8 -*-
"""

Created on Wed Feb 29 10:12:38 2012

Author: Josef Perktold
License: BSD-3

"""

import itertools

try:
    #python 2.6, 2.7
    zip_longest = itertools.izip_longest
    pass
except AttributeError:
    #python 3.2
    try:
        zip_longest = itertools.zip_longest
        pass
    except AttributeError:
        #python 2.5
        def zip_longest(*args, **kwds):
            '''python 2.5 version for transposing a list of lists

            adds None for lists of shorter length, may not have the same
            behavior as python 2.6 izip_longest or python 3.2 zip_longest for
            other cases

            Parameters
            ----------
            args : sequence of iterables
                iterables that will be combined in transposed way

            Returns
            -------
            it : iterator
                iterator that generates tuples

            Examples
            --------
            >>> lili = [['a0', 'b0', 'c0', 'd0'], ['a1', 'b1', 'c1'],
                        ['a2', 'b2', 'c2', 'd2'], ['a3', 'b3', 'c3', 'd3'],
                        ['a4', 'b4']]
            >>> list(izip_longest(*lili))
            [('a0', 'a1', 'a2', 'a3', 'a4'), ('b0', 'b1', 'b2', 'b3', 'b4'),
             ('c0', 'c1', 'c2', 'c3', None), ('d0', None, 'd2', 'd3', None)]

            '''

            # izip_longest('ABCD', 'xy', fillvalue='-') --> Ax By C- D-
            fillvalue = kwds.get('fillvalue')
            def sentinel(counter = ([fillvalue]*(len(args)-1)).pop):
                yield counter()         # yields the fillvalue, or raises IndexError
            fillers = itertools.repeat(fillvalue)
            iters = [itertools.chain(it, sentinel(), fillers) for it in args]
            try:
                for tup in itertools.izip(*iters):
                    yield tup
            except IndexError:
                pass


try:
    from itertools import combinations
except ImportError:
    #from python 2.6 documentation
    def combinations(iterable, r):
        # combinations('ABCD', 2) --> AB AC AD BC BD CD
        # combinations(range(4), 3) --> 012 013 023 123
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = range(r)
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1
            yield tuple(pool[i] for i in indices)
