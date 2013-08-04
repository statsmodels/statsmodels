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
    zip_longest = itertools.zip_longest

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
