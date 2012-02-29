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
    pass
except AttributeError:
    #python 2.5
    def zip_longest(*args):
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
        return itertools.imap(None, *args)
        pass
