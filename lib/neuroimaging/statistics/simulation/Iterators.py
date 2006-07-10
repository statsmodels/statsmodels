import probstat, whrandom
import numpy as N

from BrainSTAT.Base.Options import parallel
if parallel:
    from BrainSTAT.Base.Parallel import prange

if parallel:
    import mpi

class probstat_iter:

    def __getitem__(self, key):
        return self.results[key]

    def __init__(self, results):
        self.results = results
        self.index = 0

    def __len__(self):
        return len(self.results)

    def __iter__(self):
        return self

    def next(self):
        if self.index < len(self.results):
            value = self.results[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration

class probstat_iter_list:

    def __getitem__(self, key):
        if type(key) is not type(1):
            raise ValueError, 'key should be an integer for class probstat_iter_list'
        j = 0
        tmp = 0
        tmp2 = 0
        while key >= tmp:
            tmp2 = tmp
            tmp = tmp + self.sizes[j]
            j = j + 1
        try:
            return self.results[max(0,j-1)][key - tmp2]
        except:
            value = self.results[max(0,j-1)].next()
            self.results[max(0,j-1)] = iter([value])
            return value

    def __init__(self, results):
        self.results = [iter(result) for result in results]
        self.sizes = [len(result) for result in results]
        self.niter = len(results)
        self.index = [0] * self.niter

    def __len__(self):

        return N.add.reduce(self.sizes)

    def __iter__(self):
        return self

    def next(self):
        if len(self.results) > 0:
            try:
                value = self.results[0].next()
            except:
                self.results.pop(0)
                self.niter = self.niter - 1
                value = self.next()
                pass
            return value
        else:
            raise StopIteration

def permutations(n):
    """
    Return an iterator to run through all permutations range(n).
    """
    return iter(probstat_iter(probstat.Permutation(range(n))))

def subsets2groups(n1, n2):
    return subsets(n1, n1+n2)

def subsets(k, n):
    """
    Return an iterator to run through all subsets of size k from range(n).
    """
    if k > 0:
        return iter(probstat_iter(probstat.Combination(range(n), k)))
    else:
        return [[]]

def allsubsets(n):
    """
    Return a list of iterators to run through all subsets of range(n).
    """

    iters = [subsets(i,n) for i in range(0,n+1)]
    return iter(probstat_iter_list(iters))

def halfsubsets(n):
    """
    Return a list of iterators to run through all subsets of range(n).
    """

    iters = [subsets(i,n) for i in range(0,n/2+1)]
    return iter(probstat_iter_list(iters))

def reorder_list(input):
    """
    Randomly reorder a list.
    """
    n = len(input)
    value = []
    indices = range(n)
    for i in range(n):
        index = indices.pop(whrandom.randint(0, len(indices)-1))
        value.append(input[index])
    return value

class RandomIterator:

    def __init__(self, iterator, niter=1000):
        self.iterator = iterator
        self.niter = niter
        self.j = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.niter

    def __getitem__(self, key):
        return self.next()

    def next(self):
        self.j += 1
        if self.j <= self.niter:
            return self.iterator[whrandom.randint(0, len(self.iterator)-1)]
        else:
            raise StopIteration

def random_subsets(k, n, niter=1000):
    return RandomIterator(subsets(k, n), niter=niter)

def random_subsets2groups(n1, n2, niter=1000):
    return RandomIterator(subsets(n1, n1+n2), niter=niter)

def random_sign(n, niter=1000):
    return RandomIterator(allsubsets(n), niter=niter)

class SignIterator:
    """

    >>> from BrainSTAT.NonParametric.Iterators import *
    >>> l = SignIterator(n=3)
    >>> for x in l:
    ...     print x
    ...
    [ 1.  1.  1.]
    [-1.  1.  1.]
    [ 1. -1.  1.]
    [ 1.  1. -1.]
    [-1. -1.  1.]
    [-1.  1. -1.]
    [ 1. -1. -1.]
    [-1. -1. -1.]

    >>> l = SignIterator(generator=subsets, k=2, n=4)
    >>> for x in l:
    ...     print x
    ...
    [-1. -1.  1.  1.]
    [-1.  1. -1.  1.]
    [-1.  1.  1. -1.]
    [ 1. -1. -1.  1.]
    [ 1. -1.  1. -1.]
    [ 1.  1. -1. -1.]

    >>> l = SignIterator(generator=subsets2groups, n1=2, n2=2)
    >>> for x in l:
    ...     print x
    ...
    [-1. -1.  1.  1.]
    [-1.  1. -1.  1.]
    [-1.  1.  1. -1.]
    [ 1. -1. -1.  1.]
    [ 1. -1.  1. -1.]
    [ 1.  1. -1. -1.]

"""

    random = True

    def __init__(self, generator=halfsubsets, **keywords):
        if generator is None:
            generator = allsubsets
        self.iter = generator(**keywords)
        for key, val in keywords.items():
            setattr(self, key, val)
        if hasattr(self, 'n1'):
            try:
                self.n = self.n1 + self.n2 ### HACK ###
            except:
                pass

    def __getitem__(self, key):
        return self.iter[key]

    def __len__(self):
        return len(self.iter)

    def __iter__(self):
        self.iter = [self.iter[i] for i in range(len(self.iter))]
        self.ntotal = len(self.iter)

        if parallel and not hasattr(self, 'split'):
            self.split = True
            a, b = prange(len(self.iter))
            _iter = [self.iter[i] for i in range(a, b)]
            self.iter = _iter
        if SignIterator.random:
            n = len(self.iter)
            self.iter = reorder_list(self.iter)
        self.index = 0
        return self

    def next(self):
        try:
            subset = self.iter[self.index]
        except:
            raise StopIteration
        self.index += 1
        return subset2signs(subset, self.n)

def signs2subset(signs):
    signs = N.array(signs)
    if len(signs.shape) != 1:
        raise ValueError, 'signs2subset expecting one sign at a time.'
    return N.compress(N.greater(signs, 0), N.arange(signs.shape[0]))

def subset2signs(subset, n):
    values = N.ones((n,), N.float64)
    for i in subset:
        values[i] = -1.0
    return values


def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()


