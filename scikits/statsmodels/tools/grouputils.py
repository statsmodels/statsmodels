# -*- coding: utf-8 -*-
"""Tools for working with groups

This provides several functions to work with groups and a Group class that
keeps track of the different representations and has methods to work more
easily with groups.


Author: Josef Perktold,
Author: Nathaniel Smith, recipe for sparse_dummies on scipy user mailing list

Created on Tue Nov 29 15:44:53 2011 : sparse_dummies
Created on Wed Nov 30 14:28:24 2011 : combine_indices
changes: add Group class

Notes
~~~~~

sparse_dummies needs checking for corner cases, e.g.
what if a category level has zero elements? This can happen with subset
    selection even if the original groups where defined as arange.

Not all methods and options have been tried out yet after refactoring

need more efficient loop if groups are sorted

"""

import numpy as np

def combine_indices(groups, prefix='', sep='.', return_labels=False):
    '''use np.unique to get integer group indices for product, intersection

    '''
    if isinstance(groups, tuple):
        groups = np.column_stack(groups)
    else:
        groups = np.asarray(groups)

    dt = groups.dtype
    print dt

    is2d = (groups.ndim == 2) #need to store

    if is2d:
        ncols = groups.shape[1]
        if not groups.flags.c_contiguous:
            groups = np.array(groups, order='C')

        groups_ = groups.view([('',groups.dtype)]*groups.shape[1])
    else:
        groups_ = groups

    uni, uni_idx, uni_inv = np.unique(groups_, return_index=True,
                                      return_inverse=True)

    if is2d:
        uni = uni.view(dt).reshape(-1, ncols)

        #avoiding a view would be
#        for t in uni.dtype.fields.values():
#            assert (t[0] == dt)
#
#        uni.dtype = dt
#        uni.shape = (uni.size//ncols, ncols)

    if return_labels:
        label = [(prefix+sep.join(['%s']*len(uni[0]))) % tuple(ii)
                                           for ii in uni]
        return uni_inv, uni_idx, uni, label
    else:
        return uni_inv, uni_idx, uni


#written for and used in try_covariance_grouploop.py
def group_sums(x, group, use_bincount=True):
    '''simple bincount version, again

    group : array, integer
        assumed to be consecutive integers

    no dtype checking because I want to raise in that case

    uses loop over columns of x

    for comparison, simple python loop
    '''
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:,None]
    elif x.ndim > 2 and use_bincount:
        raise ValueError('not implemented yet')

    if use_bincount:
        return np.array([np.bincount(group, weights=x[:,col])
                               for col in range(x.shape[1])])
    else:
        uniques = np.unique(group)
        result = np.zeros([len(uniques)] + list(x.shape[1:]))
        for ii, cat in enumerate(uniques):
            result[ii] = x[g==cat].sum(0)
        return result


def group_sums_dummy(x, group_dummy):
    '''sum by groups given group dummy variable

    group_dummy can be either ndarray or sparse matrix
    '''
    if type(group_dummy) is np.ndarray:
        return np.dot(x.T, group_dummy)
    else:  #check for sparse
        return x.T * group_dummy

def dummy_sparse(groups):
    '''create a sparse indicator from a group array with integer labels

    Parameters
    ----------
    groups: ndarray, int, 1d (nobs,)
        an array of group indicators for each observation. Group levels are assumed
        to be defined as consecutive integers, i.e. range(n_groups) where
        n_groups is the number of group levels.

    Returns
    -------
    indi : ndarray, int8, 2d (nobs, n_groups)
        an indicator array with one row per observation, that has 1 in the
        column of the group level for that observation

    Examples
    --------

    >>> g = np.array([0, 0, 2, 1, 1, 2, 0])
    >>> indi = dummy_sparse(g)
    >>> indi
    <7x3 sparse matrix of type '<type 'numpy.int8'>'
        with 7 stored elements in Compressed Sparse Row format>
    >>> indi.todense()
    matrix([[1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]], dtype=int8)


    current behavior with missing groups
    >>> g = np.array([0, 0, 2, 0, 2, 0])
    >>> indi = dummy_sparse(g)
    >>> indi.todense()
    matrix([[1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0]], dtype=int8)

    '''
    from scipy import sparse

    indptr = np.arange(len(groups)+1)
    data = np.ones(len(groups), dtype=np.int8)
    indi = sparse.csr_matrix((data, g, indptr))

    return indi


class Group(object):

    def __init__(self, group, name=''):

        #self.group = np.asarray(group)   #TODO: use checks in combine_indices
        self.name = name
        uni, uni_idx, uni_inv = combine_indices(group)

        #TODO: rename these to something easier to remember
        self.group_int, self.uni_idx, self.uni = uni, uni_idx, uni_inv

        self.n_groups = len(self.uni)

        #put this here so they can be overwritten before calling labels
        self.separator = '.'
        self.prefix = self.name
        if self.prefix:
            self.prefix = self.prefix + '='

    #cache_decorator
    def labels(self):
        #is this only needed for product of groups (intersection)?
        prefix = self.prefix
        uni = self.uni
        sep = self.separator

        if uni.ndim > 1:
            label = [(prefix+sep.join(['%s']*len(uni[0]))) % tuple(ii)
            for ii in uni]
        else:
            label = [prefix + '%s' % ii for ii in uni]
        return label

    def dummy(self, drop_idx=None, sparse=False, dtype=int):
        '''
        drop_idx is only available if sparse=False

        drop_idx is supposed to index into uni
        '''
        uni = self.uni
        if drop_idx is not None:
            idx = range(len(uni))
            del idx[drop_idx]
            uni = uni[idx]

        group = self.group

        if not sparse:
            return (group[:,None] == uni[None,:]).astype(dtype)

        else:
            return dummy_sparse(self.group_int)

    def interaction(self, other):
        if isinstance(other, self.__class__):
            other = other.group
        return self.__class__((self, other))

    def group_sums(self, x, use_bincount=True):
        return group_sums(x, self.group_int, use_bincount=use_bincount)

class GroupSorted(Group):

    def __init__(self, group, name=''):
        super(self.__class__, self).__init__(group, name=name)

        idx = (np.nonzero(np.diff(group))[0]+1).tolist()
        self.groupidx = groupidx = zip([0]+idx, idx+[len(group)])

        ngroups = len(groupidx)

    def group_iter(self):
        for low, upp in self.groupidx:
            yield slice(low, upp)


if __name__ == '__main__':

    #---------- examples combine_indices
    from numpy.testing import assert_equal

    np.random.seed(985367)
    groups = np.random.randint(0,2,size=(10,2))
    uv, ux, u, label = combine_indices(groups, return_labels=True)
    uv, ux, u, label = combine_indices(groups, prefix='g1,g2=', sep=',',
                                       return_labels=True)

    group0 = np.array(['sector0', 'sector1'])[groups[:,0]]
    group1 = np.array(['region0', 'region1'])[groups[:,1]]
    uv, ux, u, label = combine_indices((group0, group1),
                                       prefix='sector,region=',
                                       sep=',',
                                       return_labels=True)
    uv, ux, u, label = combine_indices((group0, group1), prefix='', sep='.',
                                       return_labels=True)
    group_joint = np.array(label)[uv]
    group_joint_expected = np.array(
                  ['sector1.region0', 'sector0.region1', 'sector0.region0',
                   'sector0.region1', 'sector1.region1', 'sector0.region0',
                   'sector1.region0', 'sector1.region0', 'sector0.region1',
                   'sector0.region0'],
      dtype='|S15')
    assert_equal(group_joint, group_joint_expected)

    '''
    >>> uv
    array([2, 1, 0, 0, 1, 0, 2, 0, 1, 0])
    >>> label
    ['sector0.region0', 'sector1.region0', 'sector1.region1']
    >>> np.array(label)[uv]
    array(['sector1.region1', 'sector1.region0', 'sector0.region0',
           'sector0.region0', 'sector1.region0', 'sector0.region0',
           'sector1.region1', 'sector0.region0', 'sector1.region0',
           'sector0.region0'],
          dtype='|S15')
    >>> np.column_stack((group0, group1))
    array([['sector1', 'region1'],
           ['sector1', 'region0'],
           ['sector0', 'region0'],
           ['sector0', 'region0'],
           ['sector1', 'region0'],
           ['sector0', 'region0'],
           ['sector1', 'region1'],
           ['sector0', 'region0'],
           ['sector1', 'region0'],
           ['sector0', 'region0']],
          dtype='|S7')
      '''

    #------------- examples sparse_dummies
    from scipy import sparse

    g = np.array([0, 0, 1, 2, 1, 1, 2, 0])
    u = range(3)
    indptr = np.arange(len(g)+1)
    data = np.ones(len(g), dtype=np.int8)
    a = sparse.csr_matrix((data, g, indptr))
    print a.todense()
    print np.all(a.todense() == (g[:,None] == np.arange(3)).astype(int))

    x = np.arange(len(g)*3).reshape(len(g), 3, order='F')

    print 'group means'
    print x.T * a
    print np.dot(x.T, g[:,None] == np.arange(3))
    print np.array([np.bincount(g, weights=x[:,col]) for col in range(3)])
    for cat in u:
        print x[g==cat].sum(0)
    for cat in u: x[g==cat].sum(0)

    cc = sparse.csr_matrix([[0, 1, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 1, 0, 1, 0]])

    #------------- groupsums
    print group_sums(np.arange(len(g)*3*2).reshape(len(g),3,2), g,
                    use_bincount=False).T
    print group_sums(np.arange(len(g)*3*2).reshape(len(g),3,2)[:,:,0], g)
    print group_sums(np.arange(len(g)*3*2).reshape(len(g),3,2)[:,:,1], g)

    #------------- examples class
    x = np.arange(len(g)*3).reshape(len(g), 3, order='F')
    mygroup = Group(g)
    print mygroup.group_int
    print mygroup.group_sums(x)
    print mygroup.labels()
