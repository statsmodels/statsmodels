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

This reverses the class I used before, where the class was for the data and
the group was auxiliary. Here, it is only the group, no data is kept.

sparse_dummies needs checking for corner cases, e.g.
what if a category level has zero elements? This can happen with subset
    selection even if the original groups where defined as arange.

Not all methods and options have been tried out yet after refactoring

need more efficient loop if groups are sorted -> see GroupSorted.group_iter



"""

import numpy as np
import pandas as pd
from statsmodels.compatnp.np_compat import npc_unique
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly


def combine_indices(groups, prefix='', sep='.', return_labels=False):
    '''use np.unique to get integer group indices for product, intersection

    '''
    if isinstance(groups, tuple):
        groups = np.column_stack(groups)
    else:
        groups = np.asarray(groups)

    dt = groups.dtype
    #print dt

    is2d = (groups.ndim == 2)  # need to store

    if is2d:
        ncols = groups.shape[1]
        if not groups.flags.c_contiguous:
            groups = np.array(groups, order='C')

        groups_ = groups.view([('', groups.dtype)] * groups.shape[1])
    else:
        groups_ = groups

    uni, uni_idx, uni_inv = npc_unique(groups_, return_index=True,
                                       return_inverse=True)

    if is2d:
        uni = uni.view(dt).reshape(-1, ncols)

        #avoiding a view would be
        #for t in uni.dtype.fields.values():
        #    assert (t[0] == dt)
        #
        #uni.dtype = dt
        #uni.shape = (uni.size//ncols, ncols)

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
        x = x[:, None]
    elif x.ndim > 2 and use_bincount:
        raise ValueError('not implemented yet')

    if use_bincount:
        return np.array([np.bincount(group, weights=x[:, col])
                         for col in range(x.shape[1])])
    else:
        uniques = np.unique(group)
        result = np.zeros([len(uniques)] + list(x.shape[1:]))
        for ii, cat in enumerate(uniques):
            result[ii] = x[g == cat].sum(0)
        return result


def group_sums_dummy(x, group_dummy):
    '''sum by groups given group dummy variable

    group_dummy can be either ndarray or sparse matrix
    '''
    if data_util._is_using_ndarray_type(group_dummy):
        return np.dot(x.T, group_dummy)
    else:  # check for sparse
        return x.T * group_dummy


def dummy_sparse(groups):
    '''create a sparse indicator from a group array with integer labels

    Parameters
    ----------
    groups: ndarray, int, 1d (nobs,)
        an array of group indicators for each observation. Group levels are
        assumed to be defined as consecutive integers, i.e. range(n_groups)
        where n_groups is the number of group levels. A group level with no
        observations for it will still produce a column of zeros.

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

    #cache decorator
    def counts(self):
        return np.bincount(self.group_int)

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
            return (group[:, None] == uni[None, :]).astype(dtype)
        else:
            return dummy_sparse(self.group_int)

    def interaction(self, other):
        if isinstance(other, self.__class__):
            other = other.group
        return self.__class__((self, other))

    def group_sums(self, x, use_bincount=True):
        return group_sums(x, self.group_int, use_bincount=use_bincount)

    def group_demean(self, x, use_bincount=True):
        means_g = group_demean(x / float(nobs), self.group_int,
                               use_bincount=use_bincount)
        x_demeaned = x - means_g[self.group_int]  # check reverse_index?
        return x_demeaned, means_g


class GroupSorted(Group):
    def __init__(self, group, name=''):
        super(self.__class__, self).__init__(group, name=name)

        idx = (np.nonzero(np.diff(group))[0]+1).tolist()
        self.groupidx = groupidx = zip([0]+idx, idx+[len(group)])

        ngroups = len(groupidx)

    def group_iter(self):
        for low, upp in self.groupidx:
            yield slice(low, upp)

    def lag_indices(self, lag):
        '''return the index array for lagged values

        Warning: if k is larger then the number of observations for an
        individual, then no values for that individual are returned.

        TODO: for the unbalanced case, I should get the same truncation for
        the array with lag=0. From the return of lag_idx we wouldn't know
        which individual is missing.

        TODO: do I want the full equivalent of lagmat in tsa?
        maxlag or lag or lags.

        not tested yet

        '''
        lag_idx = np.asarray(self.groupidx)[:, 1] - lag  # asarray or already?
        mask_ok = (lag <= lag_idx)
        #still an observation that belongs to the same individual

        return lag_idx[mask_ok]


class Grouping():
    def __init__(self, index_pandas=None, index_list=None):
        '''
        index_pandas : pandas.MultiIndex
            The hierarchical index with panels and dates
        index_list : list
            A list of numpy arrays, pandas series or lists, all of which
            are of length nobs.
        '''
        if index_list is not None:
            try:
                index_list = [np.array(x) for x in index_list]
                tup = zip(*index_list)
                self.index = pd.MultiIndex.from_tuples(tup)
            except:
                raise Exception("index_list must be a list of lists, pandas "
                                "series, or numpy arrays, each of identitcal "
                                "length nobs.")
        else:
            self.index = index_pandas
        self.nobs = len(self.index)
        self.slices = None
        self.index_shape = self.index.levshape
        self.index_int = self.index.labels

    def reindex(self, index_pandas=None):
        """
        Resets the index in-place.
        """
        if type(index_pandas) in [pd.core.index.MultiIndex,
                                  pd.core.index.Index]:
            self.index = index_pandas
            self.index_shape = self.index.levshape
            self.index_int = self.index.labels
        else:
            raise Exception('index_pandas must be Pandas index')

    def get_slices(self):
        '''
        Sets the slices attribute to be a list of indices of the sorted
        groups for the first index level. I.e., self.slices[0] is the
        index where each observation is in the first (sorted) group.
        '''
        groups = self.index.get_level_values(0).unique()
        self.slices = [self.index.get_loc(x) for x in groups]

    def count_categories(self, level=0):
        """
        Sets the attribute counts to equal the bincount of the (integer-valued)
        labels.
        """
        self.counts = np.bincount(self.index_int[level])

    def check_index(self, sorted=True, unique=True, index=None):
        '''Sanity checks'''
        if not index:
            index = self.index
        if sorted:
            test = pd.DataFrame(range(len(index)), index=index)
            test_sorted = test.sort()
            if any(test.index != test_sorted.index):
                raise Exception('Index suggests that data may not be sorted')
        if unique:
            if len(index) != len(index.unique()):
                raise Exception('Duplicate index entries')

    def sort(self, data, index=None):
        '''Applies a (potentially hierarchical) sort operation on a numpy array
        or pandas series/dataframe based on the grouping index or a
        user-suplied index.  Returns an object of the same type as the original
        data as well as the matching (sorted) Pandas index.
        '''

        if not index:
            index = self.index
        if data_util._is_using_ndarray_type(data):
            if data.ndim == 1:
                out = pd.Series(data, index=index, copy=True)
                out.sort_index()
            else:
                out = pd.DataFrame(data, index=index)
                out = out.sort(inplace=False) # copies
            return np.array(out), index
        elif data_util._is_using_pandas(data):
            out = data
            out.index = index
            out = out.sort()
            return out, index
        else:
            msg = 'data must be a Numpy array or a Pandas Series/DataFrame'
            raise Exception(msg)

    def transform_dataframe(self, dataframe, function, level=0):
        '''Apply function to each column, by group
        Assumes that the dataframe already has a proper index'''
        if dataframe.shape[0] != self.nobs:
            raise Exception('dataframe does not have the same shape as index')
        out = dataframe.groupby(level=level).apply(function)
        if 1 in out.shape:
            return np.ravel(out)
        else:
            return np.array(out)

    def transform_array(self, array, function, level=0):
        '''Apply function to each column, by group'''
        if array.shape[0] != self.nobs:
            raise Exception('array does not have the same shape as index')
        dataframe = pd.DataFrame(array, index=self.index)
        return self.transform_dataframe(dataframe, function, level=level)

    def transform_slices(self, array, function, **kwargs):
        '''Apply function to each group. Similar to transform_array but does
        not coerce array to a DataFrame and back and only works on a 1D or 2D
        numpy array'''
        if array.shape[0] != self.nobs:
            raise Exception('array does not have the same shape as index')
        if self.slices is None:
            self.get_slices()
        processed = []
        for s in self.slices:
            if array.ndim == 2:
                subset = array[s, :]
            elif array.ndim == 1:
                subset = array[s]
            processed.append(function(subset, s, **kwargs))
        return np.concatenate(processed)

    @cache_readonly
    def dummies_time(self):
        self.dummy_sparse(level=1)
        return self._dummies

    @cache_readonly
    def dummies_groups(self):
        self.dummy_sparse(level=0)
        return self._dummies

    def dummy_sparse(self, level=0):
        '''create a sparse indicator from a group array with integer labels

        Parameters
        ----------
        groups: ndarray, int, 1d (nobs,) an array of group indicators for each
            observation. Group levels are assumed to be defined as consecutive
            integers, i.e. range(n_groups) where n_groups is the number of
            group levels. A group level with no observations for it will still
            produce a column of zeros.

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
        groups = self.index.labels[level]
        indptr = np.arange(len(groups)+1)
        data = np.ones(len(groups), dtype=np.int8)
        self._dummies = sparse.csr_matrix((data, groups, indptr))

if __name__ == '__main__':

    #---------- examples combine_indices
    from numpy.testing import assert_equal

    np.random.seed(985367)
    groups = np.random.randint(0, 2, size=(10, 2))
    uv, ux, u, label = combine_indices(groups, return_labels=True)
    uv, ux, u, label = combine_indices(groups, prefix='g1,g2=', sep=',',
                                       return_labels=True)

    group0 = np.array(['sector0', 'sector1'])[groups[:, 0]]
    group1 = np.array(['region0', 'region1'])[groups[:, 1]]
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
    print np.all(a.todense() == (g[:, None] == np.arange(3)).astype(int))

    x = np.arange(len(g)*3).reshape(len(g), 3, order='F')

    print 'group means'
    print x.T * a
    print np.dot(x.T, g[:, None] == np.arange(3))
    print np.array([np.bincount(g, weights=x[:, col]) for col in range(3)])
    for cat in u:
        print x[g == cat].sum(0)
    for cat in u:
        x[g == cat].sum(0)

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
    print group_sums(np.arange(len(g)*3*2).reshape(len(g), 3, 2), g,
                     use_bincount=False).T
    print group_sums(np.arange(len(g)*3*2).reshape(len(g), 3, 2)[:, :, 0], g)
    print group_sums(np.arange(len(g)*3*2).reshape(len(g), 3, 2)[:, :, 1], g)

    #------------- examples class
    x = np.arange(len(g)*3).reshape(len(g), 3, order='F')
    mygroup = Group(g)
    print mygroup.group_int
    print mygroup.group_sums(x)
    print mygroup.labels()
