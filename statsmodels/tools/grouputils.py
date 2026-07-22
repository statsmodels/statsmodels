"""
Tools for working with groups

This provides several functions to work with groups and a Group class that
keeps track of the different representations and has methods to work more
easily with groups.

Author: Josef Perktold,
Author: Nathaniel Smith, recipe for sparse_dummies on scipy user mailing list

Created on Tue Nov 29 15:44:53 2011 : sparse_dummies
Created on Wed Nov 30 14:28:24 2011 : combine_indices
changes: add Group class

Notes
-----
This reverses the class I used before, where the class was for the data and
the group was auxiliary. Here, it is only the group, no data is kept.

sparse_dummies needs checking for corner cases, e.g.
what if a category level has zero elements? This can happen with subset
selection even if the original groups were defined as arange.

Not all methods and options have been tried out yet after refactoring.

Need a more efficient loop if groups are sorted -> see
GroupSorted.group_iter.
"""
from statsmodels.compat.python import lrange, lzip

import numpy as np
import pandas as pd
from pandas import Index, MultiIndex

import statsmodels.tools.data as data_util


def combine_indices(groups, prefix="", sep=".", return_labels=False):
    """
    Use np.unique to get integer group indices for product, intersection

    Parameters
    ----------
    groups : array_like or tuple
        If a tuple, the elements are stacked column-wise to form a 2d
        array of groups to combine. Otherwise treated as an existing 1d
        or 2d array of group values.
    prefix : str
        Prefix prepended to each label. Only used if ``return_labels``
        is True.
    sep : str
        Separator used to join the columns of a 2d group array when
        building labels. Only used if ``return_labels`` is True.
    return_labels : bool
        If True, also return a list of string labels for each unique
        group.

    Returns
    -------
    uni_inv : ndarray
        Integer codes into `uni` that reconstruct the original `groups`
        array, i.e. the group index of each observation.
    uni_idx : ndarray
        Indices of the first occurrence of each unique group in the
        input.
    uni : ndarray
        Sorted unique groups, or unique group combinations if `groups`
        was 2d or a tuple.
    label : list[str]
        String label for each unique group in `uni`. Only returned if
        ``return_labels`` is True.
    """
    if isinstance(groups, tuple):
        groups = np.column_stack(groups)
    else:
        groups = np.asarray(groups)

    dt = groups.dtype

    is2d = (groups.ndim == 2)  # need to store

    if is2d:
        ncols = groups.shape[1]
        if not groups.flags.c_contiguous:
            groups = np.array(groups, order="C")

        groups_ = groups.view([("", groups.dtype)] * groups.shape[1])
    else:
        groups_ = groups

    uni, uni_idx, uni_inv = np.unique(groups_, return_index=True,
                                      return_inverse=True)

    if is2d:
        uni = uni.view(dt).reshape(-1, ncols)

    if return_labels:
        label = [(prefix+sep.join(["%s"]*len(uni[0]))) % tuple(ii)
                 for ii in uni]
        return uni_inv, uni_idx, uni, label
    else:
        return uni_inv, uni_idx, uni


# written for and used in try_covariance_grouploop.py
def group_sums(x, group, use_bincount=True):
    """
    Sum ``x`` within integer groups

    Parameters
    ----------
    x : array_like
        Data of shape ``(nobs,)`` or ``(nobs, n_features)``. Higher-dimensional
        ``x`` is only supported when ``use_bincount=False``.
    group : array_like of non-negative int
        Non-negative integer group labels of shape ``(nobs,)``. For predictable
        indexing (e.g. ``result[group]``), groups should be coded as
        ``0, 1, ..., n_groups-1``.
    use_bincount : bool, default True
        Use ``np.bincount`` when True, otherwise a pure-Python group loop.
        The bincount path expects non-negative, reasonably consecutive codes
        and may re-label via ``pd.factorize`` when ``max(group)`` is large.

    Returns
    -------
    ndarray
        Shape ``(n_groups, n_features)``. 1-d ``x`` is treated as one feature
        and returns shape ``(n_groups, 1)``.

    Notes
    -----
    Both code paths previously disagreed on orientation (``use_bincount``
    returned ``(n_features, n_groups)``). They now both return
    ``(n_groups, n_features)`` so indexing by group id works (GH9921).
    """
    x = np.asarray(x)
    group = np.asarray(group).squeeze()
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim > 2 and use_bincount:
        raise ValueError("not implemented yet")

    if use_bincount:

        # re-label groups or bincount takes too much memory
        if np.max(group) > 2 * x.shape[0]:
            group = pd.factorize(group)[0]

        # column-wise bincount yields (n_features, n_groups); transpose
        return np.array(
            [
                np.bincount(group, weights=x[:, col])
                for col in range(x.shape[1])
            ]
        ).T
    else:
        uniques = np.unique(group)
        result = np.zeros([len(uniques)] + list(x.shape[1:]))
        for ii, cat in enumerate(uniques):
            result[ii] = x[group == cat].sum(0)
        return result


def group_sums_dummy(x, group_dummy):
    """
    Sum by groups given group dummy variable

    Parameters
    ----------
    x : array_like
        Data of shape ``(nobs, k)`` whose columns are summed within each
        group.
    group_dummy : ndarray or sparse matrix
        Indicator/dummy matrix of shape ``(nobs, n_groups)`` with a 1 in
        the column of the group each observation belongs to.

    Returns
    -------
    ndarray
        Group sums of shape ``(k, n_groups)``.
    """
    if data_util._is_using_ndarray_type(group_dummy, None):
        return np.dot(x.T, group_dummy)
    else:  # check for sparse
        return x.T * group_dummy


# TODO: See if this can be entirely replaced by Grouping.dummy_sparse;
#  see GH#5687
def dummy_sparse(groups):
    """
    Create a sparse indicator from a group array with integer labels

    Parameters
    ----------
    groups : ndarray, int, 1d (nobs,)
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

    """
    from scipy import sparse

    indptr = np.arange(len(groups)+1)
    data = np.ones(len(groups), dtype=np.int8)
    indi = sparse.csr_matrix((data, groups, indptr))

    return indi


class Group:
    """
    Represent grouping labels and derived encodings

    Parameters
    ----------
    group : array_like or tuple
        Group labels for each observation. A tuple of arrays is combined
        into a single set of group labels representing the intersection
        of the individual groupings.
    name : str
        Name used as a prefix when constructing string group labels.

    Attributes
    ----------
    name : str
        Name of the grouping variable.
    group_int : ndarray
        Integer codes giving the group membership of each observation.
    uni_idx : ndarray
        Indices of the first occurrence of each unique group.
    uni : ndarray
        Unique group values, or unique combinations if `group` was a
        tuple or 2d array.
    n_groups : int
        Number of unique groups.
    separator : str
        Separator used to join columns when building labels for a
        multi-column (intersection) grouping.
    prefix : str
        Prefix, derived from `name`, prepended to each group label.
    """

    def __init__(self, group, name=""):

        # self.group = np.asarray(group)  # TODO: use checks in combine_indices
        self.name = name
        uni, uni_idx, uni_inv = combine_indices(group)

        # TODO: rename these to something easier to remember
        self.group_int, self.uni_idx, self.uni = uni, uni_idx, uni_inv

        self.n_groups = len(self.uni)

        # put this here so they can be overwritten before calling labels
        self.separator = "."
        self.prefix = self.name
        if self.prefix:
            self.prefix = self.prefix + "="

    # cache decorator
    def counts(self):
        """
        Return the number of observations in each group

        Returns
        -------
        ndarray
            Count of observations for each integer group code in
            `group_int`.
        """
        return np.bincount(self.group_int)

    # cache_decorator
    def labels(self):
        """
        Return string labels for each unique group

        Returns
        -------
        list[str]
            Labels built from `prefix`, `separator` and the unique group
            values in `uni`.
        """
        # is this only needed for product of groups (intersection)?
        prefix = self.prefix
        uni = self.uni
        sep = self.separator

        if uni.ndim > 1:
            label = [(prefix+sep.join(["%s"]*len(uni[0]))) % tuple(ii)
                     for ii in uni]
        else:
            label = [prefix + "%s" % ii for ii in uni]
        return label

    def dummy(self, drop_idx=None, sparse=False, dtype=int):
        """
        Return a dummy/indicator matrix for the groups

        Parameters
        ----------
        drop_idx : int, optional
            Index into `uni` of a group level to drop from the returned
            matrix. Only available if ``sparse`` is False.
        sparse : bool
            If True, return a sparse indicator matrix instead of a dense
            array.
        dtype : type
            The dtype of the returned dense indicator matrix.

        Returns
        -------
        ndarray or sparse matrix
            Indicator matrix of shape ``(nobs, n_groups)``, or with one
            fewer column if `drop_idx` is given.
        """
        uni = self.uni
        if drop_idx is not None:
            idx = lrange(len(uni))
            del idx[drop_idx]
            uni = uni[idx]

        group = self.group

        if not sparse:
            return (group[:, None] == uni[None, :]).astype(dtype)
        else:
            return dummy_sparse(self.group_int)

    def interaction(self, other):
        """
        Return a new Group formed from the intersection with another grouping

        Parameters
        ----------
        other : Group or array_like
            The other grouping to intersect with. If a `Group` instance,
            its underlying group labels are used.

        Returns
        -------
        Group
            A new instance of the same class representing the
            intersection of the two groupings.
        """
        if isinstance(other, self.__class__):
            other = other.group
        return self.__class__((self, other))

    def group_sums(self, x, use_bincount=True):
        """
        Sum `x` within each group

        Parameters
        ----------
        x : array_like
            Data of shape ``(nobs,)`` or ``(nobs, n_features)``.
        use_bincount : bool
            Use ``np.bincount`` when True, otherwise a pure-Python group
            loop. See :func:`group_sums`.

        Returns
        -------
        ndarray
            Group sums of shape ``(n_groups, n_features)``.
        """
        return group_sums(x, self.group_int, use_bincount=use_bincount)

    def group_demean(self, x, use_bincount=True):
        """
        Demean `x` by subtracting the group means

        Parameters
        ----------
        x : array_like
            Data of shape ``(nobs,)`` or ``(nobs, n_features)``.
        use_bincount : bool
            Use ``np.bincount`` when True, otherwise a pure-Python group
            loop. See :func:`group_sums`.

        Returns
        -------
        x_demeaned : ndarray
            `x` with the group mean subtracted from each observation.
            Same shape as `x`.
        means_g : ndarray
            Group means, of shape ``(n_groups,)`` for 1d `x` or
            ``(n_groups, n_features)`` for 2d `x`.
        """
        x = np.asarray(x)
        was_1d = x.ndim == 1
        if was_1d:
            x = x[:, None]
        sums_g = group_sums(x, self.group_int, use_bincount=use_bincount)
        # group_sums returns (n_groups, n_features)
        counts = np.bincount(self.group_int)
        # Defensive: bincount may include zero-count bins for sparse codes.
        # group_int from combine_indices should not contain empty labels.
        counts = np.maximum(counts, 1)
        means_g = sums_g / counts[:, None]
        x_demeaned = x - means_g[self.group_int]
        if was_1d:
            x_demeaned = x_demeaned[:, 0]
            means_g = means_g[:, 0]
        return x_demeaned, means_g


class GroupSorted(Group):
    """
    Represent grouping labels that are already sorted by group

    Parameters
    ----------
    group : array_like
        Group labels for each observation. Must already be sorted so
        that all observations belonging to the same group are
        contiguous.
    name : str
        Name used as a prefix when constructing string group labels.

    Attributes
    ----------
    groupidx : list[tuple[int, int]]
        List of ``(start, stop)`` index pairs giving the slice bounds of
        each contiguous group.
    """

    def __init__(self, group, name=""):
        super(self.__class__, self).__init__(group, name=name)

        idx = (np.nonzero(np.diff(group))[0]+1).tolist()
        self.groupidx = lzip([0] + idx, idx + [len(group)])

    def group_iter(self):
        """
        Iterate over slices for each contiguous group

        Yields
        ------
        slice
            Slice selecting the observations belonging to one group.
        """
        for low, upp in self.groupidx:
            yield slice(low, upp)

    def lag_indices(self, lag):
        """
        Return the index array for lagged values

        Parameters
        ----------
        lag : int
            Number of periods to lag.

        Returns
        -------
        ndarray
            Indices into the sorted array selecting the lagged
            observations.

        Warnings
        --------
        If `lag` is larger than the number of observations for an
        individual, no values for that individual are returned.

        Notes
        -----
        For the unbalanced case, this does not apply the same
        truncation to the ``lag=0`` array, so it is not directly
        obvious from the return value which individual is missing.
        This has not been extensively tested. It is also not the full
        equivalent of ``lagmat`` in ``tsa``, which supports multiple
        lags at once.
        """
        lag_idx = np.asarray(self.groupidx)[:, 1] - lag  # asarray or already?
        mask_ok = (lag <= lag_idx)
        # still an observation that belongs to the same individual

        return lag_idx[mask_ok]


def _is_hierarchical(x):
    """
    Check if the first item of an array-like object is also array-like

    Parameters
    ----------
    x : array_like
        Array-like object to check.

    Returns
    -------
    bool
        True if `x` appears to represent hierarchical (MultiIndex-style)
        groups, i.e. its first element is itself list-like. False
        otherwise.
    """
    item = x[0]
    # is there a better way to do this?
    if isinstance(item, (list, tuple, np.ndarray, pd.Series, pd.DataFrame)):
        return True
    else:
        return False


def _make_hierarchical_index(index, names):
    """
    Build a MultiIndex from an array-like of row-wise group tuples

    Parameters
    ----------
    index : array_like
        Array-like where each row is a tuple of group values.
    names : list[str]
        Names to assign to the levels of the resulting MultiIndex.

    Returns
    -------
    MultiIndex
        MultiIndex constructed from the rows of `index`.
    """
    return MultiIndex.from_tuples(*[index], names=names)


def _make_generic_names(index):
    """
    Create generic zero-padded level names for an index

    Parameters
    ----------
    index : Index or MultiIndex
        Index whose number of levels (``len(index.names)``) determines
        how many generic names are generated.

    Returns
    -------
    list[str]
        Names of the form ``"group0"``, ``"group1"``, ... , zero-padded
        to a consistent width.
    """
    n_names = len(index.names)
    pad = str(len(str(n_names)))  # number of digits
    return [("group{0:0"+pad+"}").format(i) for i in range(n_names)]


class Grouping:
    """
    Represent a pandas-style grouping index and related transformations

    Parameters
    ----------
    index : index-like
        Can be a pandas MultiIndex, Index, or array-like. If array-like
        and hierarchical (more than one grouping variable), groups are
        expected to be given as a tuple in each row, e.g. ``[('red', 1),
        ('red', 2), ('green', 1), ('green', 2)]``.
    names : list or str, optional
        The names to use for the groups. Should be a str if only one
        grouping variable is used.

    Attributes
    ----------
    index : Index or MultiIndex
        The (possibly constructed) pandas index representing the
        groups.
    nobs : int
        Number of observations, i.e. ``len(index)``.
    nlevels : int
        Number of grouping levels.
    slices : list or None
        Cached slices of observations for each group of the first
        index level, set by :meth:`get_slices`.

    Notes
    -----
    If `index` is already a pandas Index then there is no copy.
    """

    def __init__(self, index, names=None):
        if isinstance(index, (Index, MultiIndex)):
            if names is not None:
                if hasattr(index, "set_names"):  # newer pandas
                    index.set_names(names, inplace=True)
                else:
                    index.names = names
            self.index = index
        else:  # array_like
            if _is_hierarchical(index):
                self.index = _make_hierarchical_index(index, names)
            else:
                self.index = Index(index, name=names)
            if names is None:
                names = _make_generic_names(self.index)
                if hasattr(self.index, "set_names"):
                    self.index.set_names(names, inplace=True)
                else:
                    self.index.names = names

        self.nobs = len(self.index)
        self.nlevels = len(self.index.names)
        self.slices = None

    @property
    def index_shape(self):
        """
        Shape of the underlying (Multi)Index

        Returns
        -------
        tuple
            ``levshape`` if the index defines it (a MultiIndex), else
            the plain index ``shape``.
        """
        if hasattr(self.index, "levshape"):
            return self.index.levshape
        else:
            return self.index.shape

    @property
    def levels(self):
        """
        Unique values for each grouping level

        Returns
        -------
        FrozenList or Index
            ``levels`` of the underlying MultiIndex, or the categories
            of the index treated as a single-level Categorical.
        """
        if hasattr(self.index, "levels"):
            return self.index.levels
        else:
            return pd.Categorical(self.index).levels

    @property
    def labels(self):
        """
        Integer codes for each grouping level

        Returns
        -------
        ndarray or FrozenList
            Integer codes for the index. For a MultiIndex, these are
            the per-level ``codes``; otherwise the Categorical codes
            for the single level, wrapped in a length-1 sequence.
        """
        # this was index_int, but that's not a very good name...
        codes = getattr(self.index, "codes", None)
        if codes is None:
            if hasattr(self.index, "labels"):
                codes = self.index.labels
            else:
                codes = pd.Categorical(self.index).codes[None]
        return codes

    @property
    def group_names(self):
        """
        Names of the grouping levels

        Returns
        -------
        FrozenList
            The names of the underlying index.
        """
        return self.index.names

    def reindex(self, index=None, names=None):
        """
        Reset the index in-place

        Parameters
        ----------
        index : index-like, optional
            The new index to use.
        names : list or str, optional
            The names to use for the groups. Defaults to the current
            `group_names` if not provided.
        """
        # NOTE: this is not of much use if the rest of the data does not change
        # This needs to reset cache
        if names is None:
            names = self.group_names
        # Does nothing ??
        Grouping(index, names)

    def get_slices(self, level=0):
        """
        Set `slices` to a list of indices of the sorted groups

        Parameters
        ----------
        level : int
            Index level to group by.

        Notes
        -----
        Sets the `slices` attribute to be a list of indices of the
        sorted groups for the given index level. I.e., ``self.slices[0]``
        is the index where each observation is in the first (sorted)
        group.
        """
        # TODO: refactor this
        groups = self.index.get_level_values(level).unique()
        groups = np.sort(np.array(groups))
        if isinstance(self.index, MultiIndex):
            self.slices = [self.index.get_loc_level(x, level=level)[0]
                           for x in groups]
        else:
            self.slices = [self.index.get_loc(x) for x in groups]

    def count_categories(self, level=0):
        """
        Set `counts` to the bincount of the labels at the given level

        Parameters
        ----------
        level : int
            Index level to count.

        Notes
        -----
        Sets the `counts` attribute to equal the bincount of the
        (integer-valued) labels for `level`.
        """
        # TODO: refactor this not to set an attribute. Why would we do this?
        self.counts = np.bincount(self.labels[level])

    def check_index(self, is_sorted=True, unique=True, index=None):
        """
        Sanity check that the index is sorted and/or unique

        Parameters
        ----------
        is_sorted : bool
            If True, check that `index` is sorted.
        unique : bool
            If True, check that `index` has no duplicate entries.
        index : index-like, optional
            Index to check. Defaults to `self.index`.

        Raises
        ------
        Exception
            If `is_sorted` is True and the index is not sorted, or if
            `unique` is True and the index has duplicate entries.
        """
        if not index:
            index = self.index
        if is_sorted:
            test = pd.DataFrame(lrange(len(index)), index=index)
            test_sorted = test.sort()
            if not test.index.equals(test_sorted.index):
                raise Exception("Data is not be sorted")
        if unique:
            if len(index) != len(index.unique()):
                raise Exception("Duplicate index entries")

    def sort(self, data, index=None):
        """
        Sort data based on the grouping index or a user-supplied index

        Parameters
        ----------
        data : ndarray or Series or DataFrame
            The data to sort.
        index : index-like, optional
            Index used to determine the sort order. Defaults to
            `self.index`.

        Returns
        -------
        out : ndarray or Series or DataFrame
            Sorted `data`, same type as the input.
        out_index : Index
            The matching sorted pandas index.

        Raises
        ------
        ValueError
            If `data` is neither a NumPy array nor a pandas
            Series/DataFrame.

        Notes
        -----
        Applies a (potentially hierarchical) sort operation on a numpy
        array or pandas series/dataframe based on the grouping index or
        a user-supplied index.
        """
        if index is None:
            index = self.index
        if data_util._is_using_ndarray_type(data, None):
            if data.ndim == 1:
                out = pd.Series(data, index=index, copy=True)
                out = out.sort_index()
            else:
                out = pd.DataFrame(data, index=index)
                out = out.sort_index(inplace=False)  # copies
            return np.array(out), out.index
        elif data_util._is_using_pandas(data, None):
            out = data
            out = out.reindex(index)  # copies?
            out = out.sort_index()
            return out, out.index
        else:
            msg = "data must be a Numpy array or a Pandas Series/DataFrame"
            raise ValueError(msg)

    def transform_dataframe(self, dataframe, function, level=0, **kwargs):
        """
        Apply function to each column, by group

        Parameters
        ----------
        dataframe : DataFrame
            Data to transform. Assumed to already have a proper index
            matching `self.index`.
        function : callable
            Function applied to each group of each column via
            ``groupby(...).apply``.
        level : int
            Index level to group by.
        **kwargs
            Additional keyword arguments passed to `function`.

        Returns
        -------
        ndarray
            Result of applying `function` by group, raveled to 1d if
            one of the output dimensions is 1.

        Raises
        ------
        Exception
            If `dataframe` does not have `self.nobs` rows.
        """
        if dataframe.shape[0] != self.nobs:
            raise Exception("dataframe does not have the same shape as index")
        out = dataframe.groupby(level=level).apply(function, **kwargs)
        if 1 in out.shape:
            return np.ravel(out)
        else:
            return np.array(out)

    def transform_array(self, array, function, level=0, **kwargs):
        """
        Apply function to each column, by group

        Parameters
        ----------
        array : array_like
            Data to transform, with `self.nobs` rows.
        function : callable
            Function applied to each group of each column.
        level : int
            Index level to group by.
        **kwargs
            Additional keyword arguments passed to `function`.

        Returns
        -------
        ndarray
            Result of applying `function` by group.

        Raises
        ------
        Exception
            If `array` does not have `self.nobs` rows.
        """
        if array.shape[0] != self.nobs:
            raise Exception("array does not have the same shape as index")
        dataframe = pd.DataFrame(array, index=self.index)
        return self.transform_dataframe(dataframe, function, level=level,
                                        **kwargs)

    def transform_slices(self, array, function, level=0, **kwargs):
        """
        Apply function to each group

        Similar to `transform_array` but does not coerce `array` to a
        DataFrame and back, and only works on a 1d or 2d numpy array.
        `function` is called as ``function(group, group_idx, **kwargs)``.

        Parameters
        ----------
        array : array_like
            1d or 2d data to transform, with `self.nobs` rows.
        function : callable
            Function called as ``function(subset, group_idx, **kwargs)``
            for each group, where `subset` is the group's rows of
            `array` and `group_idx` is the index/slice used to select
            them.
        level : int
            Index level to group by.
        **kwargs
            Additional keyword arguments passed to `function`.

        Returns
        -------
        ndarray
            Results of `function` stacked across groups, reshaped to
            2d.

        Raises
        ------
        Exception
            If `array` does not have `self.nobs` rows.
        """
        array = np.asarray(array)
        if array.shape[0] != self.nobs:
            raise Exception("array does not have the same shape as index")
        # always reset because level is given. need to refactor this.
        self.get_slices(level=level)
        processed = []
        for s in self.slices:
            if array.ndim == 2:
                subset = array[s, :]
            elif array.ndim == 1:
                subset = array[s]
            processed.append(function(subset, s, **kwargs))
        processed = np.array(processed)
        return processed.reshape(-1, processed.shape[-1])

    # TODO: this is not general needs to be a PanelGrouping object
    def dummies_time(self):
        """
        Return a sparse time-indicator matrix, using level 1 of the index

        Returns
        -------
        ndarray or sparse matrix
            Sparse indicator matrix for the second (time) index level,
            as created by :meth:`dummy_sparse`.
        """
        self.dummy_sparse(level=1)
        return self._dummies

    def dummies_groups(self, level=0):
        """
        Return a sparse group-indicator matrix for the given level

        Parameters
        ----------
        level : int
            Grouping level used to form the sparse indicator.

        Returns
        -------
        ndarray or sparse matrix
            Sparse indicator matrix for `level`, as created by
            :meth:`dummy_sparse`.
        """
        self.dummy_sparse(level=level)
        return self._dummies

    def dummy_sparse(self, level=0):
        """
        Create a sparse indicator from a group array with integer labels

        Parameters
        ----------
        level : int
            Grouping level used to form the sparse indicator.

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

        """
        indi = dummy_sparse(self.labels[level])
        self._dummies = indi
