"""
Base classes for statistical test results

Created on Mon Apr 22 14:03:21 2013

Author: Josef Perktold
"""
import warnings

import numpy as np

from statsmodels.tools.testing import Holder


class HolderTuple(Holder):
    """
    Holder class with indexing

    .. deprecated:: 0.15
        ``HolderTuple`` is no longer used internally by statsmodels. Result
        classes that used to subclass ``HolderTuple`` have been replaced by
        documented ``NamedTuple`` classes; those that need the same 2-tuple
        unpacking behavior are decorated with
        :func:`~statsmodels.stats.base.compat_2tuple_unpack`.
        ``HolderTuple`` will be removed after statsmodels 0.16 is released.

    Parameters
    ----------
    tuple_ : tuple of str, optional
        Names of the attributes, in order, that should be collected into
        ``self.tuple``. If None, ``self.tuple`` is set to
        ``(self.statistic, self.pvalue)``.
    **kwds : dict
        Keyword arguments that are set as attributes, as in ``Holder``.
    """

    def __init__(self, tuple_=None, **kwds):
        warnings.warn(
            "HolderTuple is deprecated and is no longer used internally by "
            "statsmodels. It will be removed after statsmodels 0.16 is "
            "released. Use a documented NamedTuple result class instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(**kwds)
        if tuple_ is not None:
            self.tuple = tuple(getattr(self, att) for att in tuple_)
        else:
            self.tuple = (self.statistic, self.pvalue)

    def __iter__(self):
        yield from self.tuple

    def __getitem__(self, idx):
        return self.tuple[idx]

    def __len__(self):
        return len(self.tuple)

    def __array__(self, dtype=None, copy=None):
        copy = copy if copy is not None else True
        return np.array(list(self.tuple), dtype=dtype, copy=copy)


def compat_2tuple_unpack(*field_names):
    """
    Class decorator giving a NamedTuple HolderTuple-compatible unpacking.

    Several results classes that used to subclass ``HolderTuple`` have been
    replaced by plain ``NamedTuple`` classes with many more fields.
    ``HolderTuple`` only ever unpacked into a fixed, short tuple (by
    default ``(statistic, pvalue)``), so existing code may do
    ``statistic, pvalue = result``. Decorating the replacement NamedTuple
    with ``@compat_2tuple_unpack("statistic", "pvalue")`` keeps that
    working (with a deprecation warning) during a transition period,
    instead of unpacking every field and raising ``ValueError``.

    This only overrides ``__iter__``, which controls unpacking, iteration,
    ``list(result)`` and ``tuple(result)``. Indexing (``result[0]``),
    attribute access, and equality are unaffected since they operate on
    the underlying tuple directly rather than through ``__iter__``.

    ``_asdict``, ``_replace``, and ``__getnewargs__`` (used by ``pickle``
    and ``copy.deepcopy``) are also overridden to bypass the restricted
    ``__iter__`` and continue operating on all fields; without this they
    would silently truncate to just `field_names`, since the standard
    library implementations of those methods iterate over ``self``.

    Parameters
    ----------
    *field_names : str
        Names of the fields that unpacking should yield, in order. These
        must be a prefix consistent with how many values the replaced
        ``HolderTuple`` used to unpack into (commonly ``("statistic",
        "pvalue")``, or a single field for a ``HolderTuple`` constructed
        with a custom ``tuple_`` argument).
    """

    def decorator(cls):
        names = ", ".join(field_names)
        first = field_names[0]

        def __iter__(self):
            warnings.warn(
                f"Unpacking {type(self).__name__} currently returns only "
                f"({names}) for backwards compatibility with the "
                "Holder-based API it replaced. Starting in statsmodels "
                "0.16, unpacking will return all fields like a standard "
                "tuple. Use named attribute access instead, e.g., "
                f"``result.{first}``.",
                FutureWarning,
                stacklevel=3,
            )
            for name in field_names:
                yield getattr(self, name)

        def _asdict(self):
            return dict(zip(self._fields, tuple.__iter__(self), strict=True))

        def _replace(self, /, **kwds):
            result = self._make(map(kwds.pop, self._fields, tuple.__iter__(self)))
            if kwds:
                raise ValueError(f"Got unexpected field names: {list(kwds)!r}")
            return result

        def __getnewargs__(self):
            return tuple(tuple.__iter__(self))

        cls.__iter__ = __iter__
        cls._asdict = _asdict
        cls._replace = _replace
        cls.__getnewargs__ = __getnewargs__
        return cls

    return decorator


class AllPairsResults:
    """
    Results class for pairwise comparisons, based on p-values

    Parameters
    ----------
    pvals_raw : array_like, 1-D
        p-values from a pairwise comparison test
    all_pairs : list of tuples
        list of indices, one pair for each comparison
    multitest_method : str
        method that is used by default for p-value correction. This is used
        as default by the methods like if the multiple-testing method is not
        specified as argument.
    levels : {list[str], None}
        optional names of the levels or groups
    n_levels : None or int
        If None, then the number of levels or groups is inferred from the
        other arguments. It can be explicitly specified, if the inferred
        number is incorrect.

    Notes
    -----
    This class can also be used for other pairwise comparisons, for example
    comparing several treatments to a control (as in Dunnet's test).

    """

    def __init__(self, pvals_raw, all_pairs, multitest_method="hs",
                 levels=None, n_levels=None):
        self.pvals_raw = pvals_raw
        self.all_pairs = all_pairs
        if n_levels is None:
            # for all_pairs nobs*(nobs-1)/2
            self.n_levels = np.max(all_pairs) + 1
        else:
            self.n_levels = n_levels

        self.multitest_method = multitest_method
        self.levels = levels
        if levels is None:
            self.all_pairs_names = [f"{pairs}" for pairs in all_pairs]
        else:
            self.all_pairs_names = [f"{levels[pairs[0]]}-{levels[pairs[1]]}"
                                    for pairs in all_pairs]

    def pval_corrected(self, method=None):
        """
        p-values corrected for multiple testing problem

        This uses the default p-value correction of the instance stored in
        ``self.multitest_method`` if method is None.

        Parameters
        ----------
        method : str, optional
            p-value correction method to use. If None, the default method
            stored in ``self.multitest_method`` is used.

        Returns
        -------
        ndarray
            Corrected p-values.
        """
        import statsmodels.stats.multitest as smt
        if method is None:
            method = self.multitest_method
        # TODO: breaks with method=None
        return smt.multipletests(self.pvals_raw, method=method)[1]

    def __str__(self):
        return self.summary()

    def pval_table(self):
        """
        create a (n_levels, n_levels) array with corrected p_values

        this needs to improve, similar to R pairwise output

        Returns
        -------
        ndarray
            Array of shape (n_levels, n_levels) with corrected p-values.
        """
        k = self.n_levels
        pvals_mat = np.zeros((k, k))
        # if we do not assume we have all pairs
        pvals_mat[list(zip(*self.all_pairs, strict=True))] = self.pval_corrected()
        return pvals_mat

    def summary(self):
        """
        returns text summarizing the results

        uses the default pvalue correction of the instance stored in
        ``self.multitest_method``

        Returns
        -------
        str
            Summary text of the pairwise comparison results.
        """
        import statsmodels.stats.multitest as smt
        maxlevel = max(len(ss) for ss in self.all_pairs_names)

        text = (f"Corrected p-values using {smt.multitest_methods_names[self.multitest_method]} p-value correction\n\n")
        text += "Pairs" + (" " * (maxlevel - 5 + 1)) + "p-values\n"
        text += "\n".join(f"{pairs}  {pv:6.4g}" for (pairs, pv) in
                          zip(self.all_pairs_names, self.pval_corrected(), strict=True))
        return text
