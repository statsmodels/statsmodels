"""
Base classes for statistical test results

Created on Mon Apr 22 14:03:21 2013

Author: Josef Perktold
"""
from collections.abc import Iterator
from typing import ClassVar, Generic, TypeVar
import warnings

import numpy as np

from statsmodels.tools.testing import Holder


class HolderTuple(Holder):
    """
    Holder class with indexing

    .. deprecated:: 0.15
        ``HolderTuple`` is no longer used internally by statsmodels. Result
        classes that used to subclass ``HolderTuple`` have been replaced by
        documented dataclass (or ``NamedTuple``) result classes; those that
        need the same limited ``(statistic, pvalue)`` unpacking are frozen
        dataclasses that subclass
        :class:`~statsmodels.stats.base.LimitedIterationMixin`.
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


_V = TypeVar("_V")


class LimitedIterationMixin(Generic[_V]):
    """
    Base class giving a dataclass tuple-like access to some fields.

    Several result classes were changed from ``NamedTuple`` subclasses to
    frozen :func:`~dataclasses.dataclass` instances, so that additional
    fields can be added over time without changing what iteration and
    unpacking return. A dataclass is not iterable and cannot be indexed, so
    this mixin restores limited, tuple-like access to a fixed subset of the
    fields, matching the ``(statistic, pvalue)`` (or single-value) unpacking
    of the result classes they replaced. Concretely it adds

    - ``__iter__``, so that ``a, b = result`` and ``list(result)`` yield the
      selected fields in order,
    - ``__getitem__``, so that ``result[0]`` indexes into the same fields,
      and
    - ``__len__``, returning the number of selected fields.

    All other fields remain available through attribute access only, which
    is the preferred API. Unlike the ``NamedTuple`` classes these replaced,
    iteration and indexing are permanently restricted to ``_iter_fields`` and
    do not emit a warning.

    Subclassing a plain base class (rather than applying a decorator that
    attaches ``__iter__``/``__getitem__``/``__len__`` at runtime) keeps these
    methods, and their return type, visible to static type checkers: ``a, b
    = result`` and ``result[0]`` are typed as ``_V``, and ``len(result)`` is
    recognized. Parametrize the mixin with the common type of the selected
    fields and set ``_iter_fields``::

        @dataclass(frozen=True, slots=True)
        class SomeResult(LimitedIterationMixin[float]):
            _iter_fields: ClassVar[tuple[str, ...]] = ("statistic", "pvalue")

            statistic: float
            pvalue: float
            extra: float

    ``_iter_fields`` must be annotated ``ClassVar`` so that ``@dataclass``
    does not treat it as a field.
    """

    __slots__ = ()
    _iter_fields: ClassVar[tuple[str, ...]]

    def __iter__(self) -> Iterator[_V]:
        for name in self._iter_fields:
            yield getattr(self, name)

    def __getitem__(self, index: int) -> _V:
        return tuple(self)[index]

    def __len__(self) -> int:
        return len(self._iter_fields)


class AllPairsResults:
    """
    Results class for pairwise comparisons, based on p-values

    Parameters
    ----------
    pvals_raw : array_like, 1-D
        p-values from a pairwise comparison test
    all_pairs : list of tuples
        list of indices, one pair for each comparison
    multitest_method : str, optional
        method that is used by default for p-value correction. This is used
        as default by the methods like if the multiple-testing method is not
        specified as argument.
    levels : list[str] or None, optional
        optional names of the levels or groups
    n_levels : None or int, optional
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
        rows, cols = zip(*self.all_pairs, strict=True)
        pvals_mat[rows, cols] = self.pval_corrected()
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
