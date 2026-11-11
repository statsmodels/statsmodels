"""
Inter Rater Agreement

Created on Thu Dec 06 22:57:56 2012
Author: Josef Perktold
License: BSD-3

Notes
-----
Contains fleiss_kappa, cohens_kappa and krippendorff_alpha.

aggregate_raters:
    helper function to get data into fleiss_kappa format
to_table:
    helper function to create contingency table, can be used for cohens_kappa

TODO:
standard errors and hypothesis tests for fleiss_kappa
other statistics and tests,
   in R package irr, SAS has more
inconsistent internal naming, changed variable names as I added more
   functionality
convenience functions to create required data format from raw data
   DONE

References
----------
Wikipedia: kappa's initially based on these two pages
    https://en.wikipedia.org/wiki/Fleiss%27_kappa
    https://en.wikipedia.org/wiki/Cohen's_kappa
SAS-Manual : formulas for cohens_kappa, especially variances
see also R package irr
"""

import numpy as np
from scipy import stats  # get rid of this? need only norm.sf

from statsmodels.tools.validation import array_like, string_like


class ResultsBunch(dict):

    template = "%r"

    def __init__(self, **kwds):
        dict.__init__(self, kwds)
        self.__dict__ = self
        self._initialize()

    def _initialize(self):
        pass

    def __str__(self):
        return self.template % self


def _int_ifclose(x, dec=1, width=4):
    """
    Create a result string for an int or float

    Only dec=1 and width=4 is implemented.

    Parameters
    ----------
    x : int or float
        Value to format.
    dec : int, optional
        Number of decimals to print if x is not an integer.
    width : int, optional
        Width of string.

    Returns
    -------
    xint : int or float
        x is converted to int if it is within 1e-14 of an integer.
    x_string : str
        x formatted as string, either '%4d' or '%4.1f'.
    """
    xint = round(x)
    if np.max(np.abs(xint - x)) < 1e-14:
        return xint, f"{xint:4d}"
    else:
        return x, f"{x:4.1f}"


def aggregate_raters(data, n_cat=None):
    """
    Convert raw data with shape (subject, rater) to (subject, cat_counts)

    Brings data into correct format for fleiss_kappa.

    bincount will raise exception if data cannot be converted to integer.

    Parameters
    ----------
    data : array_like, 2-Dim
        Data containing category assignment with subjects in rows and raters
        in columns.
    n_cat : None or int, optional
        If None, then the data is converted to integer categories,
        0,1,2,...,n_cat-1. Because of the relabeling only category levels
        with non-zero counts are included.
        If this is an integer, then the category levels in the data are
        already assumed to be in integers, 0,1,2,...,n_cat-1. In this case,
        the returned array may contain columns with zero count, if no
        subject has been categorized with this level.

    Returns
    -------
    arr : ndarray, (n_rows, n_cat)
        Contains counts of raters that assigned a category level to
        individuals. Subjects are in rows, category levels in columns.
    categories : ndarray, (n_category_levels,)
        Contains the category levels.
    """
    data = np.asarray(data)
    n_rows = data.shape[0]
    if n_cat is None:
        # I could add int conversion (reverse_index) to np.unique
        cat_uni, cat_int = np.unique(data.ravel(), return_inverse=True)
        n_cat = len(cat_uni)
        data_ = cat_int.reshape(data.shape)
    else:
        cat_uni = np.arange(n_cat)  # for return only, assumed cat levels
        data_ = data

    tt = np.zeros((n_rows, n_cat), int)
    for idx, row in enumerate(data_):
        ro = np.bincount(row)
        tt[idx, :len(ro)] = ro

    return tt, cat_uni


def to_table(data, bins=None):
    """
    Convert raw data with shape (subject, rater) to (rater1, rater2)

    Brings data into correct format for cohens_kappa.

    Parameters
    ----------
    data : array_like, 2-Dim
        Data containing category assignment with subjects in rows and raters
        in columns.
    bins : None, int or tuple of array_like, optional
        If None, then the data is converted to integer categories,
        0,1,2,...,n_cat-1. Because of the relabeling only category levels
        with non-zero counts are included.
        If this is an integer, then the category levels in the data are
        already assumed to be in integers, 0,1,2,...,n_cat-1. In this case,
        the returned array may contain columns with zero count, if no
        subject has been categorized with this level.
        If bins are a tuple of two array_like, then the bins are directly
        used by ``numpy.histogramdd``. This is useful if we want to merge
        categories.

    Returns
    -------
    arr : ndarray, (n_cat, n_cat)
        Contingency table that contains counts of category level with rater1
        in rows and rater2 in columns.
    bins : ndarray
        The bin edges used to construct the contingency table.

    Notes
    -----
    No NaN handling, delete rows with missing values.

    This works also for more than two raters. In that case the dimension of
    the resulting contingency table is the same as the number of raters
    instead of 2-dimensional.
    """

    data = np.asarray(data)
    if data.ndim != 2:
        # without the 2-d check a 1-d table crashed with a bare
        # "not enough values to unpack" from the shape assignment
        raise ValueError(
            f"data must be a 2-d array of subjects by raters, got a "
            f"{data.ndim}-d array"
        )
    n_rows, n_cols = data.shape
    if bins is None:
        # I could add int conversion (reverse_index) to np.unique
        cat_uni, cat_int = np.unique(data.ravel(), return_inverse=True)
        n_cat = len(cat_uni)
        data_ = cat_int.reshape(data.shape)
        bins_ = np.arange(n_cat+1) - 0.5
        # alternative implementation with double loop
        # tt = np.asarray([[(x == [i,j]).all(1).sum() for j in cat_uni]
        #                 for i in cat_uni] )
        # other altervative: unique rows and bincount
    elif np.isscalar(bins):
        bins_ = np.arange(bins+1) - 0.5
        data_ = data
    else:
        bins_ = bins
        data_ = data

    tt = np.histogramdd(data_, (bins_,)*n_cols)

    return tt[0], bins_


def fleiss_kappa(table, method="fleiss"):
    """
    Fleiss' and Randolph's kappa multi-rater agreement measure

    Parameters
    ----------
    table : array_like, 2-D
        Assumes subjects in rows, and categories in columns. Convert raw
        data into this format by using
        :func:`statsmodels.stats.inter_rater.aggregate_raters`.
    method : str, optional
        Method 'fleiss' returns Fleiss' kappa which uses the sample margin
        to define the chance outcome.
        Method 'randolph' or 'uniform' (only first 4 letters are needed)
        returns Randolph's (2005) multirater kappa which assumes a uniform
        distribution of the categories to define the chance outcome.

    Returns
    -------
    kappa : float
        Fleiss's or Randolph's kappa statistic for inter rater agreement.

    Notes
    -----
    No variance or hypothesis tests yet.

    Interrater agreement measures like Fleiss's kappa measure agreement
    relative to chance agreement. Different authors have proposed ways of
    defining these chance agreements. Fleiss' is based on the marginal
    sample distribution of categories, while Randolph uses a uniform
    distribution of categories as benchmark. Warrens (2010) showed that
    Randolph's kappa is always larger or equal to Fleiss' kappa. Under some
    commonly observed condition, Fleiss' and Randolph's kappa provide lower
    and upper bounds for two similar kappa_like measures by Light (1971) and
    Hubert (1977).

    References
    ----------
    Wikipedia https://en.wikipedia.org/wiki/Fleiss%27_kappa

    Fleiss, Joseph L. 1971. "Measuring Nominal Scale Agreement among Many
    Raters." Psychological Bulletin 76 (5): 378-82.
    https://doi.org/10.1037/h0031619.

    Randolph, Justus J. 2005 "Free-Marginal Multirater Kappa (multirater
    K [free]): An Alternative to Fleiss' Fixed-Marginal Multirater Kappa."
    Presented at the Joensuu Learning and Instruction Symposium, vol. 2005
    https://eric.ed.gov/?id=ED490661

    Warrens, Matthijs J. 2010. "Inequalities between Multi-Rater Kappas."
    Advances in Data Analysis and Classification 4 (4): 271-86.
    https://doi.org/10.1007/s11634-010-0073-4.
    """

    table = 1.0 * np.asarray(table)
    if table.ndim != 2:
        # without the 2-d check a 1-d table crashed with a bare
        # "not enough values to unpack" from the shape assignment
        raise ValueError(
            f"table must be a 2-d array of subjects by categories, got a "
            f"{table.ndim}-d array"
        )
    n_sub, n_cat = table.shape
    n_total = table.sum()
    n_rater = table.sum(1)
    n_rat = n_rater.max()
    # assume fully ranked
    assert n_total == n_sub * n_rat

    # marginal frequency  of categories
    p_cat = table.sum(0) / n_total

    table2 = table * table
    p_rat = (table2.sum(1) - n_rat) / (n_rat * (n_rat - 1.))
    p_mean = p_rat.mean()

    if method == "fleiss":
        p_mean_exp = (p_cat*p_cat).sum()
    elif method.startswith(("rand", "unif")):
        p_mean_exp = 1 / n_cat

    kappa = (p_mean - p_mean_exp) / (1 - p_mean_exp)
    return kappa


def krippendorff_alpha(data, level="nominal"):
    r"""
    Compute Krippendorff's alpha reliability coefficient.

    Agreement among any number of raters, for nominal, ordinal, interval or
    ratio data, with any number of missing ratings.

    .. versionadded:: 0.15.1

    Parameters
    ----------
    data : array_like
        Numeric ratings in a 2-D array with subjects in rows and raters in
        columns, the layout used by
        :func:`statsmodels.stats.inter_rater.aggregate_raters`. Missing
        ratings are coded as ``nan``. Nominal categories can be any numeric
        codes.
    level : {"nominal", "ordinal", "interval", "ratio"}, optional
        Level of measurement of the ratings. It defines the squared difference
        :math:`\delta^2_{ck}` between two ratings :math:`c` and :math:`k`.
        The default is "nominal".

        - "nominal": 0 if :math:`c = k` and 1 otherwise.
        - "ordinal": :math:`(\sum_{g=c}^{k} n_g - (n_c + n_k) / 2)^2`, which
          depends on how often the ratings between :math:`c` and :math:`k`
          are used, not on their numeric distance.
        - "interval": :math:`(c - k)^2`.
        - "ratio": :math:`((c - k) / (c + k))^2`. The ratings must be
          nonnegative.

    Returns
    -------
    float
        Krippendorff's alpha. It is 1 for perfect agreement and 0 for
        agreement at the level of chance. It is ``nan`` if no subject has two
        or more ratings, or if all those ratings are equal, because alpha is
        0 / 0 then.

    See Also
    --------
    fleiss_kappa : Fleiss' kappa for nominal ratings without missing values.
    cohens_kappa : Cohen's kappa for two raters.

    Notes
    -----
    Only subjects with at least two ratings are used. Following Krippendorff
    (2011), the ratings within each subject :math:`u` with :math:`m_u`
    ratings are tabulated in the coincidence matrix :math:`o_{ck}`: each of
    the :math:`m_u (m_u - 1)` ordered pairs of ratings adds
    :math:`1 / (m_u - 1)` to the cell of its two values. With the margins
    :math:`n_c = \sum_k o_{ck}` and :math:`n = \sum_c n_c`,

    .. math::

       \alpha = 1 - (n - 1)
       \frac{\sum_c \sum_k o_{ck} \delta^2_{ck}}
       {\sum_c \sum_k n_c n_k \delta^2_{ck}}

    For nominal data without missing ratings, alpha is a function of
    Fleiss' kappa, :math:`\alpha = 1 - (1 - \kappa) (n - 1) / n` (see
    :func:`fleiss_kappa`). For two raters this is Scott's pi.

    References
    ----------
    Krippendorff, Klaus. 2011. "Computing Krippendorff's Alpha-Reliability."
    Annenberg School for Communication, University of Pennsylvania.
    https://repository.upenn.edu/asc_papers/43

    Hayes, Andrew F., and Klaus Krippendorff. 2007. "Answering the Call for a
    Standard Reliability Measure for Coding Data." Communication Methods and
    Measures 1 (1): 77-89. https://doi.org/10.1080/19312450709336664

    Examples
    --------
    The example of Krippendorff (2011): 12 subjects rated by 4 raters, with
    7 missing ratings.

    >>> import numpy as np
    >>> from statsmodels.stats.inter_rater import krippendorff_alpha
    >>> nan = np.nan
    >>> data = [[1, 1, nan, 1], [2, 2, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3],
    ...         [2, 2, 2, 2], [1, 2, 3, 4], [4, 4, 4, 4], [1, 1, 2, 1],
    ...         [2, 2, 2, 2], [nan, 5, 5, 5], [nan, nan, 1, 1],
    ...         [nan, 3, nan, nan]]
    >>> round(krippendorff_alpha(data), 3)
    0.743
    >>> round(krippendorff_alpha(data, level="interval"), 3)
    0.849
    """
    data = array_like(data, "data", maxdim=2)
    if data.ndim != 2:
        raise ValueError(
            f"data must be a 2-d array of subjects by raters, got a {data.ndim}-d array"
        )
    level = string_like(
        level, "level", options=("nominal", "ordinal", "interval", "ratio")
    )
    if np.isinf(data).any():
        raise ValueError("data must not contain inf; code missing ratings as nan")
    if level == "ratio" and (data < 0).any():
        raise ValueError("ratings must be nonnegative for level='ratio'")

    observed = ~np.isnan(data)
    n_ratings = observed.sum(1)
    # a subject with fewer than two ratings has no pair of ratings to compare
    pairable = n_ratings >= 2
    data = data[pairable]
    observed = observed[pairable]
    n_ratings = n_ratings[pairable]
    if data.shape[0] == 0:
        return np.nan

    values, codes = np.unique(data[observed], return_inverse=True)
    if len(values) < 2:
        return np.nan
    # counts[u, c]: number of ratings of subject u equal to values[c]
    counts = np.zeros((data.shape[0], len(values)))
    np.add.at(counts, (np.nonzero(observed)[0], codes), 1)

    # coincidence matrix: sum over subjects of
    # (n_uc * n_uk - [c == k] * n_uc) / (m_u - 1)
    weights = counts / (n_ratings - 1)[:, None]
    coincidence = counts.T @ weights - np.diag(weights.sum(0))
    n_c = coincidence.sum(1)
    n = n_c.sum()

    if level == "nominal":
        delta2 = 1.0 - np.eye(len(values))
    elif level == "ordinal":
        # with midpoints cumsum(n)_c - n_c / 2, the difference of two
        # midpoints is sum_{g=c}^{k} n_g - (n_c + n_k) / 2
        midpoint = np.cumsum(n_c) - n_c / 2
        delta2 = np.subtract.outer(midpoint, midpoint) ** 2
    elif level == "interval":
        delta2 = np.subtract.outer(values, values) ** 2
    else:
        diff = np.subtract.outer(values, values)
        total = np.add.outer(values, values)
        # c + k is 0 only for c = k = 0, where the difference is 0
        ratio = np.divide(diff, total, out=np.zeros_like(diff), where=total > 0)
        delta2 = ratio**2

    d_observed = (coincidence * delta2).sum()
    d_expected = (np.outer(n_c, n_c) * delta2).sum()
    return float(1.0 - (n - 1) * d_observed / d_expected)


def cohens_kappa(table, weights=None, return_results=True, wt=None):
    '''
    Compute Cohen's kappa with variance and equal-zero test

    Parameters
    ----------
    table : array_like, 2-Dim
        Square array with results of two raters, one rater in rows, second
        rater in columns.
    weights : array_like, optional
        The interpretation of weights depends on the wt argument.
        If both are None, then the simple kappa is computed.
        See wt for the case when wt is not None.
        If weights is two dimensional, then it is directly used as a weight
        matrix. For computing the variance of kappa, the maximum of the
        weights is assumed to be smaller or equal to one.
        TODO: fix conflicting definitions in the 2-Dim case for.
    return_results : bool, optional
        If True (default), then an instance of KappaResults is returned.
        If False, then only kappa is computed and returned.
    wt : {None, 'ca', 'linear', 'fc', 'quadratic', 'toeplitz'}, optional
        If wt and weights are None, then the simple kappa is computed.
        If wt is given, but weights is None, then the weights are set to
        be [0, 1, 2, ..., k].
        If weights is a one-dimensional array, then it is used to construct
        the weight matrix given the following options.

        wt in ['linear', 'ca' or None] : use linear weights, Cicchetti-Allison
            actual weights are linear in the score "weights" difference
        wt in ['quadratic', 'fc'] : use quadratic weights, Fleiss-Cohen
            actual weights are squared in the score "weights" difference
        wt = 'toeplitz' : weight matrix is constructed as a toeplitz matrix
            from the one dimensional weights.

    Returns
    -------
    results or kappa
        If return_results is True (default), then a results instance with
        all statistics is returned.
        If return_results is False, then only kappa is calculated and
        returned.

    Notes
    -----
    There are two conflicting definitions of the weight matrix, Wikipedia
    versus SAS manual. However, the computation are invariant to rescaling
    of the weights matrix, so there is no difference in the results.

    Weights for 'linear' and 'quadratic' are interpreted as scores for the
    categories, the weights in the computation are based on the pairwise
    difference between the scores.
    Weights for 'toeplitz' are interpreted as weighted distance. The
    distance only depends on how many levels apart two entries in the table
    are but not on the levels themselves.

    example:

    weights = '0, 1, 2, 3' and wt is either linear or toeplitz means that the
    weighting only depends on the simple distance of levels.

    weights = '0, 0, 1, 1' and wt = 'linear' means that the first two levels
    are zero distance apart and the same for the last two levels. This is
    the same as forming two aggregated levels by merging the first two and
    the last two levels, respectively.

    weights = [0, 1, 2, 3] and wt = 'quadratic' is the same as squaring these
    weights and using wt = 'toeplitz'.

    References
    ----------
    Wikipedia
    SAS Manual
    '''
    table = np.asarray(table, float)  # avoid integer division
    if table.ndim != 2 or table.shape[0] != table.shape[1]:
        # without this a 1-d table crashed with a raw AxisError from the
        # diag/axis-1 operations below
        raise ValueError(
            f"table must be a square 2-d array, got shape {table.shape}"
        )
    agree = np.diag(table).sum()
    nobs = table.sum()
    probs = table / nobs
    freqs = probs  # TODO: rename to use freqs instead of probs for observed
    probs_diag = np.diag(probs)
    freq_row = table.sum(1) / nobs
    freq_col = table.sum(0) / nobs
    prob_exp = freq_col * freq_row[:, None]
    assert np.allclose(prob_exp.sum(), 1)
    # print prob_exp.sum()
    agree_exp = np.diag(prob_exp).sum()  # need for kappa_max
    if weights is None and wt is None:
        kind = "Simple"
        kappa = (agree / nobs - agree_exp) / (1 - agree_exp)

        if return_results:
            # variance
            term_a = probs_diag * (1 - (freq_row + freq_col) * (1 - kappa))**2
            term_a = term_a.sum()
            term_b = probs * (freq_col[:, None] + freq_row)**2
            d_idx = np.arange(table.shape[0])
            term_b[d_idx, d_idx] = 0   # set diagonal to zero
            term_b = (1 - kappa)**2 * term_b.sum()
            term_c = (kappa - agree_exp * (1-kappa))**2
            var_kappa = (term_a + term_b - term_c) / (1 - agree_exp)**2 / nobs
            # term_c = freq_col * freq_row[:, None] * (freq_col + freq_row[:,None])
            term_c = freq_col * freq_row * (freq_col + freq_row)
            var_kappa0 = (agree_exp + agree_exp**2 - term_c.sum())
            var_kappa0 /= (1 - agree_exp)**2 * nobs

    else:
        if weights is None:
            weights = np.arange(table.shape[0])
        # weights follows the Wikipedia definition, not the SAS, which is 1 -
        kind = "Weighted"
        weights = np.asarray(weights, float)
        if weights.ndim == 1:
            if wt in ["ca", "linear", None]:
                weights = np.abs(weights[:, None] - weights) /  \
                           (weights[-1] - weights[0])
            elif wt in ["fc", "quadratic"]:
                weights = (weights[:, None] - weights)**2 /  \
                           (weights[-1] - weights[0])**2
            elif wt == "toeplitz":
                # assume toeplitz structure
                from scipy.linalg import toeplitz

                # weights = toeplitz(np.arange(table.shape[0]))
                weights = toeplitz(weights)
            else:
                raise ValueError("wt option is not known")
        else:
            rows, cols = table.shape
            if (table.shape != weights.shape):
                raise ValueError("weights are not square")
        # this is formula from Wikipedia
        kappa = 1 - (weights * table).sum() / nobs / (weights * prob_exp).sum()
        # TODO: add var_kappa for weighted version
        if return_results:
            var_kappa = np.nan
            var_kappa0 = np.nan
            # switch to SAS manual weights, problem if user specifies weights
            # w is negative in some examples,
            # but weights is scale invariant in examples and rough check of source
            w = 1. - weights
            w_row = (freq_col * w).sum(1)
            w_col = (freq_row[:, None] * w).sum(0)
            agree_wexp = (w * freq_col * freq_row[:, None]).sum()
            term_a = freqs * (w - (w_col + w_row[:, None]) * (1 - kappa))**2
            fac = 1. / ((1 - agree_wexp)**2 * nobs)
            var_kappa = term_a.sum() - (kappa - agree_wexp * (1 - kappa))**2
            var_kappa *= fac

            freqse = freq_col * freq_row[:, None]
            var_kappa0 = (freqse * (w - (w_col + w_row[:, None]))**2).sum()
            var_kappa0 -= agree_wexp**2
            var_kappa0 *= fac

    kappa_max = (np.minimum(freq_row, freq_col).sum() - agree_exp) / \
                (1 - agree_exp)

    if return_results:
        res = KappaResults(
            kind=kind,
            kappa=kappa,
            kappa_max=kappa_max,
            weights=weights,
            var_kappa=var_kappa,
            var_kappa0=var_kappa0
        )
        return res
    else:
        return kappa


_kappa_template = """\
                  %(kind)s Kappa Coefficient
              --------------------------------
              Kappa                     %(kappa)6.4f
              ASE                       %(std_kappa)6.4f
            %(alpha_ci)s%% Lower Conf Limit      %(kappa_low)6.4f
            %(alpha_ci)s%% Upper Conf Limit      %(kappa_upp)6.4f

                 Test of H0: %(kind)s Kappa = 0

              ASE under H0              %(std_kappa0)6.4f
              Z                         %(z_value)6.4f
              One-sided Pr >  Z         %(pvalue_one_sided)6.4f
              Two-sided Pr > |Z|        %(pvalue_two_sided)6.4f
"""

"""
                   Weighted Kappa Coefficient
              --------------------------------
              Weighted Kappa            0.4701
              ASE                       0.1457
              95% Lower Conf Limit      0.1845
              95% Upper Conf Limit      0.7558

               Test of H0: Weighted Kappa = 0

              ASE under H0              0.1426
              Z                         3.2971
              One-sided Pr >  Z         0.0005
              Two-sided Pr > |Z|        0.0010
"""


class KappaResults(ResultsBunch):
    """
    Results for Cohen's kappa

    Attributes
    ----------
    kind : str
        "Simple" for the unweighted kappa, or "Weighted" if a weight
        matrix was used.
    kappa : float
        Cohen's kappa.
    kappa_max : float
        Maximum value that kappa can attain given the observed marginal
        frequencies.
    weights : ndarray or None
        The weight matrix used to compute the weighted kappa. ``None``
        for the simple, unweighted kappa.
    var_kappa : float
        Variance of kappa.
    std_kappa : float
        Standard deviation of kappa.
    alpha : float
        One-sided probability for confidence interval.
    kappa_low : float
        Lower (1-alpha) confidence limit.
    kappa_upp : float
        Upper (1-alpha) confidence limit.
    var_kappa0 : float
        Variance of kappa under H0: kappa=0.
    std_kappa0 : float
        Standard deviation of kappa under H0: kappa=0.
    z_value : float
        Test statistic for H0: kappa=0, is standard normal distributed.
    pvalue_one_sided : float
        One sided p-value for H0: kappa=0 and H1: kappa>0.
    pvalue_two_sided : float
        Two sided p-value for H0: kappa=0 and H1: kappa!=0.
    distribution_kappa : scipy.stats.norm
        Asymptotic normal distribution of kappa.
    distribution_zero_null : scipy.stats.norm
        Asymptotic normal distribution of kappa under H0: kappa=0.

    Notes
    -----
    The confidence interval for kappa and the statistics for the test of
    H0: kappa=0 are based on the asymptotic normal distribution of kappa.
    """

    template = _kappa_template

    def _initialize(self):
        if "alpha" not in self:
            self["alpha"] = 0.025
            self["alpha_ci"] = _int_ifclose(100 - 0.025 * 200)[1]
        self["std_kappa"] = np.sqrt(self["var_kappa"])
        self["std_kappa0"] = np.sqrt(self["var_kappa0"])

        self["z_value"] = self["kappa"] / self["std_kappa0"]

        self["pvalue_one_sided"] = stats.norm.sf(self["z_value"])
        self["pvalue_two_sided"] = stats.norm.sf(np.abs(self["z_value"])) * 2

        delta = stats.norm.isf(self["alpha"]) * self["std_kappa"]
        self["kappa_low"] = self["kappa"] - delta
        self["kappa_upp"] = self["kappa"] + delta
        self["distribution_kappa"] = stats.norm(
            loc=self["kappa"], scale=self["std_kappa"]
        )
        self["distribution_zero_null"] = stats.norm(loc=0, scale=self["std_kappa0"])

    def __str__(self):
        return self.template % self
