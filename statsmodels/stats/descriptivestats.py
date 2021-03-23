from statsmodels.compat.pandas import Appender, is_numeric_dtype
from statsmodels.compat.python import lmap, lrange

from typing import Sequence, Union
import warnings

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_categorical_dtype
from scipy import stats

from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.docstring import Docstring, Parameter
from statsmodels.tools.validation import (
    array_like,
    bool_like,
    float_like,
    int_like,
)

DEPRECATION_MSG = """/
``Describe`` has been deprecated in favor of ``Description`` and it's
simplified functional version, ``describe``. ``Describe`` will be removed
after 0.13.
"""

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
QUANTILES = np.array(PERCENTILES) / 100.0


def pd_ptp(df):
    return df.max() - df.min()


def pd_percentiles(df):
    return df.quantiles(QUANTILES)


PANDAS = {
    "obs": lambda df: df.count(),
    "mean": lambda df: df.mean(),
    "std": lambda df: df.std(),
    "max": lambda df: df.max(),
    "min": lambda df: df.min(),
    "mode": lambda df: df.mode(),
    "ptp": pd_ptp,
    "var": lambda df: df.var(),
    "skew": lambda df: df.skewness(),
    "uss": lambda df: (df ** 2).sum(),
    "kurtosis": lambda df: df.kurtosis(),
    "percentiles": pd_percentiles,
}


def nancount(x, axis=0):
    return (1 - np.isnan(x)).sum(axis=axis)


def nanptp(arr, axis=0):
    return np.nanmax(arr, axis=axis) - np.nanmin(arr, axis=axis)


def nanuss(arr, axis=0):
    return np.nansum(arr ** 2, axis=axis)


def nanpercentile(arr, axis=0):
    return np.nanpercentile(arr, PERCENTILES, axis=axis)


def nankurtosis(arr, axis=0):
    return stats.kurtosis(arr, axis=axis, nan_policy="omit")


def nanskewness(arr, axis=0):
    return stats.skew(arr, axis=axis, nan_policy="omit")


MISSING = {
    "obs": nancount,
    "mean": np.nanmean,
    "std": np.nanstd,
    "max": np.nanmax,
    "min": np.nanmin,
    "ptp": nanptp,
    "var": np.nanvar,
    "skew": nanskewness,
    "uss": nanuss,
    "kurtosis": nankurtosis,
    "percentiles": nanpercentile,
}


def _kurtosis(a):
    """
    wrapper for scipy.stats.kurtosis that returns nan instead of raising Error

    missing options
    """
    try:
        res = stats.kurtosis(a)
    except ValueError:
        res = np.nan
    return res


def _skew(a):
    """
    wrapper for scipy.stats.skew that returns nan instead of raising Error

    missing options
    """
    try:
        res = stats.skew(a)
    except ValueError:
        res = np.nan
    return res


def sign_test(samp, mu0=0):
    """
    Signs test

    Parameters
    ----------
    samp : array_like
        1d array. The sample for which you want to perform the sign test.
    mu0 : float
        See Notes for the definition of the sign test. mu0 is 0 by
        default, but it is common to set it to the median.

    Returns
    -------
    M
    p-value

    Notes
    -----
    The signs test returns

    M = (N(+) - N(-))/2

    where N(+) is the number of values above `mu0`, N(-) is the number of
    values below.  Values equal to `mu0` are discarded.

    The p-value for M is calculated using the binomial distribution
    and can be interpreted the same as for a t-test. The test-statistic
    is distributed Binom(min(N(+), N(-)), n_trials, .5) where n_trials
    equals N(+) + N(-).

    See Also
    --------
    scipy.stats.wilcoxon
    """
    samp = np.asarray(samp)
    pos = np.sum(samp > mu0)
    neg = np.sum(samp < mu0)
    M = (pos - neg) / 2.0
    p = stats.binom_test(min(pos, neg), pos + neg, 0.5)
    return M, p


NUMERIC_STATISTICS = (
    "nobs",
    "missing",
    "mean",
    "std_err",
    "ci",
    "std",
    "iqr",
    "iqr_normal",
    "mad",
    "mad_normal",
    "coef_var",
    "range",
    "max",
    "min",
    "skew",
    "kurtosis",
    "jarque_bera",
    "mode",
    "median",
    "percentiles",
)
CATEGORICAL_STATISTICS = ("nobs", "missing", "distinct", "top", "freq")
_additional = [
    stat for stat in CATEGORICAL_STATISTICS if stat not in NUMERIC_STATISTICS
]
DEFAULT_STATISTICS = NUMERIC_STATISTICS + tuple(_additional)


class Description:
    """
    Extended descriptive statistics for data

    Parameters
    ----------
    data : array_like
        Data to describe. Must be convertible to a pandas DataFrame.
    stats : Sequence[str], optional
        Statistics to include. If not provided the full set of statistics is
        computed. This list may evolve across versions to reflect best
        practices. Supported options are:
        "nobs", "missing", "mean", "std_err", "ci", "ci", "std", "iqr",
        "iqr_normal", "mad", "mad_normal", "coef_var", "range", "max",
        "min", "skew", "kurtosis", "jarque_bera", "mode", "freq",
        "median", "percentiles", "distinct", "top", and "freq". See Notes for
        details.
    numeric : bool, default True
        Whether to include numeric columns in the descriptive statistics.
    categorical : bool, default True
        Whether to include categorical columns in the descriptive statistics.
    alpha : float, default 0.05
        A number between 0 and 1 representing the size used to compute the
        confidence interval, which has coverage 1 - alpha.
    use_t : bool, default False
        Use the Student's t distribution to construct confidence intervals.
    percentiles : sequence[float]
        A distinct sequence of floating point values all between 0 and 100.
        The default percentiles are 1, 5, 10, 25, 50, 75, 90, 95, 99.
    ntop : int, default 5
        The number of top categorical labels to report. Default is

    Attributes
    ----------
    numeric_statistics
        The list of supported statistics for numeric data
    categorical_statistics
        The list of supported statistics for categorical data
    default_statistics
        The default list of statistics

    See Also
    --------
    pandas.DataFrame.describe
        Basic descriptive statistics
    describe
        A simplified version that returns a DataFrame

    Notes
    -----
    The selectable statistics include:

    * "nobs" - Number of observations
    * "missing" - Number of missing observations
    * "mean" - Mean
    * "std_err" - Standard Error of the mean assuming no correlation
    * "ci" - Confidence interval with coverage (1 - alpha) using the normal or
      t. This option creates two entries in any tables: lower_ci and upper_ci.
    * "std" - Standard Deviation
    * "iqr" - Interquartile range
    * "iqr_normal" - Interquartile range relative to a Normal
    * "mad" - Mean absolute deviation
    * "mad_normal" - Mean absolute deviation relative to a Normal
    * "coef_var" - Coefficient of variation
    * "range" - Range between the maximum and the minimum
    * "max" - The maximum
    * "min" - The minimum
    * "skew" - The skewness defined as the standardized 3rd central moment
    * "kurtosis" - The kurtosis defined as the standardized 4th central moment
    * "jarque_bera" - The Jarque-Bera test statistic for normality based on
      the skewness and kurtosis. This option creates two entries, jarque_bera
      and jarque_beta_pval.
    * "mode" - The mode of the data. This option creates two entries in all tables,
      mode and mode_freq which is the empirical frequency of the modal value.
    * "median" - The median of the data.
    * "percentiles" - The percentiles. Values included depend on the input value of
      ``percentiles``.
    * "distinct" - The number of distinct categories in a categorical.
    * "top" - The mode common categories. Labeled top_n for n in 1, 2, ..., ``ntop``.
    * "freq" - The frequency of the common categories. Labeled freq_n for n in 1,
      2, ..., ``ntop``.
    """

    _int_fmt = ["nobs", "missing", "distinct"]
    numeric_statistics = NUMERIC_STATISTICS
    categorical_statistics = CATEGORICAL_STATISTICS
    default_statistics = DEFAULT_STATISTICS

    def __init__(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        stats: Sequence[str] = None,
        *,
        numeric: bool = True,
        categorical: bool = True,
        alpha: float = 0.05,
        use_t: bool = False,
        percentiles: Sequence[Union[int, float]] = PERCENTILES,
        ntop: bool = 5,
    ):
        data_arr = data
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            data_arr = array_like(data, "data", maxdim=2)
        if data_arr.ndim == 1:
            data = pd.Series(data)
        numeric = bool_like(numeric, "numeric")
        categorical = bool_like(categorical, "categorical")
        include = []
        col_types = ""
        if numeric:
            include.append(np.number)
            col_types = "numeric"
        if categorical:
            include.append("category")
            col_types += "and " if col_types != "" else ""
            col_types += "categorical"
        if not numeric and not categorical:
            raise ValueError(
                "At least one of numeric and categorical must be True"
            )
        self._data = pd.DataFrame(data).select_dtypes(include)
        if self._data.shape[1] == 0:

            raise ValueError(
                "Selecting {col_types} results in an empty DataFrame"
            )
        self._is_numeric = [is_numeric_dtype(dt) for dt in self._data.dtypes]
        self._is_cat_like = [
            is_categorical_dtype(dt) for dt in self._data.dtypes
        ]

        if stats is not None:
            undef = [stat for stat in stats if stat not in DEFAULT_STATISTICS]
            if undef:
                raise ValueError(
                    f"{', '.join(undef)} are not known statistics"
                )
        self._stats = (
            list(DEFAULT_STATISTICS) if stats is None else list(stats)
        )
        self._ntop = int_like(ntop, "ntop")
        self._compute_top = "top" in self._stats
        self._compute_freq = "freq" in self._stats
        if self._compute_top and self._ntop <= 0 < sum(self._is_cat_like):
            raise ValueError("top must be a non-negative integer")

        # Expand special stats
        replacements = {
            "mode": ["mode", "mode_freq"],
            "ci": ["upper_ci", "lower_ci"],
            "jarque_bera": ["jarque_bera", "jarque_bera_pval"],
            "top": [f"top_{i}" for i in range(1, self._ntop + 1)],
            "freq": [f"freq_{i}" for i in range(1, self._ntop + 1)],
        }

        for key in replacements:
            if key in self._stats:
                idx = self._stats.index(key)
                self._stats = (
                    self._stats[:idx]
                    + replacements[key]
                    + self._stats[idx + 1 :]
                )

        self._percentiles = array_like(
            percentiles, "percentiles", maxdim=1, dtype="d"
        )
        self._percentiles = np.sort(self._percentiles)
        if np.unique(self._percentiles).shape[0] != self._percentiles.shape[0]:
            raise ValueError("percentiles must be distinct")
        if np.any(self._percentiles >= 100) or np.any(self._percentiles <= 0):
            raise ValueError("percentiles must be strictly between 0 and 100")
        self._alpha = float_like(alpha, "alpha")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1")
        self._use_t = bool_like(use_t, "use_t")

    def _reorder(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[[s for s in self._stats if s in df.index]]

    @cache_readonly
    def frame(self) -> pd.DataFrame:
        """
        Descriptive statistics for both numeric and categorical data

        Returns
        -------
        DataFrame
            The statistics
        """
        numeric = self.numeric
        categorical = self.categorical
        if categorical.shape[1] == 0:
            return numeric
        elif numeric.shape[1] == 0:
            return categorical
        df = pd.concat([numeric, categorical], axis=1)
        return self._reorder(df[self._data.columns])

    @cache_readonly
    def numeric(self) -> pd.DataFrame:
        """
        Descriptive statistics for numeric data

        Returns
        -------
        DataFrame
            The statistics of the numeric columns
        """
        df: pd.DataFrame = self._data.loc[:, self._is_numeric]
        cols = df.columns
        _, k = df.shape
        std = df.std()
        count = df.count()
        mean = df.mean()
        mad = (df - mean).abs().mean()
        std_err = std.copy()
        std_err.loc[count > 0] /= count.loc[count > 0]
        if self._use_t:
            q = stats.t(count - 1).ppf(1.0 - self._alpha / 2)
        else:
            q = stats.norm.ppf(1.0 - self._alpha / 2)

        def _mode(ser):
            mode_res = stats.mode(ser.dropna())
            if mode_res[0].shape[0] > 0:
                return [float(val) for val in mode_res]
            return np.nan, np.nan

        mode_values = df.apply(_mode).T
        if mode_values.size > 0:
            if isinstance(mode_values, pd.DataFrame):
                # pandas 1.0 or later
                mode = np.asarray(mode_values[0], dtype=float)
                mode_counts = np.asarray(mode_values[1], dtype=np.int64)
            else:
                # pandas before 1.0 returns a Series of 2-elem list
                mode = []
                mode_counts = []
                for idx in mode_values.index:
                    val = mode_values.loc[idx]
                    mode.append(val[0])
                    mode_counts.append(val[1])
                mode = np.atleast_1d(mode)
                mode_counts = np.atleast_1d(mode_counts)
        else:
            mode = mode_counts = np.empty(0)
        loc = count > 0
        mode_freq = np.full(mode.shape[0], np.nan)
        mode_freq[loc] = mode_counts[loc] / count.loc[loc]
        if df.shape[1] > 0:
            iqr = df.quantile(0.75) - df.quantile(0.25)
        else:
            iqr = mean

        def _safe_jarque_bera(c):
            a = np.asarray(c)
            if a.shape[0] < 2:
                return (np.nan,) * 4
            return jarque_bera(a)

        jb = df.apply(
            lambda x: list(_safe_jarque_bera(x.dropna())), result_type="expand"
        ).T
        nan_mean = mean.copy()
        nan_mean.loc[nan_mean == 0] = np.nan
        coef_var = std / nan_mean

        results = {
            "nobs": pd.Series(
                np.ones(k, dtype=np.int64) * df.shape[0], index=cols
            ),
            "missing": df.shape[0] - count,
            "mean": mean,
            "std_err": std_err,
            "upper_ci": mean + q * std_err,
            "lower_ci": mean - q * std_err,
            "std": std,
            "iqr": iqr,
            "mad": mad,
            "coef_var": coef_var,
            "range": pd_ptp(df),
            "max": df.max(),
            "min": df.min(),
            "skew": jb[2],
            "kurtosis": jb[3],
            "iqr_normal": iqr / np.diff(stats.norm.ppf([0.25, 0.75])),
            "mad_normal": mad / np.sqrt(2 / np.pi),
            "jarque_bera": jb[0],
            "jarque_bera_pval": jb[1],
            "mode": pd.Series(mode, index=cols),
            "mode_freq": pd.Series(mode_freq, index=cols),
            "median": df.median(),
        }
        final = {k: v for k, v in results.items() if k in self._stats}
        results_df = pd.DataFrame(
            list(final.values()), columns=cols, index=list(final.keys())
        )
        if "percentiles" not in self._stats:
            return results_df
        # Pandas before 1.0 cannot handle empty DF
        if df.shape[1] > 0:
            perc = df.quantile(self._percentiles / 100).astype(float)
        else:
            perc = pd.DataFrame(index=self._percentiles / 100, dtype=float)
        if np.all(np.floor(100 * perc.index) == (100 * perc.index)):
            perc.index = [f"{int(100 * idx)}%" for idx in perc.index]
        else:
            dupe = True
            scale = 100
            index = perc.index
            while dupe:
                scale *= 10
                idx = np.floor(scale * perc.index)
                if np.all(np.diff(idx) > 0):
                    dupe = False
            index = np.floor(scale * index) / (scale / 100)
            fmt = f"0.{len(str(scale//100))-1}f"
            output = f"{{0:{fmt}}}%"
            perc.index = [output.format(val) for val in index]

        return self._reorder(pd.concat([results_df, perc], 0))

    @cache_readonly
    def categorical(self) -> pd.DataFrame:
        """
        Descriptive statistics for categorical data

        Returns
        -------
        DataFrame
            The statistics of the categorical columns
        """

        df = self._data.loc[:, [col for col in self._is_cat_like]]
        k = df.shape[1]
        cols = df.columns
        vc = {col: df[col].value_counts(normalize=True) for col in df}
        distinct = pd.Series(
            {col: vc[col].shape[0] for col in vc}, dtype=np.int64
        )
        top = {}
        freq = {}
        for col in vc:
            single = vc[col]
            if single.shape[0] >= self._ntop:
                top[col] = single.index[: self._ntop]
                freq[col] = np.asarray(single.iloc[:5])
            else:
                val = list(single.index)
                val += [None] * (self._ntop - len(val))
                top[col] = val
                freq_val = list(single)
                freq_val += [np.nan] * (self._ntop - len(freq_val))
                freq[col] = np.asarray(freq_val)
        index = [f"top_{i}" for i in range(1, self._ntop + 1)]
        top_df = pd.DataFrame(top, dtype="object", index=index, columns=cols)
        index = [f"freq_{i}" for i in range(1, self._ntop + 1)]
        freq_df = pd.DataFrame(freq, dtype="object", index=index, columns=cols)

        results = {
            "nobs": pd.Series(
                np.ones(k, dtype=np.int64) * df.shape[0], index=cols
            ),
            "missing": df.shape[0] - df.count(),
            "distinct": distinct,
        }
        final = {k: v for k, v in results.items() if k in self._stats}
        results_df = pd.DataFrame(
            list(final.values()),
            columns=cols,
            index=list(final.keys()),
            dtype="object",
        )
        if self._compute_top:
            results_df = pd.concat([results_df, top_df], axis=0)
        if self._compute_freq:
            results_df = pd.concat([results_df, freq_df], axis=0)

        return self._reorder(results_df)

    def summary(self) -> SimpleTable:
        """
        Summary table of the descriptive statistics

        Returns
        -------
        SimpleTable
            A table instance supporting export to text, csv and LaTeX
        """
        df = self.frame.astype(object)
        df = df.fillna("")
        cols = [str(col) for col in df.columns]
        stubs = [str(idx) for idx in df.index]
        data = []
        for _, row in df.iterrows():
            data.append([v for v in row])

        def _formatter(v):
            if isinstance(v, str):
                return v
            elif v // 1 == v:
                return str(int(v))
            return f"{v:0.4g}"

        return SimpleTable(
            data,
            header=cols,
            stubs=stubs,
            title="Descriptive Statistics",
            txt_fmt={"data_fmts": {0: "%s", 1: _formatter}},
            datatypes=[1] * len(data),
        )

    def __str__(self) -> str:
        return str(self.summary().as_text())


ds = Docstring(Description.__doc__)
ds.replace_block(
    "Returns", Parameter(None, "DataFrame", ["Descriptive statistics"])
)
ds.replace_block("Attributes", [])
ds.replace_block(
    "See Also",
    [
        (
            [("pandas.DataFrame.describe", None)],
            ["Basic descriptive statistics"],
        ),
        (
            [("Description", None)],
            ["Descriptive statistics class with additional output options"],
        ),
    ],
)


@Appender(str(ds))
def describe(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    stats: Sequence[str] = None,
    *,
    numeric: bool = True,
    categorical: bool = True,
    alpha: float = 0.05,
    use_t: bool = False,
    percentiles: Sequence[Union[int, float]] = PERCENTILES,
    ntop: bool = 5,
) -> pd.DataFrame:
    return Description(
        data,
        stats,
        numeric=numeric,
        categorical=categorical,
        alpha=alpha,
        use_t=use_t,
        percentiles=percentiles,
        ntop=ntop,
    ).frame


class Describe(object):
    """
    Calculates descriptive statistics for data.

    .. deprecated:: 0.12

        Use ``Description`` or ``describe`` instead

    Defaults to a basic set of statistics, "all" can be specified, or a list
    can be given.

    Parameters
    ----------
    dataset : array_like
        2D dataset for descriptive statistics.
    """

    def __init__(self, dataset):
        warnings.warn(DEPRECATION_MSG, DeprecationWarning)
        self.dataset = dataset

        # better if this is initially a list to define order, or use an
        # ordered dict. First position is the function
        # Second position is the tuple/list of column names/numbers
        # third is are the results in order of the columns
        self.univariate = dict(
            obs=[len, None, None],
            mean=[np.mean, None, None],
            std=[np.std, None, None],
            min=[np.min, None, None],
            max=[np.max, None, None],
            ptp=[np.ptp, None, None],
            var=[np.var, None, None],
            mode_val=[self._mode_val, None, None],
            mode_bin=[self._mode_bin, None, None],
            median=[np.median, None, None],
            skew=[stats.skew, None, None],
            uss=[lambda x: np.sum(np.asarray(x) ** 2, axis=0), None, None],
            kurtosis=[stats.kurtosis, None, None],
            percentiles=[self._percentiles, None, None],
            # BUG: not single value
            # sign_test_M = [self.sign_test_m, None, None],
            # sign_test_P = [self.sign_test_p, None, None]
        )

        # TODO: Basic stats for strings
        # self.strings = dict(
        #    unique = [np.unique, None, None],
        #    number_uniq = [len(
        #    most = [
        #    least = [

        # TODO: Multivariate
        # self.multivariate = dict(
        #    corrcoef(x[, y, rowvar, bias]),
        #    cov(m[, y, rowvar, bias]),
        #    histogram2d(x, y[, bins, range, normed, weights])
        #    )
        self._arraytype = None
        self._columns_list = None

    def _percentiles(self, x):
        p = [
            stats.scoreatpercentile(x, per)
            for per in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        ]
        return p

    def _mode_val(self, x):
        return stats.mode(x)[0][0]

    def _mode_bin(self, x):
        return stats.mode(x)[1][0]

    def _array_typer(self):
        """if not a sctructured array"""
        if not (self.dataset.dtype.names):
            """homogeneous dtype array"""
            self._arraytype = "homog"
        elif self.dataset.dtype.names:
            """structured or rec array"""
            self._arraytype = "sctruct"
        else:
            assert self._arraytype == "sctruct" or self._arraytype == "homog"

    def _is_dtype_like(self, col):
        """
        Check whether self.dataset.[col][0] behaves like a string, numbern
        unknown. `numpy.lib._iotools._is_string_like`
        """

        def string_like():
            # TODO: not sure what the result is if the first item is some
            #   type of missing value
            try:
                self.dataset[col][0] + ""
            except (TypeError, ValueError):
                return False
            return True

        def number_like():
            try:
                self.dataset[col][0] + 1.0
            except (TypeError, ValueError):
                return False
            return True

        if number_like() and not string_like():
            return "number"
        elif not number_like() and string_like():
            return "string"
        else:
            assert number_like() or string_like(), (
                "\
            Not sure of dtype"
                + str(self.dataset[col][0])
            )

    # @property
    def summary(self, stats="basic", columns="all", orientation="auto"):
        """
        Return a summary of descriptive statistics.

        Parameters
        ----------
        stats: list or str
            The desired statistics, Accepts 'basic' or 'all' or a list.
               'basic' = ('obs', 'mean', 'std', 'min', 'max')
               'all' = ('obs', 'mean', 'std', 'min', 'max', 'ptp', 'var',
                        'mode', 'meadian', 'skew', 'uss', 'kurtosis',
                        'percentiles')
        columns : list or str
          The columns/variables to report the statistics, default is 'all'
          If an object with named columns is given, you may specify the
          column names. For example
        """
        # NOTE
        # standard array: Specifiy column numbers (NEED TO TEST)
        # percentiles currently broken
        # mode requires mode_val and mode_bin separately
        if self._arraytype is None:
            self._array_typer()

        if stats == "basic":
            stats = ("obs", "mean", "std", "min", "max")
        elif stats == "all":
            # stats = self.univariate.keys()
            # dict does not keep an order, use full list instead
            stats = [
                "obs",
                "mean",
                "std",
                "min",
                "max",
                "ptp",
                "var",
                "mode_val",
                "mode_bin",
                "median",
                "uss",
                "skew",
                "kurtosis",
                "percentiles",
            ]
        else:
            for astat in stats:
                pass
                # assert astat in self.univariate

        # hack around percentiles multiple output

        # bad naming
        import scipy.stats

        # BUG: the following has all per the same per=99
        ##perdict = dict(('perc_%2d'%per, [lambda x:
        #       scipy.stats.scoreatpercentile(x, per), None, None])
        ##          for per in (1,5,10,25,50,75,90,95,99))

        def _fun(per):
            return lambda x: scipy.stats.scoreatpercentile(x, per)

        perdict = dict(
            ("perc_%02d" % per, [_fun(per), None, None])
            for per in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        )

        if "percentiles" in stats:
            self.univariate.update(perdict)
            idx = stats.index("percentiles")
            stats[idx : idx + 1] = sorted(perdict.keys())

        # JP: this does not allow a change in sequence, sequence in stats is
        # ignored
        # this is just an if condition
        if any(
            [
                aitem[1]
                for aitem in self.univariate.items()
                if aitem[0] in stats
            ]
        ):
            if columns == "all":
                self._columns_list = []
                if self._arraytype == "sctruct":
                    self._columns_list = self.dataset.dtype.names
                    # self._columns_list = [col for col in
                    #                      self.dataset.dtype.names if
                    #        (self._is_dtype_like(col)=='number')]
                else:
                    self._columns_list = lrange(self.dataset.shape[1])
            else:
                self._columns_list = columns
                if self._arraytype == "sctruct":
                    for col in self._columns_list:
                        assert col in self.dataset.dtype.names
                else:
                    assert self._is_dtype_like(self.dataset) == "number"

            columstypes = self.dataset.dtype
            # TODO: do we need to make sure they dtype is float64 ?
            for astat in stats:
                calc = self.univariate[astat]
                if self._arraytype == "sctruct":
                    calc[1] = self._columns_list
                    calc[2] = [
                        calc[0](self.dataset[col])
                        for col in self._columns_list
                        if (self._is_dtype_like(col) == "number")
                    ]
                    # calc[2].append([len(np.unique(self.dataset[col])) for col
                    #                in self._columns_list if
                    #                self._is_dtype_like(col)=='string']
                else:
                    calc[1] = ["Col " + str(col) for col in self._columns_list]
                    calc[2] = [
                        calc[0](self.dataset[:, col])
                        for col in self._columns_list
                    ]
            return self.print_summary(stats, orientation=orientation)
        else:
            return self.print_summary(stats, orientation=orientation)

    def print_summary(self, stats, orientation="auto"):
        # TODO: need to specify a table formating for the numbers, using defualt
        title = "Summary Statistics"
        header = stats
        stubs = self.univariate["obs"][1]
        data = [
            [self.univariate[astat][2][col] for astat in stats]
            for col in range(len(self.univariate["obs"][2]))
        ]

        if (orientation == "varcols") or (
            orientation == "auto" and len(stubs) < len(header)
        ):
            # swap rows and columns
            data = lmap(lambda *row: list(row), *data)
            header, stubs = stubs, header

        part_fmt = dict(data_fmts=["%#8.4g"] * (len(header) - 1))
        table = SimpleTable(data, header, stubs, title=title, txt_fmt=part_fmt)

        return table

    @Appender(sign_test.__doc__)  # i.e. module-level sign_test
    def sign_test(self, samp, mu0=0):
        return sign_test(samp, mu0)

    # TODO: There must be a better way but formating the stats of a fuction that
    #      returns 2 values is a problem.
    # def sign_test_m(samp,mu0=0):
    # return self.sign_test(samp,mu0)[0]
    # def sign_test_p(samp,mu0=0):
    # return self.sign_test(samp,mu0)[1]
