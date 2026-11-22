import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.tools.validation import string_like


class PredictionResults:
    """
    Prediction results

    Parameters
    ----------
    predicted_mean : ndarray, Series, or DataFrame
        The predicted mean values.
    var_pred_mean : ndarray or Series
        The variance of the predicted mean values. Either 1-d, or 3-d with
        shape (nobs, k, k) holding the covariance matrix of each prediction.
    dist : {"norm", "t"} or rv_frozen, optional
        The distribution to use when constructing prediction intervals.
        Default is the normal distribution.
    df : float, optional
        The degrees of freedom parameter for the t distribution. Required
        if dist is "t", and must be None otherwise.
    row_labels : sequence of hashable or pandas.Index, optional
        Row labels to use for the summary frame. If None, attempts to read
        the index of ``predicted_mean``.

    Notes
    -----
    A predicted mean with a single column, such as a one-column DataFrame,
    is treated as 1-d, and so is a variance with shape (nobs, 1, 1).

    .. versionchanged:: 0.15.1

       A single column was kept as 2-d, which gave a confidence interval
       with 2 * nobs columns for an (nobs, 1) array and raised for a
       one-column DataFrame.

    .. versionchanged:: 0.15.1

       ``predicted_mean`` and ``se_mean`` of a DataFrame predicted mean with
       several columns keep its column names. They had the columns 0, ...,
       k - 1.
    """

    def __init__(
        self,
        predicted_mean,
        var_pred_mean,
        dist=None,
        df=None,
        row_labels=None,
    ):
        self._predicted_mean = np.asarray(predicted_mean)
        self._var_pred_mean = np.asarray(var_pred_mean)
        # Names of the columns of a 2-d predicted mean
        self._columns = getattr(predicted_mean, "columns", None)
        # A single column and its (nobs, 1, 1) variance are 1-d
        mean_shape = self._predicted_mean.shape
        if len(mean_shape) == 2 and mean_shape[1] == 1:
            self._predicted_mean = self._predicted_mean[:, 0]
        if self._var_pred_mean.shape[1:] == (1, 1):
            self._var_pred_mean = self._var_pred_mean[:, 0, 0]
        self._df = df
        self._row_labels = row_labels
        if row_labels is None:
            self._row_labels = getattr(predicted_mean, "index", None)
        self._use_pandas = self._row_labels is not None

        if dist != "t" and df is not None:
            raise ValueError('df must be None when dist is not "t"')
        if dist == "t" and df is None:
            raise ValueError('df must be given when dist is "t"')

        if dist is None or dist == "norm":
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == "t":
            self.dist = stats.t
            self.dist_args = (self._df,)
        elif isinstance(dist, stats.distributions.rv_frozen):
            self.dist = dist
            self.dist_args = ()
        else:
            raise ValueError('dist must be a None, "norm", "t" or a callable.')

    def _column_names(self, prefix):
        # Labels of statistics of the columns of a 2-d predicted mean
        names = self._columns
        if names is None:
            names = range(self._predicted_mean.shape[1])
        return [f"{prefix} {name}" for name in names]

    def _wrap_pandas(self, value, name=None, columns=None):
        if not self._use_pandas:
            return value
        if value.ndim == 1:
            return pd.Series(value, index=self._row_labels, name=name)
        if columns is None and self._columns is not None:
            if len(self._columns) == value.shape[1]:
                # The columns of a DataFrame predicted mean, so that the
                # mean, its standard error and the t-values align
                columns = self._columns
        return pd.DataFrame(value, index=self._row_labels, columns=columns)

    @property
    def row_labels(self):
        """The row labels used in pandas-types"""
        return self._row_labels

    @property
    def predicted_mean(self):
        """The predicted mean"""
        return self._wrap_pandas(self._predicted_mean, "predicted_mean")

    @property
    def var_pred_mean(self):
        """The variance of the predicted mean"""
        if self._var_pred_mean.ndim > 2:
            return self._var_pred_mean
        return self._wrap_pandas(self._var_pred_mean, "var_pred_mean")

    @property
    def se_mean(self):
        """The standard deviation of the predicted mean"""
        ndim = self._var_pred_mean.ndim
        if ndim == 1:
            values = np.sqrt(self._var_pred_mean)
        elif ndim == 3:
            values = np.sqrt(self._var_pred_mean.T.diagonal())
        else:
            raise NotImplementedError("var_pre_mean must be 1 or 3 dim")
        return self._wrap_pandas(values, "mean_se")

    @property
    def tvalues(self):
        """The ratio of the predicted mean to its standard deviation"""
        val = self.predicted_mean / self.se_mean
        if isinstance(val, pd.Series):
            val.name = "tvalues"
        return val

    def t_test(self, value=0, alternative="two-sided"):
        """
        z- or t-test for hypothesis that mean is equal to value

        Parameters
        ----------
        value : array_like, optional
            Value under the null hypothesis.
        alternative : {"two-sided", "larger", "smaller"}, optional
            The alternative hypothesis for the test.

        Returns
        -------
        stat : ndarray or Series
            The test statistic.
        pvalue : ndarray or Series
            The p-value of the hypothesis test. The distribution used is
            given by the attribute of the instance specified in `__init__`.
            The default, if not specified, is the normal distribution.
        """
        alternative = string_like(
            alternative,
            "alternative",
            options=("two-sided", "larger", "smaller"),
            lower=False,
            deprecated={
                "2-sided": "two-sided",
                "2s": "two-sided",
                "l": "larger",
                "s": "smaller",
            },
            removed_after="0.16",
        )
        # assumes symmetric distribution
        stat = (self.predicted_mean - value) / self.se_mean

        if alternative == "two-sided":
            pvalue = self.dist.sf(np.abs(stat), *self.dist_args) * 2
        elif alternative == "larger":
            pvalue = self.dist.sf(stat, *self.dist_args)
        else:  # smaller
            pvalue = self.dist.cdf(stat, *self.dist_args)
        return stat, pvalue

    def conf_int(self, alpha=0.05):
        """
        Confidence interval construction for the predicted mean

        This is currently only available for t and z tests.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the prediction interval.
            The default `alpha` = .05 returns a 95% confidence interval.

        Returns
        -------
        pi : ndarray or DataFrame
            The array has the lower and the upper limit of the prediction
            interval in the columns. For a predicted mean with several
            columns, the lower limits of all columns come first, labelled
            "lower <name>" in a DataFrame, and then the upper limits,
            labelled "upper <name>", where <name> is the column name of the
            predicted mean, or its position if it has none.

            .. versionchanged:: 0.15.1

               The interval of a predicted mean with several columns is
               labelled in a DataFrame instead of raising.
        """
        se = self.se_mean
        q = self.dist.ppf(1 - alpha / 2.0, *self.dist_args)
        lower = self.predicted_mean - q * se
        upper = self.predicted_mean + q * se
        ci = np.column_stack((lower, upper))
        if self._use_pandas:
            if self._predicted_mean.ndim == 1:
                columns = ["lower", "upper"]
            else:
                columns = self._column_names("lower") + self._column_names("upper")
            return self._wrap_pandas(ci, columns=columns)
        return ci

    def summary_frame(self, alpha=0.05):
        """
        Summary frame of mean, variance and confidence interval

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval. The default
            `alpha` = .05 returns a 95% confidence interval.

        Returns
        -------
        DataFrame
            DataFrame containing four columns:

            * mean
            * mean_se
            * mean_ci_lower
            * mean_ci_upper

            For a predicted mean with several columns, it has these four
            columns for each column of the predicted mean, labelled for
            example "mean <name>", where <name> is the column name of the
            predicted mean, or its position if it has none. All "mean"
            columns come first, then all "mean_se" columns and so on.

            .. versionchanged:: 0.15.1

               A predicted mean with several columns is supported instead
               of raising.
        """
        ci_mean = np.asarray(self.conf_int(alpha=alpha))
        if self._predicted_mean.ndim == 2:
            values = np.column_stack(
                (self._predicted_mean, np.asarray(self.se_mean), ci_mean)
            )
            columns = (
                self._column_names("mean")
                + self._column_names("mean_se")
                + self._column_names("mean_ci_lower")
                + self._column_names("mean_ci_upper")
            )
            return pd.DataFrame(values, index=self._row_labels, columns=columns)
        lower, upper = ci_mean[:, 0], ci_mean[:, 1]
        to_include = {
            "mean": self.predicted_mean,
            "mean_se": self.se_mean,
            "mean_ci_lower": lower,
            "mean_ci_upper": upper,
        }
        return pd.DataFrame(to_include)
