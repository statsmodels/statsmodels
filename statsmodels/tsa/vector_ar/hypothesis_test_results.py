from typing import NamedTuple

import numpy as np

from statsmodels.iolib.table import SimpleTable


class ForecastInterval(NamedTuple):
    """
    Result of the module-level
    :func:`~statsmodels.tsa.vector_ar.var_model.forecast_interval` and
    :meth:`~statsmodels.tsa.vector_ar.var_model.VARProcess.forecast_interval`.

    Parameters
    ----------
    point_forecast : ndarray
        Mean value of forecast.
    forc_lower : ndarray
        Lower bound of confidence interval.
    forc_upper : ndarray
        Upper bound of confidence interval.
    """

    point_forecast: np.ndarray
    forc_lower: np.ndarray
    forc_upper: np.ndarray


class ErrorBand(NamedTuple):
    """
    Impulse-response error band, shared by
    :meth:`~statsmodels.tsa.vector_ar.irf.IRAnalysis.err_band_sz1`,
    :meth:`~statsmodels.tsa.vector_ar.irf.IRAnalysis.err_band_sz2`,
    :meth:`~statsmodels.tsa.vector_ar.irf.IRAnalysis.err_band_sz3`,
    :meth:`~statsmodels.tsa.vector_ar.irf.IRAnalysis.errband_mc`,
    :meth:`~statsmodels.tsa.vector_ar.svar_model.SVARResults.sirf_errband_mc`,
    and :meth:`~statsmodels.tsa.vector_ar.var_model.VARResults.irf_errband_mc`.

    Parameters
    ----------
    lower : ndarray
        Lower error band for the impulse responses.
    upper : ndarray
        Upper error band for the impulse responses.
    """

    lower: np.ndarray
    upper: np.ndarray


class HypothesisTestResults:
    """
    Results class for hypothesis tests

    Parameters
    ----------
    test_statistic : float
        The test's test statistic.
    crit_value : float
        The test's critical value.
    pvalue : float
        The test's p-value. Must be between 0 and 1.
    df : int
        Degrees of freedom.
    signif : float
        Significance level. Must be between 0 and 1.
    method : str
        The kind of test (e.g., ``"f"`` for F-test, ``"wald"`` for Wald-test).
    title : str
        A title describing the test. It will be part of the summary.
    h0 : str
        A string describing the null hypothesis. It will be used in the
        summary.
    """
    def __init__(self, test_statistic, crit_value, pvalue, df,
                 signif, method, title, h0):
        self.test_statistic = test_statistic
        self.crit_value = crit_value
        self.pvalue = pvalue
        self.df = df
        self.signif = signif
        self.method = method.capitalize()
        if test_statistic < crit_value:
            self.conclusion = "fail to reject"
        else:
            self.conclusion = "reject"
        self.title = title
        self.h0 = h0
        self.conclusion_str = f"Conclusion: {self.conclusion} H_0"
        self.signif_str = f" at {self.signif:.0%} significance level"

    def summary(self):
        """Return summary"""
        title = self.title + ". " + self.h0 + ". " \
                                  + self.conclusion_str + self.signif_str + "."
        data_fmt = {"data_fmts": ["%#0.4g", "%#0.4g", "%#0.3F", "%s"]}
        html_data_fmt = dict(data_fmt)
        html_data_fmt["data_fmts"] = ["<td>" + i + "</td>"
                                      for i in html_data_fmt["data_fmts"]]
        return SimpleTable(data=[[self.test_statistic, self.crit_value,
                                  self.pvalue, str(self.df)]],
                           headers=["Test statistic", "Critical value",
                                    "p-value", "df"],
                           title=title,
                           txt_fmt=data_fmt,
                           html_fmt=html_data_fmt,
                           ltx_fmt=data_fmt)

    def __str__(self):
        return "<" + self.__module__ + "." + self.__class__.__name__ \
                   + " object. " + self.h0 + ": " + self.conclusion \
                   + self.signif_str \
                   + f". Test statistic: {self.test_statistic:.3f}" \
                   + f", critical value: {self.crit_value:.3f}>" \
                   + f", p-value: {self.pvalue:.3f}>"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return np.allclose(self.test_statistic, other.test_statistic) \
            and np.allclose(self.crit_value, other.crit_value) \
            and np.allclose(self.pvalue, other.pvalue) \
            and np.allclose(self.signif, other.signif)

    # Equality is based on np.allclose, which is not compatible with a
    # hash based on field values, so instances remain unhashable (this
    # matches the implicit behavior of defining __eq__ without __hash__).
    __hash__ = None


class CausalityTestResults(HypothesisTestResults):
    """
    Results class for Granger-causality and instantaneous causality

    Parameters
    ----------
    causing : list of str
        This list contains the potentially causing variables.
    caused : list of str
        This list contains the potentially caused variables.
    test_statistic : float
        The test's test statistic.
    crit_value : float
        The test's critical value.
    pvalue : float
        The test's p-value. Must be between 0 and 1.
    df : int
        Degrees of freedom.
    signif : float
        Significance level.
    test : {"granger", "inst"}, optional
        If "granger", Granger-causality has been tested. If "inst",
        instantaneous causality has been tested.
    method : {"f", "wald"}, optional
        The kind of test. "f" indicates an F-test, "wald" indicates a
        Wald-test. Must be specified explicitly; a ValueError is raised
        if left as None.
    """
    def __init__(self, causing, caused, test_statistic, crit_value, pvalue, df,
                 signif, test="granger", method=None):
        self.causing = causing
        self.caused = caused
        self.test = test
        if method is None or method.lower() not in ["f", "wald"]:
            raise ValueError('The method ("f" for F-test, "wald" for '
                             "Wald-test) must not be None.")
        method = method.capitalize()
        # attributes used in summary and string representation:
        title = "Granger" if self.test == "granger" else "Instantaneous"
        title += f" causality {method}-test"
        h0 = "H_0: "
        if len(self.causing) == 1:
            h0 += f"{self.causing[0]} does not "
        else:
            h0 += f"{self.causing} do not "
        h0 += "Granger-" if self.test == "granger" else "instantaneously "
        h0 += "cause "
        if len(self.caused) == 1:
            h0 += self.caused[0]
        else:
            h0 += "[" + ", ".join(caused) + "]"

        super().__init__(test_statistic, crit_value,
                         pvalue, df, signif, method,
                         title, h0)

    def __eq__(self, other):
        basic_test = super().__eq__(other)
        if not basic_test:
            return False
        test = self.test == other.test
        variables = (self.causing == other.causing and
                     self.caused == other.caused)
        # instantaneous causality is a symmetric relation ==> causing and
        # caused may be swapped
        if not variables and self.test == "inst":
            variables = (self.causing == other.caused and
                         self.caused == other.causing)
        return test and variables

    __hash__ = None


class NormalityTestResults(HypothesisTestResults):
    """
    Results class for the Jarque-Bera-test for nonnormality

    Parameters
    ----------
    test_statistic : float
        The test's test statistic.
    crit_value : float
        The test's critical value.
    pvalue : float
        The test's p-value.
    df : int
        Degrees of freedom.
    signif : float
        Significance level.
    """
    def __init__(self, test_statistic, crit_value, pvalue, df, signif):
        method = "Jarque-Bera"
        title = "normality (skew and kurtosis) test"
        h0 = "H_0: data generated by normally-distributed process"
        super().__init__(test_statistic, crit_value,
                         pvalue, df, signif,
                         method, title, h0)


class WhitenessTestResults(HypothesisTestResults):
    """
    Results class for the Portmanteau-test for residual autocorrelation

    Parameters
    ----------
    test_statistic : float
        The test's test statistic.
    crit_value : float
        The test's critical value.
    pvalue : float
        The test's p-value.
    df : int
        Degrees of freedom.
    signif : float
        Significance level.
    nlags : int
        Number of lags tested.
    adjusted : bool
        Whether the test statistic is adjusted for the number of
        observations used to estimate the autocorrelations.
    """
    def __init__(self, test_statistic, crit_value, pvalue, df, signif, nlags,
                 adjusted):
        self.lags = nlags
        self.adjusted = adjusted
        method = "Portmanteau"
        title = f"{method}-test for residual autocorrelation"
        if adjusted:
            title = "Adjusted " + title
        h0 = f"H_0: residual autocorrelation up to lag {nlags} is zero"
        super().__init__(
            test_statistic,
            crit_value,
            pvalue,
            df,
            signif,
            method,
            title,
            h0
        )
