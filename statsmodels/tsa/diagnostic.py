import numpy as np
from scipy import stats

from statsmodels.stats.tools import TestResult
from statsmodels.tsa.stattools import acf
from statsmodels.tools.decorators import nottest


@nottest
def acorr_box_test(x, nlags, df=0, boxpierce=False):
    r"""
    Ljung-Box or Box-Pierce test for no autocorrelation

    Parameters
    ----------
    x : array-like
        Data to test for autocorrelation. Usually regression residuals.
    lags : int
        The returned statistic will be based on this many lags.
    df : int
        Degrees of freedom correction. If `x' is based on model residuals,
        should be the number of parameters fit for the model. For example,
        if residuals come from an ARMA model, df should be the number of AR
        coefficients plus the number of MA coefficients.
    boxpierce : bool
        If True, returns the Box-Pierce test statistic. If False, returns
        the Ljung-Box. See notes. Default is False.

    Returns
    -------
    results : TestResults object
        An object containing relevant test statistics

    Notes
    -----
    The Ljung-Box test statistic is

    .. math::

       Q = nobs * (nobs + 2) * \sum_{k=1}^(nlags)\frac{\rho_k^2}{nobs - k}

    where :math:`\rho_k` is the sample autocorrelation at lag :math:`k`.
    :math:`Q` is assumed to be distributed :math:`\chi^2_` with
    :math:`nlags-df` degrees of freedom.

    The Box-Pierce test statistic is

    .. math::

       Q = n * \sum_{k=1}^{nlags}\rho^2_k

    The Ljung-Box statistic is preferred for all sample sizes.
    """
    x = np.asarray(x)
    nobs = float(len(x))
    lags = np.arange(1, nlags + 1)

    acf_p = acf(x, nlags=nlags, fft=True)[1:]  # drop lag 0
    if not boxpierce:
        q_stat = nobs * (nobs + 2) * np.sum(acf_p**2 / (nobs - lags))
    else:
        q_stat = nobs * np.sum(acf_p ** 2)
    p_value = stats.chi2.sf(q_stat, nlags - df)

    return_doc = """
    Ljung-Box or Box-Pierce Test Results

    Attributes
    ----------
    q_stat : float
        Ljung-Box test statistic
    p_value : float
        p-value of the test statistic
    """

    return TestResult(return_doc, "The series is random.",
                      "The series is not independently distributed.",
                      order=["q_stat", "p_value"], q_stat=q_stat,
                      p_value=p_value)

if __name__ == "__main__":
    import statsmodels.api as sm
    from statsmodels.tsa.arima_process import arma_generate_sample
    np.random.seed(12345)
    y = arma_generate_sample([1, .75, .35], [1, .25], nsample=100)

    # order identification
    order_select = sm.tsa.arma_order_select_ic(y)
    print order_select.bic_min_order

    # model validation
    mod = sm.tsa.ARMA(y, (2, 1)).fit(trend='nc')

    mod.arroots
    mod.maroots # really close to the unit circle

    resid = mod.resid

    box_test = acorr_box_test(resid, 10, 3)
    print box_test.summary()
