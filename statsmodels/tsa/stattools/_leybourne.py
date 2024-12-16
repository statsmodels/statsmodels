import numpy as np

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import statsmodels.tsa._leybourne
from statsmodels.tsa.stattools._stattools import lagmat, pacf


class LeybourneMcCabeStationarity:
    """
    Class wrapper for Leybourne-McCabe stationarity test
    """

    def __init__(self):
        """
        Asymptotic critical values for the two different models specified
        for the Leybourne-McCabe stationarity test. Asymptotic CVs are the
        same as the asymptotic CVs for the KPSS stationarity test.

        Notes
        -----
        The p-values are generated through Monte Carlo simulation using
        1,000,000 replications and 10,000 data points.
        """
        self.__leybourne_critical_values = {
            # constant-only model
            "c": statsmodels.tsa._leybourne.c,
            # constant-trend model
            "ct": statsmodels.tsa._leybourne.ct,
        }

    def __leybourne_crit(self, stat, model="c"):
        """
        Linear interpolation for Leybourne p-values and critical values

        Parameters
        ----------
        stat : float
            The Leybourne-McCabe test statistic
        model : {'c','ct'}
            The model used when computing the test statistic. 'c' is default.

        Returns
        -------
        pvalue : float
            The interpolated p-value
        cvdict : dict
            Critical values for the test statistic at the 1%, 5%, and 10%
            levels

        Notes
        -----
        The p-values are linear interpolated from the quantiles of the
        simulated Leybourne-McCabe (KPSS) test statistic distribution
        """
        table = self.__leybourne_critical_values[model]
        # reverse the order
        y = table[:, 0]
        x = table[:, 1]
        # LM cv table contains quantiles multiplied by 100
        pvalue = np.interp(stat, x, y) / 100.0
        cv = [1.0, 5.0, 10.0]
        crit_value = np.interp(cv, np.flip(y), np.flip(x))
        cvdict = {"1%": crit_value[0], "5%": crit_value[1], "10%": crit_value[2]}
        return pvalue, cvdict

    def _tsls_arima(self, x, arlags, model):
        """
        Two-stage least squares approach for estimating ARIMA(p, 1, 1)
        parameters as an alternative to MLE estimation in the case of
        solver non-convergence

        Parameters
        ----------
        x : array_like
            data series
        arlags : int
            AR(p) order
        model : {'c','ct'}
            Constant and trend order to include in regression
            * 'c'  : constant only
            * 'ct' : constant and trend

        Returns
        -------
        arparams : int
            AR(1) coefficient plus constant
        theta : int
            MA(1) coefficient
        olsfit.resid : ndarray
            residuals from second-stage regression
        """
        endog = np.diff(x, axis=0)
        exog = lagmat(endog, arlags, trim="both")
        # add constant if requested
        if model == "ct":
            exog = add_constant(exog)
        # remove extra terms from front of endog
        endog = endog[arlags:]
        if arlags > 0:
            resids = lagmat(OLS(endog, exog).fit().resid, 1, trim="forward")
        else:
            resids = lagmat(-endog, 1, trim="forward")
        # add negated residuals column to exog as MA(1) term
        exog = np.append(exog, -resids, axis=1)
        olsfit = OLS(endog, exog).fit()
        if model == "ct":
            arparams = olsfit.params[1 : (len(olsfit.params) - 1)]
        else:
            arparams = olsfit.params[0 : (len(olsfit.params) - 1)]
        theta = olsfit.params[len(olsfit.params) - 1]
        return arparams, theta, olsfit.resid

    def _autolag(self, x):
        """
        Empirical method for Leybourne-McCabe auto AR lag detection.
        Set number of AR lags equal to the first PACF falling within the
        95% confidence interval. Maximum nuber of AR lags is limited to
        the smaller of 10 or 1/2 series length. Minimum is zero lags.

        Parameters
        ----------
        x : array_like
            data series

        Returns
        -------
        arlags : int
            AR(p) order
        """
        p = pacf(x, nlags=min(len(x) // 2, 10), method="ols")
        ci = 1.960 / np.sqrt(len(x))
        arlags = max(
            0, ([n - 1 for n, i in enumerate(p) if abs(i) < ci] + [len(p) - 1])[0]
        )
        return arlags

    def run(self, x, arlags=1, regression="c", method="mle", varest="var94"):
        """
        Leybourne-McCabe stationarity test

        The Leybourne-McCabe test can be used to test for stationarity in a
        univariate process.

        Parameters
        ----------
        x : array_like
            data series
        arlags : int
            number of autoregressive terms to include, default=None
        regression : {'c','ct'}
            Constant and trend order to include in regression
            * 'c'  : constant only (default)
            * 'ct' : constant and trend
        method : {'mle','ols'}
            Method used to estimate ARIMA(p, 1, 1) filter model
            * 'mle' : condition sum of squares maximum likelihood
            * 'ols' : two-stage least squares (default)
        varest : {'var94','var99'}
            Method used for residual variance estimation
            * 'var94' : method used in original Leybourne-McCabe paper (1994)
                        (default)
            * 'var99' : method used in follow-up paper (1999)

        Returns
        -------
        lmstat : float
            test statistic
        pvalue : float
            based on MC-derived critical values
        arlags : int
            AR(p) order used to create the filtered series
        cvdict : dict
            critical values for the test statistic at the 1%, 5%, and 10%
            levels

        Notes
        -----
        H0 = series is stationary

        Basic process is to create a filtered series which removes the AR(p)
        effects from the series under test followed by an auxiliary regression
        similar to that of Kwiatkowski et al (1992). The AR(p) coefficients
        are obtained by estimating an ARIMA(p, 1, 1) model. Two methods are
        provided for ARIMA estimation: MLE and two-stage least squares.

        Two methods are provided for residual variance estimation used in the
        calculation of the test statistic. The first method ('var94') is the
        mean of the squared residuals from the filtered regression. The second
        method ('var99') is the MA(1) coefficient times the mean of the squared
        residuals from the ARIMA(p, 1, 1) filtering model.

        An empirical autolag procedure is provided. In this context, the number
        of lags is equal to the number of AR(p) terms used in the filtering
        step. The number of AR(p) terms is set equal to the to the first PACF
        falling within the 95% confidence interval. Maximum nuber of AR lags is
        limited to 1/2 series length.

        References
        ----------
        Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y. (1992).
        Testing the null hypothesis of stationarity against the alternative of
        a unit root. Journal of Econometrics, 54: 159–178.

        Leybourne, S.J., & McCabe, B.P.M. (1994). A consistent test for a
        unit root. Journal of Business and Economic Statistics, 12: 157–166.

        Leybourne, S.J., & McCabe, B.P.M. (1999). Modified stationarity tests
        with data-dependent model-selection rules. Journal of Business and
        Economic Statistics, 17: 264-270.

        Schwert, G W. (1987). Effects of model specification on tests for unit
        roots in macroeconomic data. Journal of Monetary Economics, 20: 73–103.
        """
        if regression not in ["c", "ct"]:
            raise ValueError("LM: regression option '%s' not understood" % regression)
        if method not in ["mle", "ols"]:
            raise ValueError("LM: method option '%s' not understood" % method)
        if varest not in ["var94", "var99"]:
            raise ValueError("LM: varest option '%s' not understood" % varest)
        x = np.asarray(x)
        if x.ndim > 2 or (x.ndim == 2 and x.shape[1] != 1):
            raise ValueError(
                "LM: x must be a 1d array or a 2d array with a single column"
            )
        x = np.reshape(x, (-1, 1))
        # determine AR order if not specified
        if arlags is None:
            arlags = self._autolag(x)
        elif not isinstance(arlags, int) or arlags < 0 or arlags > int(len(x) / 2):
            raise ValueError(
                "LM: arlags must be an integer in range [0..%s]" % str(int(len(x) / 2))
            )
        # estimate the reduced ARIMA(p, 1, 1) model
        if method == "mle":
            if regression == "ct":
                reg = "t"
            else:
                reg = None

            from statsmodels.tsa.arima.model import ARIMA

            arima = ARIMA(
                x, order=(arlags, 1, 1), trend=reg, enforce_invertibility=False
            )
            arfit = arima.fit()
            resids = arfit.resid
            arcoeffs = []
            if arlags > 0:
                arcoeffs = arfit.arparams
            theta = arfit.maparams[0]
        else:
            arcoeffs, theta, resids = self._tsls_arima(x, arlags, model=regression)
        # variance estimator from (1999) LM paper
        var99 = abs(theta * np.sum(resids**2) / len(resids))
        # create the filtered series:
        #   z(t) = x(t) - arcoeffs[0]*x(t-1) - ... - arcoeffs[p-1]*x(t-p)
        z = np.full(len(x) - arlags, np.inf)
        for i in range(len(z)):
            z[i] = x[i + arlags, 0]
            for j in range(len(arcoeffs)):
                z[i] -= arcoeffs[j] * x[i + arlags - j - 1, 0]
        # regress the filtered series against a constant and
        # trend term (if requested)
        if regression == "c":
            resids = z - z.mean()
        else:
            resids = OLS(z, add_constant(np.arange(1, len(z) + 1))).fit().resid
        # variance estimator from (1994) LM paper
        var94 = np.sum(resids**2) / len(resids)
        # compute test statistic with specified variance estimator
        eta = np.sum(resids.cumsum() ** 2) / (len(resids) ** 2)
        if varest == "var99":
            lmstat = eta / var99
        else:
            lmstat = eta / var94
        # calculate pval
        lmpval, cvdict = self.__leybourne_crit(lmstat, regression)
        return lmstat, lmpval, arlags, cvdict

    def __call__(self, x, arlags=None, regression="c", method="ols", varest="var94"):
        return self.run(
            x, arlags=arlags, regression=regression, method=method, varest=varest
        )


leybourne = LeybourneMcCabeStationarity()
leybourne.__doc__ = leybourne.run.__doc__
