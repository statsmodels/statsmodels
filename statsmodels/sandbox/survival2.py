#Kaplan-Meier Estimator

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import fmin_ncg

from statsmodels.iolib.table import SimpleTable
from statsmodels.base.model import LikelihoodModel, LikelihoodModelResults

##Need to update all docstrings

class Survival(object):

    """
    Survival(...)
        Survival(data, time1, time2=None, censoring=None)

        Create an object to store survival data for precessing
        by other survival analysis functions

        Parameters
        -----------

        censoring: index of the column containing an indicator
            of whether an observation is an event, or a censored
            observation, with 0 for censored, and 1 for an event

        data: array_like
            An array, with observations in each row, and
            variables in the columns

        time1 : if time2=None, index of comlumn containing the duration
            that the suject survivals and remains uncensored (e.g. observed
            survival time), if time2 is not None, then time1 is the index of
            a column containing start times for the observation of each subject
            (e.g. oberved survival time is end time minus start time)

        time2: index of column containing end times for each observation

        Attributes
        -----------

        times: vectore of survival times

        censoring: vector of censoring indicators

        Examples
        ---------

        see other survival analysis functions for examples of usage with those
        functions

    """

    def __init__(self, time1, time2=None, censoring=None, data=None):
        if not data is None:
            data = np.asarray(data)
            if censoring is None:
                self.censoring = None
            else:
                self.censoring = (data[:,censoring]).astype(int)
            if time2 is None:
                self.times = (data[:,time1]).astype(int)
            else:
                self.times = (((data[:,time2]).astype(int))
                              - ((data[:,time1]).astype(int)))
        else:
            time1 = (np.asarray(time1)).astype(int)
            if not time2 is None:
                time2 = (np.array(time2)).astype(int)
                self.times = time2 - time1
            else:
                self.times = time1
            if censoring is None:
                self.censoring == None
            else:
                self.censoring = (np.asarray(censoring)).astype(int)


class KaplanMeier(object):

    """
    KaplanMeier(...)
        KaplanMeier(data, endog, exog=None, censoring=None)

        Create an object of class KaplanMeier for estimating
        Kaplan-Meier survival curves.

        Parameters
        ----------
        data: array_like
            An array, with observations in each row, and
            variables in the columns

        surv: Survival object containing desire times and censoring

        endog: index (starting at zero) of the column
            containing the endogenous variable (time)

        exog: index of the column containing the exogenous
            variable (must be catagorical). If exog = None, this
            is equivalent to a single survival curve. Alternatively,
            this can be a vector of exogenous variables index in the same
            manner as data provided either from data or surv

        censoring: index of the column containing an indicator
            of whether an observation is an event, or a censored
            observation, with 0 for censored, and 1 for an event


        Attributes
        -----------
        censorings: List of censorings associated with each unique
            time, at each value of exog

        events: List of the number of events at each unique time
            for each value of exog

        results: List of arrays containing estimates of the value
            value of the survival function and its standard error
            at each unique time, for each value of exog

        ts: List of unique times for each value of exog

        Methods
        -------
        fit: Calcuate the Kaplan-Meier estimates of the survival
            function and its standard error at each time, for each
            value of exog

        plot: Plot the survival curves using matplotlib.plyplot

        summary: Display the results of fit in a table. Gives results
            for all (including censored) times

        test_diff: Test for difference between survival curves

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from statsmodels.sandbox.survival2 import KaplanMeier
        >>> dta = sm.datasets.strikes.load()
        >>> dta = dta.values()[-1]
        >>> dta[range(5),:]
        array([[  7.00000000e+00,   1.13800000e-02],
               [  9.00000000e+00,   1.13800000e-02],
               [  1.30000000e+01,   1.13800000e-02],
               [  1.40000000e+01,   1.13800000e-02],
               [  2.60000000e+01,   1.13800000e-02]])
        >>> km = KaplanMeier(dta,0)
        >>> km.fit()
        >>> km.plot()

        Doing

        >>> km.summary()

        will display a table of the estimated survival and standard errors
        for each time. The first few lines are

                  Kaplan-Meier Curve
        =====================================
         Time     Survival        Std. Err
        -------------------------------------
         1.0   0.983870967742 0.0159984306572
         2.0   0.91935483871  0.0345807888235
         3.0   0.854838709677 0.0447374942184
         4.0   0.838709677419 0.0467104592871
         5.0   0.822580645161 0.0485169952543

        Doing

        >>> plt.show()

        will plot the survival curve

        Mutliple survival curves:

        >>> km2 = KaplanMeier(dta,0,exog=1)
        >>> km2.fit()

        km2 will estimate a survival curve for each value of industrial
        production, the column of dta with index one (1).

        With censoring:

        >>> censoring = np.ones_like(dta[:,0])
        >>> censoring[dta[:,0] > 80] = 0
        >>> dta = np.c_[dta,censoring]
        >>> dta[range(5),:]
        array([[  7.00000000e+00,   1.13800000e-02,   1.00000000e+00],
               [  9.00000000e+00,   1.13800000e-02,   1.00000000e+00],
               [  1.30000000e+01,   1.13800000e-02,   1.00000000e+00],
               [  1.40000000e+01,   1.13800000e-02,   1.00000000e+00],
               [  2.60000000e+01,   1.13800000e-02,   1.00000000e+00]])

        >>> km3 = KaplanMeier(dta,0,exog=1,censoring=2)
        >>> km3.fit()

        Test for difference of survival curves

        >>> log_rank = km3.test_diff([0.0645,-0.03957])

        The zeroth element of log_rank is the chi-square test statistic
        for the difference between the survival curves for exog = 0.0645
        and exog = -0.03957, the index one element is the degrees of freedom for
        the test, and the index two element is the p-value for the test

        Groups with nan names

        >>> groups = np.ones_like(dta[:,1])
        >>> groups = groups.astype('S4')
        >>> groups[dta[:,1] > 0] = 'high'
        >>> groups[dta[:,1] <= 0] = 'low'
        >>> dta = dta.astype('S4')
        >>> dta[:,1] = groups
        >>> dta[range(5),:]
        array([['7.0', 'high', '1.0'],
               ['9.0', 'high', '1.0'],
               ['13.0', 'high', '1.0'],
               ['14.0', 'high', '1.0'],
               ['26.0', 'high', '1.0']],
              dtype='|S4')
        >>> km4 = KaplanMeier(dta,0,exog=1,censoring=2)
        >>> km4.fit()

    """

    def __init__(self, surv, exog=None, data=None):
        censoring = surv.censoring
        times = surv.times
        if not exog is None:
            if not data is None:
                data = np.asarray(data)
                if data.ndim != 2:
                    raise ValueError("Data array must be 2d")
                exog = data[:,exog]
            else:
                if type(exog) == int:
                    raise ValueError("""int exog must be column in data, which was not
                                    provided""")
                exog = np.asarray(exog)
            if exog.dtype == float or exog.dtype == int:
                if not censoring is None:
                    data = np.c_[times,censoring,exog]
                    data = data[~np.isnan(data).any(1)]
                    self.times = (data[:,0]).astype(int)
                    self.censoring = (data[:,1]).astype(int)
                    self.exog = data[:,2]
                else:
                    data = np.c_[times,exog]
                    data = data[~np.isnan(data).any(1)]
                    self.times = (data[:,0]).astype(int)
                    self.exog = data[:,1]
                del(data)
            else:
                exog = exog[~np.isnan(times)]
            if not censoring is None:
                censoring = censoring[~np.isnan(times)]
            times = times[~np.isnan(times)]
            if  not censoring is None:
                self.times = (times[~np.isnan(censoring)]).astype(int)
                self.exog = exog[~np.isnan(censoring)]
                self.censoring = (censoring[~np.isnan(censoring)]).astype(int)
                self.df_resid = len(self.exog) - 1
        else:
            self.exog = None
            data = np.c_[times,censoring]
            data = data[~np.isnan(data).any(1)]
            self.times = (data[:,0]).astype(int)
            self.censoring = (data[:,1]).astype(int)
            del(data)

    def fit(self, CI_transform="log-log", force_CI_0_1=True):
        """
        Calculate the Kaplan-Meier estimator of the survival function
        """
        self.results = []
        self.ts = []
        self.censorings = []
        self.event = []
        self.params = np.array([])
        self.normalized_cov_params = np.array([])
        if self.exog is None:
            self._fitting_proc(self.times, self.censoring, CI_transform,
                              force_CI_0_1)
        else:
            groups = np.unique(self.exog)
            self.groups = groups
            for g in groups:
                t = (self.times[self.exog == g])
                if not self.censoring is None:
                    censoring = (self.censoring[self.exog == g])
                else:
                    censoring = None
                self._fitting_proc(t, censoring, CI_transform, force_CI_0_1)
        return KMResults(self, self.params, self.normalized_cov_params)

    def _fitting_proc(self, t, censoring, CI_transform, force_CI):
        """
        For internal use
        """
        if censoring is None:
            n = len(t)
            events = np.bincount(t)
            t = np.unique(t)
            events = events[:,list(t)]
            events = events.astype(float)
            eventsSum = np.cumsum(events)
            eventsSum = np.r_[0,eventsSum]
            n -= eventsSum[:-1]
        else:
            reverseCensoring = -1*(censoring - 1)
            events = np.bincount(t,censoring)
            censored = np.bincount(t,reverseCensoring)
            t = np.unique(t)
            censored = censored[:,list(t)]
            censored = censored.astype(float)
            censoredSum = np.cumsum(censored)
            censoredSum = np.r_[0,censoredSum]
            events = events[:,list(t)]
            events = events.astype(float)
            eventsSum = np.cumsum(events)
            eventsSum = np.r_[0,eventsSum]
            n = len(censoring) - eventsSum[:-1] - censoredSum[:-1]
            (self.censorings).append(censored)
        survival = np.cumprod(1-events/n)
        var = ((survival*survival) *
               np.cumsum(events/(n*(n-events))))
        se = np.sqrt(var)
        if CI_transform == "log":
            lower = (np.exp(np.log(survival) - 1.96 *
                                    (se * (1/(survival)))))
            upper = (np.exp(np.log(survival) + 1.96 *
                                    (se * (1/(survival)))))
        if CI_transform == "log-log":
            lower = (np.exp(-np.exp(np.log(-np.log(survival)) - 1.96 *
                                    (se * (1/(survival * np.log(survival)))))))
            upper = (np.exp(-np.exp(np.log(-np.log(survival)) + 1.96 *
                                    (se * (1/(survival * np.log(survival)))))))
        if force_CI:
            lower[lower < 0] = 0
            upper[upper > 1] = 1
        self.params = np.r_[self.params,survival]
        self.normalized_cov_params = np.r_[self.normalized_cov_params, se]
        (self.results).append(np.array([survival,se,lower,upper]))
        (self.ts).append(t)
        (self.event).append(events)

class CoxPH(LikelihoodModel):
    ##Add efron fitting, and other methods

    """
    Fit a cox proportional harzard model from survival data
    """

    def __init__(self, surv, exog, data=None):
        censoring = surv.censoring
        times = surv.times
        if data is not None:
            data = np.asarray(data)
            if data.ndim != 2:
                raise ValueError("Data array must be 2d")
            exog = data[:,exog]
        else:
            exog = np.asarray(exog)
        if exog.dtype == float or exog.dtype == int:
            if censoring != None:
                data = np.c_[times,censoring,exog]
                data = data[~np.isnan(data).any(1)]
                self.times = (data[:,0]).astype(int)
                self.censoring = (data[:,1]).astype(int)
                self.exog = data[:,2:]
            else:
                data = np.c_[times,exog]
                data = data[~np.isnan(data).any(1)]
                self.times = (data[:,0]).astype(int)
                self.exog = data[:,1:]
            del(data)
        else:
            exog = exog[~np.isnan(times)]
        if censoring is not None:
            censoring = censoring[~np.isnan(times)]
        times = times[~np.isnan(times)]
        if censoring is not None:
            times = (times[~np.isnan(censoring)]).astype(int)
            exog = exog[~np.isnan(censoring)]
            censoring = (censoring[~np.isnan(censoring)]).astype(int)
        self.times = (times[~np.isnan(exog).any(1)]).astype(int)
        self.censoring = (censoring[~np.isnan(exog).any(1)]).astype(int)
        self.exog = (exog[~np.isnan(exog).any(1)]).astype(float)
        self.df_resid = len(self.exog) - 1

    def loglike(self, b):
        thetas = np.exp(np.dot(self.exog, b))
        ind = self.censoring == 1
        ti = self.times[ind]
        logL = (np.dot(self.exog[ind], b)).sum()
        for t in ti:
            logL -= np.log((thetas[self.times >= t]).sum())
        return logL

    def score(self, b):
        thetas = np.exp(np.dot(self.exog, b))
        ind = self.censoring == 1
        ti = self.times[ind]
        score = (self.exog[ind]).sum(0)
        for t in ti:
            ind = self.times >= t
            thetaj = thetas[ind]
            Xj = self.exog[ind]
            score -= (np.dot(thetaj, Xj))/(thetaj.sum())
        return score

    def hessian(self, b):
        thetas = np.exp(np.dot(self.exog, b))
        ti = self.times[self.censoring == 1]
        hess = 0
        for t in ti:
            ind = self.times >= t
            thetaj = thetas[ind]
            Xj = self.exog[ind]
            thetaX = np.mat(np.dot(thetaj, Xj))
            hess += (((np.dot(Xj.T, (Xj * thetaj[:,np.newaxis])))/(thetaj.sum()))
                     - ((np.array(thetaX.T * thetaX))/((thetaj.sum())**2)))
        return -hess

    def information(self, b):
        return -self.hessian(b)

    def covariance(self, b):
        return la.pinv(self.information(b))

    def fit(self, start_params=None, method='newton', maxiter=100,
            full_output=1,disp=1, fargs=(), callback=None, retall=0, **kwargs):
        results = super(CoxPH, self).fit(start_params, method, maxiter,
            full_output,disp, fargs, callback, retall, **kwargs)
        return CoxResults(self, results.params,
                               self.covariance(results.params))

class KMResults(LikelihoodModelResults):
    """
    Results for a Kaplan-Meier model
    """

    def __init__(self, model, params, normalized_cov_params=None, scale=1.0):
        super(KMResults, self).__init__(model, params, normalized_cov_params,
                                        scale)
        self.results = model.results
        self.times = model.times
        self.ts = model.ts
        self.censoring = model.censoring
        self.censorings = model.censorings
        self.exog = model.exog
        self.event = model.event
        self.groups = model.groups

    def test_diff(self, groups, rho=None, weight=None):

        """
        test_diff(groups, rho=0)

        Test for difference between survival curves

        Parameters
        ----------
        groups: A list of the values for exog to test for difference.
        tests the null hypothesis that the survival curves for all
        values of exog in groups are equal

        rho: compute the test statistic with weight S(t)^rho, where
        S(t) is the pooled estimate for the Kaplan-Meier survival function.
        If rho = 0, this is the logrank test, if rho = 0, this is the
        Peto and Peto modification to the Gehan-Wilcoxon test.

        weight: User specified function that accepts as its sole arguement
        an array of times, and returns an array of weights for each time
        to be used in the test

        Returns
        -------
        An array whose zeroth element is the chi-square test statistic for
        the global null hypothesis, that all survival curves are equal,
        the index one element is degrees of freedom for the test, and the
        index two element is the p-value for the test.

        Examples
        --------

        >>> import statsmodels.api as sm
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from statsmodels.sandbox.survival2 import KaplanMeier
        >>> dta = sm.datasets.strikes.load()
        >>> dta = dta.values()[-1]
        >>> censoring = np.ones_like(dta[:,0])
        >>> censoring[dta[:,0] > 80] = 0
        >>> dta = np.c_[dta,censoring]
        >>> km = KaplanMeier(dta,0,exog=1,censoring=2)
        >>> km.fit()

        Test for difference of survival curves

        >>> log_rank = km3.test_diff([0.0645,-0.03957])

        The zeroth element of log_rank is the chi-square test statistic
        for the difference between the survival curves using the log rank test
        for exog = 0.0645 and exog = -0.03957, the index one element
        is the degrees of freedom for the test, and the index two element
        is the p-value for the test

        >>> wilcoxon = km.test_diff([0.0645,-0.03957], rho=1)

        wilcoxon is the equivalent information as log_rank, but for the
        Peto and Peto modification to the Gehan-Wilcoxon test.

        User specified weight functions

        >>> log_rank = km3.test_diff([0.0645,-0.03957], weight=np.ones_like)

        This is equivalent to the log rank test

        More than two groups

        >>> log_rank = km.test_diff([0.0645,-0.03957,0.01138])

        The test can be performed with arbitrarily many groups, so long as
        they are all in the column exog
        """
        groups = np.asarray(groups)
        exog = self.exog
        if exog is None:
            raise ValueError("Need an exogenous variable for tests")

        elif (np.in1d(groups,self.groups)).all():
            ind = np.in1d(exog,groups)
            t = self.times[ind]
            if not self.censoring is None:
                censoring = self.censoring[ind]
            else:
                censoring = None
            del(ind)
            tind = np.unique(t)
            NK = []
            N = []
            D = []
            Z = []
            if rho is not None and weight is not None:
                raise ValueError("Must use either rho or weights, not both")

            elif rho != None:
                s = KaplanMeier(Survival(t,censoring=censoring))
                s.fit()
                s = (s.results[0][0]) ** (rho)
                s = np.r_[1,s[:-1]]

            elif weight is not None:
                s = weight(tind)

            else:
                s = np.ones_like(tind)

            if censoring is None:
                for g in groups:
                    n = len(t)
                    exog_idx = self.exog == g
                    dk = np.bincount(t[exog_idx])
                    d = np.bincount(t)
                    if np.max(tind) != len(dk):
                        dif = np.max(tind) - len(dk) + 1
                        dk = np.r_[dk,[0]*dif]
                    dk = dk[:,list(tind)]
                    d = d[:,list(tind)]
                    dk = dk.astype(float)
                    d = d.astype(float)
                    dkSum = np.cumsum(dk)
                    dSum = np.cumsum(d)
                    dkSum = np.r_[0,dkSum]
                    dSum = np.r_[0,dSum]
                    nk = len(exog[exog_idx]) - dkSum[:-1]
                    n -= dSum[:-1]
                    d = d[n>1]
                    dk = dk[n>1]
                    nk = nk[n>1]
                    n = n[n>1]
                    s = s[n>1]
                    ek = (nk * d)/(n)
                    Z.append(np.sum(s * (dk - ek)))
                    NK.append(nk)
                    N.append(n)
                    D.append(d)
            else:
                for g in groups:
                    exog_idx = self.exog == g
                    reverseCensoring = -1*(censoring - 1)
                    censored = np.bincount(t,reverseCensoring)
                    ck = np.bincount(t[exog_idx],
                                     reverseCensoring[exog_idx])
                    dk = np.bincount(t[exog_idx],
                                     censoring[exog_idx])
                    d = np.bincount(t,censoring)
                    if np.max(tind) != len(dk):
                        dif = np.max(tind) - len(dk) + 1
                        dk = np.r_[dk,[0]*dif]
                        ck = np.r_[ck,[0]*dif]
                    dk = dk[:,list(tind)]
                    ck = ck[:,list(tind)]
                    d = d[:,list(tind)]
                    dk = dk.astype(float)
                    d = d.astype(float)
                    ck = ck.astype(float)
                    dkSum = np.cumsum(dk)
                    dSum = np.cumsum(d)
                    ck = np.cumsum(ck)
                    ck = np.r_[0,ck]
                    dkSum = np.r_[0,dkSum]
                    dSum = np.r_[0,dSum]
                    censored = censored[:,list(tind)]
                    censored = censored.astype(float)
                    censoredSum = np.cumsum(censored)
                    censoredSum = np.r_[0,censoredSum]
                    nk = (len(self.exog[self.exog == g]) - dkSum[:-1]
                          - ck[:-1])
                    n = len(censoring) - dSum[:-1] - censoredSum[:-1]
                    d = d[n>1]
                    dk = dk[n>1]
                    nk = nk[n>1]
                    n = n[n>1]
                    s = s[n>1]
                    ek = (nk * d)/(n)
                    Z.append(np.sum(s * (dk - ek)))
                    NK.append(nk)
                    N.append(n)
                    D.append(d)
            Z = np.array(Z)
            N = np.array(N)
            D = np.array(D)
            NK = np.array(NK)
            sigma = -1 * np.dot((NK/N) * ((N - D)/(N - 1)) * D
                                * np.array([(s ** 2)]*len(D))
                            ,np.transpose(NK/N))
            np.fill_diagonal(sigma, np.diagonal(np.dot((NK/N)
                                                  * ((N - D)/(N - 1)) * D
                                                       * np.array([(s ** 2)]*len(D))
                                                  ,np.transpose(1 - (NK/N)))))
            chisq = np.dot(np.transpose(Z),np.dot(la.pinv(sigma), Z))
            df = len(groups) - 1
            return np.array([chisq, df, stats.chi2.sf(chisq,df)])
        else:
            raise ValueError("groups must be in column exog")

    def isolate_curve(self, exog):
        """
        Get results for one curve from a model that fits mulitple survival
        curves

        Parameters
        ----------

        exog: The value of that exogenous variable for the curve to be
        isolated.

        returns
        --------
        A SurvivalResults object for the isolated curve
        """

        exogs = self.exog
        if exog is None:
            raise ValueError("Already a single curve")
        else:
            ind = (list((self.model).groups)).index(exog)
            results = self.results[ind]
            ts = self.ts[ind]
            censoring = self.censoring[exogs == exog]
            censorings = self.censorings[ind]
            event = self.event[ind]
            r = KMResults(self.model, results[0], results[1])
            r.results = results
            r.ts = ts
            r.censoring = censoring
            r.censorings = censorings
            r.event = event
            r.exog = None
            r.groups = None
            return r

    def plot(self, confidence_band=False):
        """
        Plot the estimated survival curves. After using this method
        do

        plt.show()

        to display the plot
        """
        plt.figure()
        if self.exog is None:
            self._plotting_proc(0, confidence_band)
        else:
            for g in range(len(self.groups)):
                self._plotting_proc(g, confidence_band)
        plt.ylim(ymax=1.05)
        plt.ylabel('Survival')
        plt.xlabel('Time')

    def summary(self):
        """
        Print a set of tables containing the estimates of the survival
        function, and its standard errors
        """
        if self.exog is None:
            self._summary_proc(0)
        else:
            for g in range(len(self.groups)):
                self._summary_proc(g)

    def _plotting_proc(self, g, confidence_band):
        """
        For internal use
        """
        survival = self.results[g][0]
        t = self.ts[g]
        e = (self.event)[g]
        if self.censoring is not None:
            c = self.censorings[g]
            csurvival = survival[c != 0]
            ct = t[c != 0]
            if len(ct) != 0:
                plt.vlines(ct,csurvival+0.02,csurvival-0.02)
        t = np.repeat(t[e != 0], 2)
        s = np.repeat(survival[e != 0], 2)
        if confidence_band:
            lower = self.results[g][2]
            upper = self.results[g][3]
            lower = np.repeat(lower[e != 0], 2)
            upper = np.repeat(upper[e != 0], 2)
        if self.ts[g][-1] in t[e != 0]:
            t = np.r_[0,t]
            s = np.r_[1,1,s[:-1]]
            if confidence_band:
                lower = np.r_[1,1,lower[:-1]]
                upper = np.r_[1,1,upper[:-1]]
        else:
            t = np.r_[0,t,self.ts[g][-1]]
            s = np.r_[1,1,s]
            if confidence_band:
                lower = np.r_[1,1,lower]
                upper = np.r_[1,1,upper]
        if confidence_band:
            plt.plot(t,(np.c_[lower,upper]),'k--')
        plt.plot(t,s)

    def _summary_proc(self, g):
        """
        For internal use
        """
        if self.exog is not None:
            myTitle = ('exog = ' + str(self.groups[g]) + '\n')
            table = np.transpose(self.results[g])
            table = np.c_[np.transpose(self.ts[g]),table]
            table = SimpleTable(table, headers=['Time','Survival','Std. Err',
                                                'Lower 95% CI', 'Upper 95% CI'],
                                title = myTitle)
        else:
            myTitle = "Kaplan-Meier Curve"
            table = np.transpose(self.results)
            table = np.c_[self.ts,table]
            table = SimpleTable(table, headers=['Time','Survival','Std. Err',
                                                'Lower 95% CI', 'Upper 95% CI'],
                                title = myTitle)
        print(table)

class CoxResults(LikelihoodModelResults):

    """
    Results for cox proportional hazard models
    """

    def test_coefficients(self, method="wald"):
        ##Need to check values for tests
        params = self.params
        model = self.model
        cov_params = self.normalized_cov_params
        ##Other methods (e.g. score?)
        if method == "wald":
            se = 1/(np.sqrt(np.diagonal(model.information(params))))
            z = params/se
        return np.c_[params,se,z,2 * stats.norm.sf(np.abs(z), 0, 1)]

    def wald_test(self):
        ##Need to check values for tests
        params = self.params
        model = self.model
        cov_params = self.normalized_cov_params
        return np.dot(((params) ** 2), model.information(params))

    def score_test(self):
        ##Need to check values for tests
        params = self.params
        model = self.model
        cov_params = self.normalized_cov_params
        return (np.dot(((model).score(np.zeros_like(params)))** 2, cov_params))

    def likelihood_ratio_test(self, restricted="zeros"):
        ##Need to check values for tests
        params = self.params
        model = self.model
        if restricted == "zeros":
            return (2 * (model.loglike(params)
                         - model.loglike(np.zeros_like(params))))
        elif isinstance(restricted, CoxResults):
            restricted = restricted.model.loglike(restricted.params)
            return 2 * (model.loglike(params) - restricted)
        else:
            raise ValueError('''restricted must be either CoxResults instance, or
                             "zeros"''')

    def conf_int(self, alpha=.05, cols=None, method='default', exp=True):
        ##Need to check values
        CI = super(CoxResults, self).conf_int(alpha, cols, method)
        if exp:
            CI = np.exp(CI)
        return CI
