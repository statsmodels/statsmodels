#Kaplan-Meier Estimator

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy import stats
from scikits.statsmodels.iolib.table import SimpleTable

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

        endog: index (starting at zero) of the column
            containing the endogenous variable (time)

        exog: index of the column containing the exogenous
            variable (must be catagorical). If exog = None, this
            is equivalent to a single survival curve

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
        >>> import scikits.statsmodels.api as sm
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from scikits.statsmodels.sandbox.survival2 import KaplanMeier
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
        production, the columnof dta with index one (1).

        With censoring:

        >>> censoring = np.ones_like(dta[:,0])
        >>> censoring[dta[:,0] > 80] = 0
        >>> dta = np.c_[dta,np.transpose(censoring)]
        >>> dta[range(5),:]
        array([[  7.00000000e+00,   1.13800000e-02,   1.00000000e+00],
               [  9.00000000e+00,   1.13800000e-02,   1.00000000e+00],
               [  1.30000000e+01,   1.13800000e-02,   1.00000000e+00],
               [  1.40000000e+01,   1.13800000e-02,   1.00000000e+00],
               [  2.60000000e+01,   1.13800000e-02,   1.00000000e+00]])

        >>> km3 = KaplanMeier(dta,0,exog=1,censoring=2)
        >>> km3.fit()

        Test for difference of survival curves

        >>> log_rank = km3.test_diff([0,1])

        The zeroth element of log_rank is the chi-square test statistic
        for the difference between the survival curves for exog = 1
        and exog = 0, the index one element is the degrees of freedom for
        the test, and the index two element is the p-value for the test

        >>> wilcoxon = km3.test_diff([0,1], rho=1)

        wilcoxon is the equivalent information as logg_rank, but for the
        Peto and Peto modification to the Gehan-Wilcoxon test.
    """

    def __init__(self, data, endog, exog=None, censoring=None):
    #TODO: optional choice of left or right continuous?
        self.censoring = censoring
        self.exog = exog
        self.data = data
        self.endog = endog

    def fit(self):
        self.results = []
        self.ts = []
        self.censorings = []
        self.event = []
        if self.exog == None:
            self.fitting_proc(self.data)
        else:
            groups = np.unique(self.data[:,self.exog])
            self.groups = groups
            #TODO: vectorize loop?
            for g in groups:
                group = self.data[self.data[:,self.exog] == g]
                self.fitting_proc(group)

    def plot(self):
        if self.exog == None:
            self.plotting_proc(0)
        else:
            #TODO: vectorize loop?
            for g in range(len(self.groups)):
                self.plotting_proc(g)
        plt.ylim(ymax=1.05)
        plt.ylabel('Survival')
        plt.xlabel('Time')
        #TODO: check plotting for multiple censored observations
        #at one time (formula for distance between tick marks?)

    def summary(self):
        if self.exog == None:
            self.summary_proc(0)
        else:
            #TODO: vectorize loop?
            for g in range(len(self.groups)):
                self.summary_proc(g)

    def fitting_proc(self, group):
        #TODO: calculate hazard?
        #TODO: check multiple censored observations at one time
        #TODO: check non-int values for exog (strings)
        #TODO: implement fitting with np.in1d?
        t = group[:,self.endog]
        if self.censoring == None:
            events = np.bincount(t.astype(int))
            t = np.unique(t)
            events = events[:,list(t)]
            events = events.astype(float)
            eventsSum = np.cumsum(events)
            eventsSum = np.r_[0,eventsSum]
            n = len(group) - eventsSum[:-1]
        else:
            censoring = group[:,self.censoring]
            reverseCensoring = -1*(censoring - 1)
            events = np.bincount(t.astype(int),censoring)
            censored = np.bincount(t.astype(int),reverseCensoring)
            t = np.unique(t)
            censored = censored[:,list(t)]
            censored = censored.astype(float)
            censoredSum = np.cumsum(censored)
            censoredSum = np.r_[0,censoredSum]
            events = events[:,list(t)]
            events = events.astype(float)
            eventsSum = np.cumsum(events)
            eventsSum = np.r_[0,eventsSum]
            n = len(group) - eventsSum[:-1] - censoredSum[:-1]
            (self.censorings).append(censored)
        survival = np.cumprod(1-events/n)
        var = ((survival*survival) *
               np.cumsum(events/(n*(n-events))))
        se = np.sqrt(var)
        (self.results).append(np.array([survival,se]))
        (self.ts).append(t)
        (self.event).append(events)
        #TODO: save less data?

    def plotting_proc(self, g):
        survival = self.results[g][0]
        t = self.ts[g]
        e = (self.event)[g]
        if self.censoring != None:
            c = self.censorings[g]
            csurvival = survival[c != 0]
            ct = t[c != 0]
            plt.vlines(ct,csurvival+0.02,csurvival-0.02)
        x = np.repeat(t[e != 0], 2)
        y = np.repeat(survival[e != 0], 2)
        if self.ts[g][-1] in t[e != 0]:
            x = np.r_[0,x]
            y = np.r_[1,1,y[:-1]]
        else:
            x = np.r_[0,x,self.ts[g][-1]]
            y = np.r_[1,1,y]
        plt.plot(x,y)

    def summary_proc(self, g):
        if self.exog != None:
            myTitle = ('exog = ' + str(self.groups[g]) + '\n')
        else:
            myTitle = "Kaplan-Meier Curve"
        table = np.transpose(self.results[g])
        table = np.c_[np.transpose(self.ts[g]),table]
        table = SimpleTable(table, headers=['Time','Survival','Std. Err'],
                            title = myTitle)
        print(table)

    def test_diff(self, groups, rho=0):

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

        Returns
        -------
        An array whose zeroth element is the chi-square test statistic for
        the global null hypothesis, that all survival curves are equal,
        the index one element is degrees of freedom for the test, and the
        index two element is the p-value for the test.

        Examples
        --------

        >>> import scikits.statsmodels.api as sm
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from scikits.statsmodels.sandbox.survival2 import KaplanMeier
        >>> dta = sm.datasets.strikes.load()
        >>> dta = dta.values()[-1]
        >>> censoring = np.ones_like(dta[:,0])
        >>> censoring[dta[:,0] > 80] = 0
        >>> dta = np.c_[dta,np.transpose(censoring)]
        >>> km = KaplanMeier(dta,0,exog=1,censoring=2)
        >>> km.fit()

        Test for difference of survival curves

        >>> log_rank = km.test_diff([0,1])

        The zeroth element of log_rank is the chi-square test statistic
        for the difference between the survival curves for exog = 1
        and exog = 0, the index one element is the degrees of freedom for
        the test, and the index two element is the p-value for the test

        >>> wilcoxon = km.test_diff([0,1], rho=1)

        wilcoxon is the equivalent information as logg_rank, but for the
        Peto and Peto modification to the Gehan-Wilcoxon test.
        """
        groups = np.asarray(groups)
        if self.exog == None:
            raise ValueError("Need an exogenous variable for logrank test")

        elif (np.in1d(groups,self.groups)).all():
            data = self.data[np.in1d(self.data[:,self.exog],groups)]
            t = data[:,self.endog]
            tind = np.unique(t)
            NK = []
            N = []
            D = []
            Z = []
            s = KaplanMeier(data,self.endog,censoring=self.censoring)
            s.fit()
            s = (s.results[0][0]) ** (rho)
            if self.censoring == None:
                for g in groups:
                    dk = np.bincount((t[data[:,self.exog] == g]).astype(int))
                    d = np.bincount(t.astype(int))
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
                    nk = len(data[data[:,self.exog] == g]) - dkSum[:-1]
                    n = len(data) - dSum[:-1]
                    ek = (nk * d)/(n)
                    O.append(np.sum(dk))
                    E.append(np.sum(ek))
                    NK.append(nk)
                    N.append(n)
                    D.append(d)
            else:
                for g in groups:
                    censoring = data[:,self.censoring]
                    reverseCensoring = -1*(censoring - 1)
                    censored = np.bincount(t.astype(int),reverseCensoring)
                    ck = np.bincount((t[self.data[:,self.exog] == g]).astype(int),
                                     reverseCensoring[self.data[:,self.exog] == g])
                    dk = np.bincount((t[data[:,self.exog] == g]).astype(int),
                                     censoring[data[:,self.exog] == g])
                    d = np.bincount(t.astype(int),censoring)
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
                    nk = (len(data[data[:,self.exog] == g]) - dkSum[:-1]
                          - ck[:-1])
                    n = len(data) - dSum[:-1] - censoredSum[:-1]
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
                                * np.array([s]*len(D))
                            ,np.transpose(NK/N))
            np.fill_diagonal(sigma, np.diagonal(np.dot((NK/N)
                                                  * ((N - D)/(N - 1)) * D
                                                       * np.array([s]*len(D))
                                                  ,np.transpose(1 - (NK/N)))))

            chisq = np.dot(np.transpose(Z),np.dot(la.pinv(sigma), Z))
            df = len(groups) - 1
            return np.array([chisq, df, stats.chi2.sf(chisq,df)])
        else:
            raise ValueError("groups must be in column exog")
