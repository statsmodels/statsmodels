#Kaplan-Meier Estimator

import numpy as np
import matplotlib.pyplot as plt
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

        Examples
        --------
        >>> import scikits.statsmodels.api as sm
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from scikits.statsmodels.sandbox.survival2 import KaplanMeier
        >>> dta = sm.datasets.strikes.load()
        >>> dta = dta.values()[-1]
        >>> dta[:,range(5)]
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

    #TODO: Log Rank Test?
    #TODO: show_life_tables method?
