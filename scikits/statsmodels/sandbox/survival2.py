#Kaplan-Meier Estimator

import numpy as np
import matplotlib.pyplot as plt
from scikits.statsmodels.iolib.table import SimpleTable

class KaplanMeier(object):

    #TODO: docstring

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
        survival = np.cumprod(1-events/n)
        var = ((survival*survival) *
               np.cumsum(events/(n*(n-events))))
        se = np.sqrt(var)
        (self.results).append(np.array([survival,se]))
        (self.ts).append(t)
        (self.event).append(events)
        (self.censorings).append(censored)
        #TODO: save less data?

    def plotting_proc(self, g):
        survival = self.results[g][0]
        c = self.censorings[g]
        t = self.ts[g]
        csurvival = survival[c != 0]
        ct = t[c != 0]
        e = self.event[g]
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
            myTitle = "Kaplan Meier Curve"
        table = np.transpose(self.results[g])
        table = np.c_[np.transpose(self.ts[g]),table]
        table = SimpleTable(table, headers=['Time','Survival','Std. Err'],
                            title = myTitle)
        print(table)

    #TODO: Log Rank Test?
    #TODO: show_life_tables method?
