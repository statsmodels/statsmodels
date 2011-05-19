#Kaplan-Meier Estimator

import numpy as np
import matplotlib.pyplot as plt

class KaplanMeier(object):

    def __init__(self, data, endog, exog, censoring=None):
    #TODO: optional choice of left or right continuous?
    #TODO: default exog=None, for single curve
        if censoring == None:
            #TODO: change self.fit to accept censoring=None
            #instead of adding all ones?
            censoring_vec = np.ones_like(data[:,endog])
            data = np.c_[data,censoring_vec]
            self.censoring = len(data[0]) - 1
        else:
            self.censoring = censoring
        self.data = data
        self.endog = endog
        self.exog = exog

    def fit(self):
        #TODO: calculate standard errors
        #TODO: check multiple censored observations at one time
        #TODO: check non-int values for exog (strings)
        groups = np.unique(self.data[:,self.exog])
        results = []
        ts = []
        censorings = []
        tEvents = []
        #TODO: vectorize loop?
        for g in groups:
            group = self.data[self.data[:,self.exog] == g]
            t = group[:,self.endog]
            censoring = group[:,self.censoring]
            reverseCensoring = -1*(censoring - 1)
            events = np.bincount(t.astype(int),censoring)
            censored = np.bincount(t.astype(int),reverseCensoring)
            t = np.unique(t)
            events = events[:,list(t)]
            censored = censored[:,list(t)]
            events = events.astype(float)
            censored = censored.astype(float)
            censoredSum = np.cumsum(censored)
            eventsSum = np.cumsum(events)
            eventsSum = np.r_[0,eventsSum]
            censoredSum = np.r_[0,censoredSum]
            n = len(group) - eventsSum[:-1] - censoredSum[:-1]
            n = n[events != 0]
            tEvent = t[events != 0]
            events = events[events != 0]
            survival = np.cumprod(1-events/n)
            results.append(survival)
            ts.append(t)
            tEvents.append(tEvent)
            censorings.append(censored)
        #TODO: save less data?
        self.groups = groups
        self.tEvents = tEvents
        self.ts = ts
        self.censorings = censorings
        self.results = results

    def plot(self):
        for g in range(len(self.groups)):
            x = np.repeat(self.tEvents[g], 2)
            y = np.repeat(self.results[g], 2)
            if self.ts[g][-1] in self.tEvents[g]:
                x = np.r_[0,x]
                y = np.r_[1,1,y[:-1]]
            else:
                x = np.r_[0,x,self.ts[g][-1]]
                y = np.r_[1,1,y]
            plt.plot(x,y)
        #TODO: set max y above 1
        #TODO: tick marks for censoring
        #TODO: check plotting for multiple censored observations
        #at one time (formula for distance between tick marks?)
        plt.show()
    #TODO: show_results method to display results
    #TODO: Log Rank Test?
    #TODO: show_life_tables method?
