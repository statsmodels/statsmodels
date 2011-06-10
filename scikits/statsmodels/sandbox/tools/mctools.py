'''Helper class for Monte Carlo Studies for (currently) statistical tests

Most of it should also be usable for Bootstrap, and for MC for estimators.
Takes the sample generator, dgb, and the statistical results, statistic,
as functions in the argument.


Author: Josef Perktold (josef-pktd)

'''


import numpy as np

#copied from stattools
class StatTestMC(object):
    """class to run Monte Carlo study on a statistical test'''

    TODO
    print summary, for quantiles and for histogram
    draft in trying out script log

    """

    def __init__(self, dgp, statistic):
        self.dgp = dgp #staticmethod(dgp)  #no self
        self.statistic = statistic # staticmethod(statistic)  #no self

    def run(self, nrepl, statindices=None, dgpargs=[], statsargs=[]):
        '''run the actual Monte Carlo and save results


        '''
        self.nrepl = nrepl
        self.statindices = statindices
        self.dgpargs = dgpargs
        self.statsargs = statsargs

        dgp = self.dgp
        statfun = self.statistic # name ?

        #single return statistic   #TODO: introspect len of return of statfun
        if statindices is None:
            self.nreturn = nreturns = 1
            mcres = np.zeros(nrepl)
            for ii in range(nrepl-1):
                x = dgp(*dgpargs) #(1e-4+np.random.randn(nobs)).cumsum()
                mcres[ii] = statfun(x, *statsargs) #unitroot_adf(x, 2,trendorder=0, autolag=None)
        #more than one return statistic
        else:
            self.nreturn = nreturns = len(statindices)
            self.mcres = mcres = np.zeros((nrepl, nreturns))
            for ii in range(nrepl-1):
                x = dgp(*dgpargs) #(1e-4+np.random.randn(nobs)).cumsum()
                ret = statfun(x, *statsargs)
                mcres[ii] = [ret[i] for i in statindices]

        self.mcres = mcres

    def histogram(self, idx=None, critval=None):
        '''calculate histogram values

        does not do any plotting
        '''
        if self.mcres.ndim == 2:
            if  not idx is None:
                mcres = self.mcres[:,idx]
            else:
                raise ValueError('currently only 1 statistic at a time')
        else:
            mcres = self.mcres

        if critval is None:
            histo = np.histogram(mcres, bins=10)
        else:
            if not critval[0] == -np.inf:
                bins=np.r_[-np.inf, critval, np.inf]
            if not critval[0] == -np.inf:
                bins=np.r_[bins, np.inf]
            histo = np.histogram(mcres,
                                 bins=np.r_[-np.inf, critval, np.inf])

        self.histo = histo
        self.cumhisto = np.cumsum(histo[0])*1./self.nrepl
        self.cumhistoreversed = np.cumsum(histo[0][::-1])[::-1]*1./self.nrepl
        return histo, self.cumhisto, self.cumhistoreversed

    def quantiles(self, idx=None, frac=[0.01, 0.025, 0.05, 0.1, 0.975]):
        '''calculate quantiles of Monte Carlo results


        changes:
        does all sort at once, but reports only one at a time

        '''

        if self.mcres.ndim == 2:
            if not idx is None:
                mcres = self.mcres[:,idx]
            else:
                raise ValueError('currently only 1 statistic at a time')
        else:
            mcres = self.mcres

        self.frac = frac = np.asarray(frac)

        if not hasattr(self, 'mcressort'):
            self.mcressort = np.sort(self.mcres, axis=0)

        mcressort = self.mcressort[:,idx]
        return frac, mcressort[(self.nrepl*frac).astype(int)]

    def plot_hist(self, idx, distpdf, bins=50, ax=None):
        if self.mcres.ndim == 2:
            if not idx is None:
                mcres = self.mcres[:,idx]
            else:
                raise ValueError('currently only 1 statistic at a time')
        else:
            mcres = self.mcres

        lsp = np.linspace(mcres.min(), mcres.max(), 100)


        import matplotlib.pyplot as plt
        #I don't want to figure this out now
#        if ax=None:
#            fig = plt.figure()
#            ax = fig.addaxis()
        fig = plt.figure()
        plt.hist(mcres, bins=bins, normed=True)
        plt.plot(lsp, distpdf(lsp), 'r')

    def summary_quantiles(self, idx, distpdf, bins=50, ax=None):
        '''summary table for quantiles

        currently just a partial copy from python session, for ljung-box example

        add also
        >>> lb_dist.ppf([0.01, 0.025, 0.05, 0.1, 0.975])
        array([  0.29710948,   0.48441856,   0.71072302,   1.06362322,  11.14328678])
        >>> stats.kstest(mc1.mcres[:,3], stats.chi2(4).cdf)
        (0.052009265258216836, 0.0086211970272969118)
        '''
        mcq = self.quantiles([1,3])[1]
        perc = stats.chi2([2,4]).ppf(np.array([[0.01, 0.025, 0.05, 0.1, 0.975]]).T)
        mml=[]
        for i in range(2):
            mml.extend([mcq[:,i],perc[:,i]])
        print SimpleTable(np.column_stack(mml),txt_fmt={'data_fmts': ["%#6.3f"]+["%#10.4f"]*(mm.shape[1]-1)},headers=['quantile']+['mc','dist']*2)









if __name__ == '__main__':
    def randwalksim(nobs=100, drift=0.0):
        return (drift+np.random.randn(nobs)).cumsum()

    def normalnoisesim(nobs=500, loc=0.0):
        return (loc+np.random.randn(nobs))

    def adf20(x):
        return unitroot_adf(x, 2,trendorder=0, autolag=None)

#    print '\nResults with MC class'
#    mc1 = StatTestMC(randwalksim, adf20)
#    mc1.run(1000)
#    print mc1.histogram(critval=[-3.5, -3.17, -2.9 , -2.58,  0.26])
#    print mc1.quantiles()

    print '\nLjung Box'
    from scikits.statsmodels.sandbox.stats.diagnostic import acorr_ljungbox

    def lb4(x):
        s,p = acorr_ljungbox(x, lags=4)
        return s[-1], p[-1]

    def lb1(x):
        s,p = acorr_ljungbox(x, lags=1)
        return s[0], p[0]

    def lb(x):
        s,p = acorr_ljungbox(x, lags=4)
        return np.r_[s, p]

    print 'Results with MC class'
    mc1 = StatTestMC(normalnoisesim, lb)
    mc1.run(1000, statindices=range(8))
    print mc1.histogram(1, critval=[0.01, 0.025, 0.05, 0.1, 0.975])
    print mc1.quantiles(1)
    print mc1.quantiles(0)
    print mc1.histogram(0)
