'''Helper class for Monte Carlo Studies for (currently) statistical tests

Most of it should also be usable for Bootstrap, and for MC for estimators.
Takes the sample generator, dgb, and the statistical results, statistic,
as functions in the argument.


Author: Josef Perktold (josef-pktd)
License: BSD-3

'''


import numpy as np

from scikits.statsmodels.iolib.table import SimpleTable

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

        Parameters
        ----------
        nrepl : int
            number of Monte Carlo repetitions
        statindices : None or list of integers
           determines which values of the return of the statistic
           functions are stored in the Monte Carlo. Default None
           means the entire return. If statindices is a list of
           integers, then it will be used as index into the return.
        dgpargs : tuple
           optional parameters for the DGP
        statsargs : tuple
           optional parameters for the statistics function

        Returns
        -------
        None, all results are attached


        '''
        self.nrepl = nrepl
        self.statindices = statindices
        self.dgpargs = dgpargs
        self.statsargs = statsargs

        dgp = self.dgp
        statfun = self.statistic # name ?
        #introspect len of return of statfun,
        #possible problems with ndim>1, check ValueError
        mcres0 = statfun(dgp(*dgpargs), *statsargs)
        self.nreturn = nreturns = len(np.ravel(mcres0))

        #single return statistic
        if statindices is None:
            #self.nreturn = nreturns = 1
            mcres = np.zeros(nrepl)
            mcres[0] = mcres0
            for ii in range(1, repl-1, nreturns):
                x = dgp(*dgpargs) #(1e-4+np.random.randn(nobs)).cumsum()
                #should I ravel?
                mcres[ii] = statfun(x, *statsargs) #unitroot_adf(x, 2,trendorder=0, autolag=None)
        #more than one return statistic
        else:
            self.nreturn = nreturns = len(statindices)
            self.mcres = mcres = np.zeros((nrepl, nreturns))
            mcres[0] = [mcres0[i] for i in statindices]
            for ii in range(1, nrepl-1):
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

    #use cache decoratot
    def get_mc_sorted(self):
        if not hasattr(self, 'mcressort'):
            self.mcressort = np.sort(self.mcres, axis=0)
        return self.mcressort


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

        mc_sorted = self.get_mc_sorted()[:,idx]
        return frac, mc_sorted[(self.nrepl*frac).astype(int)]

    def cdf(self, x, idx=None):
        '''calculate quantiles of Monte Carlo results


        changes:
        does all sort at once, but reports only one at a time

        '''
        idx = np.atleast_1d(idx).tolist()  #assure iterable, use list ?

#        if self.mcres.ndim == 2:
#            if not idx is None:
#                mcres = self.mcres[:,idx]
#            else:
#                raise ValueError('currently only 1 statistic at a time')
#        else:
#            mcres = self.mcres

        mc_sorted = self.get_mc_sorted()

        x = np.asarray(x)
        #TODO:autodetect or explicit option ?
        if x.ndim > 1 and x.shape[1]==len(idx):
            use_xi = True
        else:
            use_xi = False

        x_ = x  #alias
        probs = []
        for i,ix in enumerate(idx):
            if use_xi:
                x_ = x[:,i]
            probs.append(np.searchsorted(mc_sorted[:,ix], x_)/float(self.nrepl))
        probs = np.asarray(probs).T
        return x, probs

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

    def summary_quantiles(self, idx, distppf, frac=[0.01, 0.025, 0.05, 0.1, 0.975],
                          varnames=None, title=None):
        '''summary table for quantiles

        '''
        idx = np.atleast_1d(idx)  #assure iterable, use list ?

        quant, mcq = self.quantiles(idx, frac=frac)
        #not sure whether this will work with single quantile
        #crit = stats.chi2([2,4]).ppf(np.atleast_2d(quant).T)
        crit = distppf(np.atleast_2d(quant).T)
        mml=[]
        for i, ix in enumerate(idx):  #TODO: hardcoded 2 ?
            mml.extend([mcq[:,i], crit[:,i]])
        #mmlar = np.column_stack(mml)
        mmlar = np.column_stack([quant] + mml)
        #print mmlar.shape
        if title:
            title = title +' Quantiles (critical values)'
        else:
            title='Quantiles (critical values)'
        #TODO use stub instead
        if varnames is None:
            varnames = ['var%d' % i for i in range(mmlar.shape[1]//2)]
        headers = ['\nprob'] + ['%s\n%s' % (i, t) for i in varnames for t in ['mc', 'dist']]
        return SimpleTable(mmlar,
                          txt_fmt={'data_fmts': ["%#6.3f"]+["%#10.4f"]*(mmlar.shape[1]-1)},
                          title=title,
                          headers=headers)

    def summary_cdf(self, idx, frac, crit, varnames=None, title=None):
        '''summary table for quantiles

        currently just a partial copy from python session, for ljung-box example

        add also
        >>> lb_dist.ppf([0.01, 0.025, 0.05, 0.1, 0.975])
        array([  0.29710948,   0.48441856,   0.71072302,   1.06362322,  11.14328678])
        >>> stats.kstest(mc1.mcres[:,3], stats.chi2(4).cdf)
        (0.052009265258216836, 0.0086211970272969118)
        '''
        idx = np.atleast_1d(idx)  #assure iterable, use list ?


        mml=[]
        #TODO:need broadcasting in cdf
        for i in range(len(idx)):
            #print i, mc1.cdf(crit[:,i], [idx[i]])[1].ravel()
            mml.append(self.cdf(crit[:,i], [idx[i]])[1].ravel())
        #mml = self.cdf(crit, idx)[1]
        #mmlar = np.column_stack(mml)
        #print mml[0].shape, np.shape(frac)
        mmlar = np.column_stack([frac] + mml)
        #print mmlar.shape
        if title:
            title = title +' Probabilites'
        else:
            title='Probabilities'
        #TODO use stub instead
        #headers = ['\nprob'] + ['var%d\n%s' % (i, t) for i in range(mmlar.shape[1]-1) for t in ['mc']]

        if varnames is None:
            varnames = ['var%d' % i for i in range(mmlar.shape[1]-1)]
        headers = ['prob'] + varnames
        return SimpleTable(mmlar,
                          txt_fmt={'data_fmts': ["%#6.3f"]+["%#10.4f"]*(np.array(mml).shape[1]-1)},
                          title=title,
                          headers=headers)









if __name__ == '__main__':
    from scipy import stats
    from scikits.statsmodels.iolib.table import SimpleTable

    from scikits.statsmodels.sandbox.stats.diagnostic import (
                    acorr_ljungbox, unitroot_adf)


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
    mc1.run(10000, statindices=range(8))
    print mc1.histogram(1, critval=[0.01, 0.025, 0.05, 0.1, 0.975])
    print mc1.quantiles(1)
    print mc1.quantiles(0)
    print mc1.histogram(0)

    #print mc1.summary_quantiles([1], stats.chi2([2]).ppf, title='acorr_ljungbox')
    print mc1.summary_quantiles([1,2,3], stats.chi2([2,3,4]).ppf,
                                varnames=['lag 1', 'lag 2', 'lag 3'],
                                title='acorr_ljungbox')
    print mc1.cdf(0.1026, 1)
    print mc1.cdf(0.7278, 3)

    print mc1.cdf(0.7278, [1,2,3])
    frac = [0.01, 0.025, 0.05, 0.1, 0.975]
    crit = stats.chi2([2,4]).ppf(np.atleast_2d(frac).T)
    print mc1.summary_cdf([1,3], frac, crit, title='acorr_ljungbox')
    crit = stats.chi2([2,3,4]).ppf(np.atleast_2d(frac).T)
    print mc1.summary_cdf([1,2,3], frac, crit,
                          varnames=['lag 1', 'lag 2', 'lag 3'],
                          title='acorr_ljungbox')

    print mc1.cdf(crit, [1,2,3])[1].shape

    #fixed broadcasting in cdf  Done 2d only
    '''
    >>> mc1.cdf(crit[:,0], [1])[1].shape
    (5, 1)
    >>> mc1.cdf(crit[:,0], [1,3])[1].shape
    (5, 2)
    >>> mc1.cdf(crit[:,:], [1,3])[1].shape
    (2, 5, 2)
    '''

    doplot=0
    if doplot:
        import matplotlib.pyplot as plt
        mc1.plot_hist(0,stats.chi2(2).pdf)  #which pdf
        plt.show()
