

import numpy as np



def lagmat(x, maxlag, trim='forward'):
    '''create 2d array of lags

    Parameters
    ----------
    x : array_like, 1d or 2d
        data; if 2d, observation in rows and variables in columns
    maxlag : int
        all lags from zero to maxlag are included
    trim : string
        * 'forward' : trim invalid observations in front
        * 'backward' : trim invalid initial observations
        * 'both' : trim invalid observations on both sides
        * 'none' : no trimming of observations

    Returns
    -------
    lagmat : 2d array
        array with lagged observations

    Notes
    -----
    TODO:
    * allow list of lags additional to maxlag
    * create varnames for columns
    '''
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:,None]
    nobs, nvar = x.shape
    lm = np.zeros((nobs+maxlag, nvar*(maxlag+1)))
    for k in range(0, maxlag+1):
        #print k, maxlag-k,nobs-k, nvar*k,nvar*(k+1), x.shape, lm.shape
        lm[maxlag-k:nobs+maxlag-k, nvar*(maxlag-k):nvar*(maxlag-k+1)] = x
    trimlower = trim.lower()
    if trimlower == 'none':
        return lm
    elif trimlower == 'forward':
        return lm[:nobs+maxlag-k,:]
    elif trimlower == 'both':
        return lm[maxlag:nobs+maxlag-k,:]
    elif trimlower == 'backward':
        return lm[maxlag:,:]
    else:
        raise ValueError('trim option not valid')

def lagmat2ds(x, maxlag0, maxlagex=None, dropex=0, trim='forward'):
    '''generate lagmatrix for 2d array, columns arranged by variables

    Parameters
    ----------
    x : array_like, 2d
        2d data, observation in rows and variables in columns
    maxlag0 : int
        for first variable all lags from zero to maxlag are included
    maxlagex : None or int
        max lag for all other variables all lags from zero to maxlag are included
    dropex : int (default is 0)
        exclude first dropex lags from other variables
        for all variables, except the first, lags from dropex to maxlagex are included
    trim : string
        * 'forward' : trim invalid observations in front
        * 'backward' : trim invalid initial observations
        * 'both' : trim invalid observations on both sides
        * 'none' : no trimming of observations

    Returns
    -------
    lagmat : 2d array
        array with lagged observations, columns ordered by variable

    Notes
    -----
    very inefficient for unequal lags, just done for convenience
    '''
    if maxlagex is None:
        maxlagex = maxlag0
    maxlag = max(maxlag0, maxlagex)
    nobs, nvar = x.shape
    lagsli = [lagmat(x[:,0], maxlag, trim=trim)[:,:maxlag0]]
    for k in range(1,nvar):
        lagsli.append(lagmat(x[:,k], maxlag, trim=trim)[:,dropex:maxlagex])
    return np.column_stack(lagsli)


def grangercausalitytests(x, maxlag):
    '''four tests for granger causality of 2 timeseries

    this is a proof-of concept implementation
    not cleaned up, has some duplicate calculations,
    memory intensive - builds full lag array for variables
    prints results
    not verified with other packages,
    all four tests give similar results (1 and 4 identical)

    Parameters
    ----------
    x : array, 2d, (nobs,2)
        data for test whether the time series in the second column Granger
        causes the time series in the first column
    maxlag : integer
        the Granger causality test results are calculated for all lags up to
        maxlag

    Returns
    -------
    None : no returns
        all test results are currently printed

    Notes
    -----
    TODO: convert to function that returns and compare with other packages

    '''
    from scipy import stats # lazy import
    import scikits.statsmodels as sm  # absolute import for now

    for mlg in range(1, maxlag+1):
        print '\nGranger Causality'
        print 'number of lags (no zero)', mlg
        mxlg = mlg + 1 # Note number of lags starting at zero in lagmat

        # create lagmat of both time series
        dta = lagmat2ds(x, mxlg, trim='both', dropex=1)

        #add constant
        dtaown = sm.add_constant(dta[:,1:mxlg])
        dtajoint = sm.add_constant(dta[:,1:])

        #run ols on both models without and with lags of second variable
        res2down = sm.OLS(dta[:,0], dtaown).fit()
        res2djoint = sm.OLS(dta[:,0], dtajoint).fit()

        #print results
        #for ssr based tests see: http://support.sas.com/rnd/app/examples/ets/granger/index.htm
        #the other tests are made-up

        # Granger Causality test using ssr (F statistic)
        fgc1 = (res2down.ssr-res2djoint.ssr)/res2djoint.ssr/(mxlg-1)*res2djoint.df_resid
        print 'ssr based F test:         F=%-8.4f, p=%-8.4f, df_denom=%d, df_num=%d' % \
              (fgc1, stats.f.sf(fgc1, mxlg-1, res2djoint.df_resid), res2djoint.df_resid, mxlg-1)

        # Granger Causality test using ssr (ch2 statistic)
        fgc2 = res2down.nobs*(res2down.ssr-res2djoint.ssr)/res2djoint.ssr
        print 'ssr based chi2 test:   chi2=%-8.4f, p=%-8.4f, df=%d' %  \
              (fgc2, stats.chi2.sf(fgc2, mxlg-1), mxlg-1)

        #likelihood ratio test pvalue:
        lr = -2*(res2down.llf-res2djoint.llf)
        print 'likelihood ratio test: chi2=%-8.4f, p=%-8.4f, df=%d' %  \
              (lr, stats.chi2.sf(lr, mxlg-1), mxlg-1)

        # F test that all lag coefficients of exog are zero
        rconstr = np.column_stack((np.zeros((mxlg-1,mxlg-1)), np.eye(mxlg-1, mxlg-1),\
                                   np.zeros((mxlg-1, 1))))
        ftres = res2djoint.f_test(rconstr)
        print 'parameter F test:         F=%-8.4f, p=%-8.4f, df_denom=%d, df_num=%d' % \
              (ftres.fvalue, ftres.pvalue, ftres.df_denom, ftres.df_num)

__all__ = ['lagmat', 'lagmat2ds', 'grangercausalitytests']

if __name__ == '__main__':
    # sanity check, mainly for imports
    x = np.random.normal(size=(100,2))
    tmp = lagmat(x,2)
    tmp = lagmat2ds(x,2)
    grangercausalitytests(x, 2)
