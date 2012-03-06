'''
Glue for returning descriptive statistics.
'''
import numpy as np
from scipy import stats
import os

#############################################
#
#============================================
#       Univariate Descriptive Statistics
#============================================
#

def sign_test(samp,mu0=0):
        '''
        Signs test with mu0=0 by default (though
        the median is often used in practice)

        Parameters
        ----------
        samp

        mu0

        Returns
        ---------
        M, p-value

        where

        M=(N(+) - N(-))/2, N(+) is the number of values above Mu0,
        N(-) is the number of values below.  Values equal to Mu0
        are discarded.

        The p-value for M is calculated using the binomial distrubution
        and can be intrepreted the same as for a t-test.

        See Also
        ---------
        scipy.stats.wilcoxon
        '''
        pos=np.sum(samp>mu0)
        neg=np.sum(samp<mu0)
        M=(pos-neg)/2.
        p=stats.binom_test(min(pos,neg),pos+neg,.5)
        return M, p

def descstats(data, cols=None, axis=0):
    '''
    Prints descriptive statistics for one or multiple variables.

    Parameters
    ------------
    data: numpy array
        `x` is the data

    v: list, optional
        A list of the column number or field names (for a recarray) of variables.
        Default is all columns.

    axis: 1 or 0
        axis order of data.  Default is 0 for column-ordered data.

    Examples
    --------
    >>> descstats(data.exog,v=['x_1','x_2','x_3'])
    '''

    x = np.array(data)  # or rather, the data we're interested in
    if cols is None:
#       if isinstance(x, np.recarray):
#            cols = np.array(len(x.dtype.names))
        if not isinstance(x, np.recarray) and x.ndim == 1:
            x = x[:,None]

    if x.shape[1] == 1:
        desc = '''
    ---------------------------------------------
    Univariate Descriptive Statistics
    ---------------------------------------------

    Var. Name   %(name)12s
    ----------
    Obs.          %(nobs)22i  Range                  %(range)22s
    Sum of Wts.   %(sum)22s  Coeff. of Variation     %(coeffvar)22.4g
    Mode          %(mode)22.4g  Skewness                %(skewness)22.4g
    Repeats       %(nmode)22i  Kurtosis                %(kurtosis)22.4g
    Mean          %(mean)22.4g  Uncorrected SS          %(uss)22.4g
    Median        %(median)22.4g  Corrected SS            %(ss)22.4g
    Variance      %(variance)22.4g  Sum Observations        %(sobs)22.4g
    Std. Dev.     %(stddev)22.4g
    ''' % {'name': cols, 'sum': 'N/A', 'nobs': len(x), 'mode': \
    stats.mode(x)[0][0], 'nmode': stats.mode(x)[1][0], \
    'mean': x.mean(), 'median': np.median(x), 'range': \
    '('+str(x.min())+', '+str(x.max())+')', 'variance': \
    x.var(), 'stddev': x.std(), 'coeffvar': \
    stats.variation(x), 'skewness': stats.skew(x), \
    'kurtosis': stats.kurtosis(x), 'uss': stats.ss(x),\
    'ss': stats.ss(x-x.mean()), 'sobs': np.sum(x)}

#    ''' % {'name': cols[0], 'sum': 'N/A', 'nobs': len(x[cols[0]]), 'mode': \
#    stats.mode(x[cols[0]])[0][0], 'nmode': stats.mode(x[cols[0]])[1][0], \
#    'mean': x[cols[0]].mean(), 'median': np.median(x[cols[0]]), 'range': \
#    '('+str(x[cols[0]].min())+', '+str(x[cols[0]].max())+')', 'variance': \
#    x[cols[0]].var(), 'stddev': x[cols[0]].std(), 'coeffvar': \
#    stats.variation(x[cols[0]]), 'skewness': stats.skew(x[cols[0]]), \
#    'kurtosis': stats.kurtosis(x[cols[0]]), 'uss': stats.ss(x[cols[0]]),\
#    'ss': stats.ss(x[cols[0]]-x[cols[0]].mean()), 'sobs': np.sum(x[cols[0]])}

        desc+= '''

    Percentiles
    -------------
    1  %%          %12.4g
    5  %%          %12.4g
    10 %%          %12.4g
    25 %%          %12.4g

    50 %%          %12.4g

    75 %%          %12.4g
    90 %%          %12.4g
    95 %%          %12.4g
    99 %%          %12.4g
    ''' % tuple([stats.scoreatpercentile(x,per) for per in (1,5,10,25,
                50,75,90,95,99)])
        t,p_t=stats.ttest_1samp(x,0)
        M,p_M=sign_test(x)
        S,p_S=stats.wilcoxon(np.squeeze(x))

        desc+= '''

    Tests of Location (H0: Mu0=0)
    -----------------------------
    Test                Statistic       Two-tailed probability
    -----------------+-----------------------------------------
    Student's t      |  t %7.5f   Pr > |t|   <%.4f
    Sign             |  M %8.2f   Pr >= |M|  <%.4f
    Signed Rank      |  S %8.2f   Pr >= |S|  <%.4f

    ''' % (t,p_t,M,p_M,S,p_S)
# Should this be part of a 'descstats'
# in any event these should be split up, so that they can be called
# individually and only returned together if someone calls summary
# or something of the sort

    elif x.shape[1] > 1:
        desc ='''
    Var. Name   |     Obs.        Mean    Std. Dev.           Range
    ------------+--------------------------------------------------------'''+\
            os.linesep

# for recarrays with columns passed as names
#        if isinstance(cols[0],str):
#            for var in cols:
#                desc += "%(name)15s %(obs)9i %(mean)12.4g %(stddev)12.4g \
#%(range)20s" %  {'name': var, 'obs': len(x[var]), 'mean': x[var].mean(),
#        'stddev': x[var].std(), 'range': '('+str(x[var].min())+', '\
#                +str(x[var].max())+')'+os.linesep}
#        else:
        for var in range(x.shape[1]):
                desc += "%(name)15s %(obs)9i %(mean)12.4g %(stddev)12.4g \
%(range)20s" % {'name': var, 'obs': len(x[:,var]), 'mean': x[:,var].mean(),
                'stddev': x[:,var].std(), 'range': '('+str(x[:,var].min())+', '+\
                str(x[:,var].max())+')'+os.linesep}
    else:
        raise ValueError, "data not understood"

    return desc

#if __name__=='__main__':
# test descstats
#    import os
#    loc='http://eagle1.american.edu/~js2796a/data/handguns_data.csv'
#    relpath=(load_dataset(loc))
#    dta=np.recfromcsv(relpath)
#    descstats(dta,['stpop'])
#    raw_input('Hit enter for multivariate test')
#    descstats(dta,['stpop','avginc','vio'])

# with plain arrays
#    import string2dummy as s2d
#    dts=s2d.string2dummy(dta)
#    ndts=np.vstack(dts[col] for col in dts.dtype.names)
# observations in columns and data in rows
# is easier for the call to stats

# what to make of
# ndts=np.column_stack(dts[col] for col in dts.dtype.names)
# ntda=ntds.swapaxis(1,0)
# ntda is ntds returns false?

# or now we just have detailed information about the different strings
# would this approach ever be inappropriate for a string typed variable
# other than dates?
#    descstats(ndts, [1])
#    raw_input("Enter to try second part")
#    descstats(ndts, [1,20,3])

if __name__ == '__main__':
    import statsmodels.api as sm
    import os
    data = sm.datasets.longley.load()
    data.exog = sm.add_constant(data.exog)
    sum1 = descstats(data.exog)
    sum1a = descstats(data.exog[:,:1])

#    loc='http://eagle1.american.edu/~js2796a/data/handguns_data.csv'
#    dta=np.recfromcsv(loc)
#    summary2 = descstats(dta,['stpop'])
#    summary3 =  descstats(dta,['stpop','avginc','vio'])
#TODO: needs a by argument
#    summary4 = descstats(dta) this fails
# this is a bug
# p = dta[['stpop']]
# p.view(dtype = np.float, type = np.ndarray)
# this works
# p.view(dtype = np.int, type = np.ndarray)

### This is *really* slow ###
    if os.path.isfile('./Econ724_PS_I_Data.csv'):
        data2 = np.recfromcsv('./Econ724_PS_I_Data.csv')
        sum2 = descstats(data2.ahe)
        sum3 = descstats(np.column_stack((data2.ahe,data2.yrseduc)))
        sum4 = descstats(np.column_stack(([data2[_] for \
                _ in data2.dtype.names])))


