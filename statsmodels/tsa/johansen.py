'''
function result = johansen(x,p,k)
% PURPOSE: perform Johansen cointegration tests
% -------------------------------------------------------
% USAGE: result = johansen(x,p,k)
% where:      x = input matrix of time-series in levels, (nobs x m)
%             p = order of time polynomial in the null-hypothesis
%                 p = -1, no deterministic part
%                 p =  0, for constant term
%                 p =  1, for constant plus time-trend
%                 p >  1, for higher order polynomial
%             k = number of lagged difference terms used when
%                 computing the estimator
% -------------------------------------------------------
% RETURNS: a results structure:
%          result.eig  = eigenvalues  (m x 1)
%          result.evec = eigenvectors (m x m), where first
%                        r columns are normalized coint vectors
%          result.lr1  = likelihood ratio trace statistic for r=0 to m-1
%                        (m x 1) vector
%          result.lr2  = maximum eigenvalue statistic for r=0 to m-1
%                        (m x 1) vector
%          result.cvt  = critical values for trace statistic
%                        (m x 3) vector [90% 95% 99%]
%          result.cvm  = critical values for max eigen value statistic
%                        (m x 3) vector [90% 95% 99%]
%          result.ind  = index of co-integrating variables ordered by
%                        size of the eigenvalues from large to small
% -------------------------------------------------------
% NOTE: c_sja(), c_sjt() provide critical values generated using
%       a method of MacKinnon (1994, 1996).
%       critical values are available for n<=12 and -1 <= p <= 1,
%       zeros are returned for other cases.
% -------------------------------------------------------
% SEE ALSO: prt_coint, a function that prints results
% -------------------------------------------------------
% References: Johansen (1988), 'Statistical Analysis of Co-integration
% vectors', Journal of Economic Dynamics and Control, 12, pp. 231-254.
% MacKinnon, Haug, Michelis (1996) 'Numerical distribution
% functions of likelihood ratio tests for cointegration',
% Queen's University Institute for Economic Research Discussion paper.
% (see also: MacKinnon's JBES 1994 article
% -------------------------------------------------------

% written by:
% James P. LeSage, Dept of Economics
% University of Toledo
% 2801 W. Bancroft St,
% Toledo, OH 43606
% jlesage@spatial-econometrics.com

% ****************************************************************
% NOTE: Adina Enache provided some bug fixes and corrections that
%       she notes below in comments. 4/10/2000
% ****************************************************************
'''

import numpy as np
from numpy import zeros, ones, flipud, log
from numpy.linalg import inv, eig, cholesky as chol
from statsmodels.regression.linear_model import OLS

from .coint_tables import c_sja, c_sjt

tdiff = np.diff

class Holder(object):
    pass

def rows(x):
    return x.shape[0]

def trimr(x, front, end):
    if end > 0:
        return x[front:-end]
    else:
        return x[front:]

import statsmodels.tsa.tsatools as tsat
mlag = tsat.lagmat

def mlag_(x, maxlag):
    '''return all lags up to maxlag
    '''
    return x[:-lag]

def lag(x, lag):
    return x[:-lag]

def detrend(y, order):
    if order == -1:
        return y
    return OLS(y, np.vander(np.linspace(-1,1,len(y)), order+1)).fit().resid

def resid(y, x):
    if x.size == 0:
        return y
    r = y - np.dot(x, np.dot(np.linalg.pinv(x), y))
    return r




def coint_johansen(x, p, k, coint_trend=None):

    #    % error checking on inputs
    #    if (nargin ~= 3)
    #     error('Wrong # of inputs to johansen')
    #    end
    nobs, m = x.shape

    #why this?  f is detrend transformed series, p is detrend data
    if (p > -1):
        f = 0
    else:
        f = p

    if coint_trend is not None:
        f = coint_trend  #matlab has separate options

    x     = detrend(x,p)
    dx    = tdiff(x,1, axis=0)
    #dx    = trimr(dx,1,0)
    z     = mlag(dx,k)#[k-1:]
    print(z.shape)
    z = trimr(z,k,0)
    z     = detrend(z,f)
    print(dx.shape)
    dx = trimr(dx,k,0)

    dx    = detrend(dx,f)
    #r0t   = dx - z*(z\dx)
    r0t   = resid(dx, z)  #diff on lagged diffs
    #lx = trimr(lag(x,k),k,0)
    lx = lag(x,k)
    lx = trimr(lx, 1, 0)
    dx    = detrend(lx,f)
    print('rkt', dx.shape, z.shape)
    #rkt   = dx - z*(z\dx)
    rkt   = resid(dx, z)  #level on lagged diffs
    skk   = np.dot(rkt.T, rkt) / rows(rkt)
    sk0   = np.dot(rkt.T, r0t) / rows(rkt)
    s00   = np.dot(r0t.T, r0t) / rows(r0t)
    sig   = np.dot(sk0, np.dot(inv(s00), (sk0.T)))
    tmp   = inv(skk)
    #du, au = eig(np.dot(tmp, sig))
    au, du = eig(np.dot(tmp, sig))  #au is eval, du is evec
    #orig = np.dot(tmp, sig)

    #% Normalize the eigen vectors such that (du'skk*du) = I
    temp   = inv(chol(np.dot(du.T, np.dot(skk, du))))
    dt     = np.dot(du, temp)


    #JP: the next part can be done much  easier

    #%      NOTE: At this point, the eigenvectors are aligned by column. To
    #%            physically move the column elements using the MATLAB sort,
    #%            take the transpose to put the eigenvectors across the row

    #dt = transpose(dt)

    #% sort eigenvalues and vectors

    #au, auind = np.sort(diag(au))
    auind = np.argsort(au)
    #a = flipud(au)
    aind = flipud(auind)
    a = au[aind]
    #d = dt[aind,:]
    d = dt[:,aind]

    #%NOTE: The eigenvectors have been sorted by row based on auind and moved to array "d".
    #%      Put the eigenvectors back in column format after the sort by taking the
    #%      transpose of "d". Since the eigenvectors have been physically moved, there is
    #%      no need for aind at all. To preserve existing programming, aind is reset back to
    #%      1, 2, 3, ....

    #d  =  transpose(d)
    #test = np.dot(transpose(d), np.dot(skk, d))

    #%EXPLANATION:  The MATLAB sort function sorts from low to high. The flip realigns
    #%auind to go from the largest to the smallest eigenvalue (now aind). The original procedure
    #%physically moved the rows of dt (to d) based on the alignment in aind and then used
    #%aind as a column index to address the eigenvectors from high to low. This is a double
    #%sort. If you wanted to extract the eigenvector corresponding to the largest eigenvalue by,
    #%using aind as a reference, you would get the correct eigenvector, but with sorted
    #%coefficients and, therefore, any follow-on calculation would seem to be in error.
    #%If alternative programming methods are used to evaluate the eigenvalues, e.g. Frame method
    #%followed by a root extraction on the characteristic equation, then the roots can be
    #%quickly sorted. One by one, the corresponding eigenvectors can be generated. The resultant
    #%array can be operated on using the Cholesky transformation, which enables a unit
    #%diagonalization of skk. But nowhere along the way are the coefficients within the
    #%eigenvector array ever changed. The final value of the "beta" array using either method
    #%should be the same.


    #% Compute the trace and max eigenvalue statistics */
    lr1 = zeros(m)
    lr2 = zeros(m)
    cvm = zeros((m,3))
    cvt = zeros((m,3))
    iota = ones(m)
    t, junk = rkt.shape
    for i in range(0, m):
        tmp = trimr(log(iota-a), i ,0)
        lr1[i] = -t * np.sum(tmp, 0)  #columnsum ?
        #tmp = np.log(1-a)
        #lr1[i] = -t * np.sum(tmp[i:])
        lr2[i] = -t * log(1-a[i])
        cvm[i,:] = c_sja(m-i,p)
        cvt[i,:] = c_sjt(m-i,p)
        aind[i]  = i
    #end

    result = Holder()
    #% set up results structure
    #estimation results, residuals
    result.rkt = rkt
    result.r0t = r0t
    result.eig = a
    result.evec = d  #transposed compared to matlab ?
    result.lr1 = lr1
    result.lr2 = lr2
    result.cvt = cvt
    result.cvm = cvm
    result.ind = aind
    result.meth = 'johansen'

    return result
