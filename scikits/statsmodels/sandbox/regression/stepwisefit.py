#!/usr/bin/python

"""Stepwise Regression

Author: Collin RM Stocks

References:
[1] Draper, N. R., and H. Smith. Applied Regression Analysis.
    Hoboken, NJ: Wiley-Interscience, 1998. pp. 307–312.
[2] The MatLab implementation of the same function.
    http://www.mathworks.com/help/toolbox/stats/stepwisefit.html.
"""

__all__ = ["stepwisefit"]

import numpy
import numpy as np
import scipy
import scipy.stats
from scipy import linalg
from scipy.linalg import qr

class InfoDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def stepnext(inmodel, pval, b, penter, premove, keep):
    """
    Figure out next step.
    """

    swap = -1
    p = np.NaN

    # Look for terms out that should be in.
    termsout = (~inmodel & ~keep).nonzero()[0]
    if termsout.size != 0:
        kmin = pval[termsout].argmin()
        pmin = pval[termsout][kmin]
        if pmin < penter:
            swap = termsout[kmin]
            p = pmin

    # Otherwise look for terms that should be out.
    if swap == -1:
        termsin = (inmodel & ~keep).nonzero()[0]
        if termsin.size != 0:
            badterms = termsin[np.isnan(pval.ravel()[termsin])]
            if badterms.size != 0:
                # Apparently we have a perfect fit but it is also
                # overdetermined. Terms with NaN coefficients may as well
                # be removed.
                swap = np.isnan(b[badterms])
                if swap.any():
                    swap = badterms[swap]
                    swap = swap[0]
                else:
                    # If there are many terms contributing to a perfect fit, we
                    # may as well remove the term that contributes the least.
                    # For convenience we'll pick the one with the smallest
                    # coeff.
                    swap = badterms[abs(b[badterms]).argmin()]
                p = np.NaN
            else:
                kmax = pval[termsin].argmax()
                pmax = pval[termsin][kmax]
                if pmax > premove:
                    if not np.isscalar(kmax):
                        kmax = kmax.ravel()[0]
                    swap = termsin[kmax]
                    p = pmax
    return swap, p

def stepcalc(allx, y, inmodel):
    """
    Perform fit and other calculations as part of stepwise regression.
    """

    N = y.size # Number of independent tests (rows in allx).
    P = inmodel.size # Number of independent variables in each test
                     # (cols in allx).
    X = np.concatenate((np.ones((N, 1)), allx[:, inmodel]), 1)
    nin = inmodel.sum() + 1
    tol = max(N, P + 1) * np.finfo(allx.dtype).eps
    x = allx[:, ~inmodel]
    sumxsq = (x ** 2).sum(axis = 0)

    # Compute b and its standard error.
    Q, R, perm = qr(X, mode = "economic", pivoting = True)
    Rrank = (abs(np.diag(R)) > tol * abs(R.ravel()[0])).sum()
    if Rrank < nin:
        R = R[0:Rrank, 0:Rrank]
        Q = Q[:, 0:Rrank]
        perm = perm[0:Rrank]

    # Compute the LS coefficients, filling in zeros in elements corresponding
    # to rows of X that were thrown out.
    b = np.zeros((nin, 1))
    Qb = np.dot(Q.conj().T, y)
    Qb[abs(Qb) < tol * max(abs(Qb))] = 0
    b[perm] = linalg.solve(R, Qb)

    r = y - np.dot(X, b)
    dfe = X.shape[0] - Rrank
    df0 = Rrank - 1
    SStotal = linalg.norm(y - y.mean())
    SStotal = np.dot(SStotal, SStotal)
    SSresid = linalg.norm(r)
    SSresid = np.dot(SSresid, SSresid)
    perfectyfit = (dfe == 0) or (SSresid < tol * SStotal)
    if perfectyfit:
        SSresid = 0
        r[:] = 0
    rmse = np.sqrt(np.divide(SSresid, dfe))
    Rinv = linalg.solve(R, np.eye(max(R.shape))[0:R.shape[0], 0:R.shape[1]])
    se = np.zeros((nin, 1))
    se[perm] = rmse * np.expand_dims(np.sqrt((Rinv ** 2).sum(axis = 1)), 1)

    # Compute separate added-variable coeffs and their standard errors.
    xr = x - np.dot(Q, np.dot(Q.conj().T, x))
        # remove effect of "in" predictors on "out" predictors
    yr = r
        # remove effect of "in" predictors on response

    xx = (xr ** 2).sum(axis = 0)

    perfectxfit = (xx <= tol * sumxsq)
    if perfectxfit.any(): # to coef==0 for columns dependent in "in" cols
        xr[:, perfectxfit] = 0
        xx[perfectxfit] = 1
    b2 = np.divide(np.dot(yr.conj().T, xr), xx)
    r2 = np.tile(yr, (1, (~inmodel).sum())) - xr * np.tile(b2, (N, 1))
    df2 = max(0, dfe - 1)
    s2 = np.divide(np.sqrt(np.divide((r2 ** 2).sum(axis = 0), df2)),
        np.sqrt(xx))
    if len(s2.shape) == 1:
        s2 = s2.reshape((1, s2.shape[0]))

    # Combine in/out coefficients and standard errors.
    B = np.zeros((P, 1))
    B[inmodel] = b[1:]
    B[~inmodel] = b2.conj().T
    SE = np.zeros((P, 1))
    SE[inmodel] = se[1:]
    SE[~inmodel] = s2.conj().T

    #Get P-to-enter or P-to-remove for each term.
    PVAL = np.zeros((P, 1))
    tstat = np.zeros((P, 1))
    if any(inmodel):
        tval = np.divide(B[inmodel], SE[inmodel])
        ptemp = 2 * scipy.stats.t.cdf(-abs(tval), dfe)
        PVAL[inmodel] = ptemp
        tstat[inmodel] = tval
    if any(~inmodel):
        if dfe > 1:
            tval = np.divide(B[~inmodel], SE[~inmodel])
            ptemp = 2 * scipy.stats.t.cdf(-abs(tval), dfe - 1)
            flat_tval = tval.ravel()
            flat_ptemp = ptemp.ravel()
            for i in xrange(flat_tval.size):
                if np.isnan(flat_tval[i]):
                    flat_ptemp[i] = np.NaN
        else:
            tval = np.NaN
            ptemp = np.NaN
        PVAL[~inmodel] = ptemp
        tstat[~inmodel] = tval

    # Compute some summary statistics.
    MSexplained = np.divide(SStotal - SSresid, df0)
    fstat = np.divide(MSexplained, np.dot(rmse, rmse))
    pval = scipy.stats.f.cdf(1. / fstat, dfe, df0)

    # Return summary statistics as a single structure.
    stats = InfoDict()
    stats.source = "stepwisefit"
    stats.dfe = dfe
    stats.df0 = df0
    stats.SStotal = SStotal
    stats.SSresid = SSresid
    stats.fstat = fstat
    stats.pval = pval
    stats.rmse = rmse
    stats.xr = xr
    stats.yr = yr
    stats.B = B
    stats.SE = SE
    stats.TSTAT = tstat
    stats.PVAL = PVAL
    stats.intercept = b[0]

    return B, SE, PVAL, stats

def stepwisefit(allx, y, inmodel = [], penter = 0.05, premove = 0.10,
    display = False, maxiter = np.Inf, keep = [], scale = False):
    """
    Original Source for Documentation (and code reference):
        stepwisefit.m from the MATrix LABoratory statistics toolbox

    Fit regression model using stepwise regression
      B=STEPWISEFIT(X,Y)[0] uses stepwise regression to model the response variable
      Y as a function of the predictor variables represented by the columns
      of the matrix X.  The result B is a vector of estimated coefficient values
      for all columns of X.  The B value for a column not included in the final
      model is the coefficient that would be obtained by adding that column to
      the model.  STEPWISEFIT automatically includes a constant term in all
      models.

      [B,SE,PVAL,INMODEL,STATS,NEXTSTEP,HISTORY]=STEPWISEFIT(...) returns additional
      results.  SE is a vector of standard errors for B.  PVAL is a vector of
      p-values for testing if B is 0.  INMODEL is a logical vector indicating
      which predictors are in the final model.  STATS is a structure containing
      additional statistics.  NEXTSTEP is the recommended next step -- either
      the index of the next predictor to move in or out, or 0 if no further
      steps are recommended.  HISTORY is a structure containing information
      about the history of steps taken.

      [...]=STEPWISEFIT(X,Y,PARAM1=val1,PARAM2=val2,...) specifies one or
      more of the following name/value pairs:

        'inmodel'  A logical vector, or a list of column numbers, indicating which
                   predictors to include in the initial fit (default none)
        'penter'   Max p-value for a predictor to be added (default 0.05)
        'premove'  Min p-value for a predictor to be removed (default 0.10)
        'display'  Either 'on' [True] (default) to display information about each
                   step or 'off' [False] to omit the display
        'maxiter'  Maximum number of steps to take (default is no maximum)
        'keep'     A logical vector, or a list of column numbers, indicating which
                   predictors to keep in their initial state (default none)
        'scale'    Either 'on' [True] to scale each column of X by its standard deviation
                   before fitting, or 'off' [False] (the default) to omit scaling.

      Example:
         load hald
         stepwisefit(ingredients,heat,penter=.08)

      Reference code and documentation copyright 1993-2009 The MathWorks, Inc.
      $Revision: 1.6.4.10 $  $Date: 2009/11/05 17:03:38 $

      [1] Draper, N. R., and H. Smith. Applied Regression Analysis.
            Hoboken, NJ: Wiley-Interscience, 1998. pp. 307-312.
    """

    old_err_settings = np.seterr(divide = 'ignore')

    # Begin Housekeeping

    if maxiter < 0:
        maxiter = np.Inf

    allx = np.asarray(allx)
    assert len(allx.shape) == 2
    p = allx.shape[1]

    y = np.asarray(y)
    assert y.size in y.shape
    assert allx.shape[0] == y.size
    y = y.reshape((y.size, 1))

    wasnan = np.isnan(allx).any(axis = 1) | np.isnan(y)

    inmodel = np.asarray(inmodel)
    if inmodel.size == 0 or inmodel.dtype != bool:
        new_inmodel = np.zeros(p, dtype = bool)
        if inmodel.size > 0:
            new_inmodel[inmodel] = True
        inmodel = new_inmodel
    else:
        assert inmodel.shape == (p,)
        assert inmodel.dtype == bool

    keep = np.asarray(keep)
    if keep.size == 0 or keep.dtype != bool:
        new_keep = np.zeros(p, dtype = bool)
        if keep.size > 0:
            new_keep[keep] = True
        keep = new_keep
    else:
        assert keep.shape == (p,)
        assert keep.dtype == bool

    assert 0 < penter <= premove < 1

    rmse = []
    df0 = []
    inmat = []

    # End Housekeeping

    sx = allx.std(axis = 0, ddof = 1)
    sx[sx == 0] = 1 # All the values must be equal anyway, so change nothing.
    allx = allx / sx # Standardize x values.
    sx = sx.reshape((1, ) + sx.shape)

    if display:
        if not inmodel.any():
            coltext = "None"
        else:
            coltext = repr(list(inmodel.nonzero()[0]))
        print "Initial columns included: %s" % coltext

    jstep = 0
    while True:
        # Perform current fit
        b, se, pval, stats = stepcalc(allx, y, inmodel)
        if not scale:
            # Undo scaling if this was not requested.
            b = b / sx.conj().T
            se = se / sx.conj().T
            stats.b = b
            stats.se = se
            stats.xr = stats.xr * np.tile(sx[:, ~inmodel],
                (stats.xr.shape[0], 1))

        nextstep, pinout = stepnext(inmodel, pval, b, penter, premove, keep)

        if 0 < jstep:
            rmse.append(stats.rmse)
            df0.append(stats.df0)
            inmat.append(inmodel.copy())

        if maxiter <= jstep:
            break
        jstep += 1

        if nextstep == -1:
            break
        elif display:
            addremoved = "removed" if inmodel[nextstep] else "added"
            print "Step %d, %s column %d, p=%f" % \
                (jstep, addremoved, nextstep, pinout)

        inmodel[nextstep] = not inmodel[nextstep]

    if display:
        if not inmodel.any():
            coltext = "None"
        else:
            coltext = repr(list(inmodel.nonzero()[0]))
        print "Final columns include: %s" %coltext
        # Possibly include more debugging information.

    stats.wasnan = wasnan

    history = InfoDict()
    history.rmse = rmse
    history.df0 = df0
    history.inmat = inmat

    np.seterr(**old_err_settings)

    return b, se, pval, inmodel, stats, nextstep, history

