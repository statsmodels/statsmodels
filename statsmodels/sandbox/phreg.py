import numpy as np
from scipy import sparse
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import (cache_readonly,
      resettable_cache)


class PH_SurvivalTime(object):

    def __init__(self, time, status, exog, strata=None, entry=None):
        """
        Represent a collection of survival times with possible
        stratification and left truncation.  Various indexes needed
        for fitting proportional hazards regression models are
        precalculated and used in the PHreg class.

        Parameters
        ----------
        time : array_like
            The times at which either the event (failure) occurs or
            the observation is censored.
        status : array_like
            Indicates whether the event (failure) occurs at `time`
            (`status` is 1), or `time` is a censoring time (`status`
            is 0).
        exog : array_like
            The exogeneous data matrix, cases are rows and variables
            are columns.
        strata : array_like
            Grouping variable defining the strata.  If None, all
            observations are in a single stratum.
        entry : array_like
            Entry (left truncation) times.  The observation is not
            part of the risk set for times before the entry time.  If
            None, the entry time is treated as being zero, which
            corresponds to no left truncation.  The entry time must be
            less than or equal to `time`.

        Notes
        ------
        time, event, strata, entry, and the first dimension of exog
        all must have the same length
        """

        # Default strata
        if strata is None:
            strata = np.zeros(len(time))

        # Default entry times
        if entry is None:
            entry = np.zeros(len(time))

        # Parameter validity checks.
        n1, n2, n3, n4 = len(time), len(status), len(strata),\
            len(entry)
        nv = [n1, n2, n3, n4]
        if max(nv) != min(nv):
            raise ValueError("PHreg: time, status, strata, and " +
                             "entry must all have the same length")
        if min(time) < 0:
            raise ValueError("PHreg: time must be non-negative")
        if min(entry) < 0:
            raise ValueError("PHreg: entry time must be non-negative")
        if np.any(entry > time):
            raise ValueError("PHreg: entry times may not occur " +
                             "after event or censoring times")

        # Get the row indices for the cases in each stratum
        if strata is not None:
            stu = np.unique(strata)
            sth = {x: [] for x in stu}
            for i,k in enumerate(strata):
                sth[k].append(i)
            stratum_rows = [sth[k] for k in stu]
        else:
            stratum_rows = [np.arange(len(time)),]

        # Split everything by stratum
        self.time_s = [time[ix] for ix in stratum_rows]
        self.status_s = [status[ix].astype(np.int32) for ix in stratum_rows]
        self.exog_s = [exog[ix,:] for ix in stratum_rows]
        self.entry_s = [entry[ix] for ix in stratum_rows]

        # Remove strata with no events
        ix = [i for i,x in enumerate(self.status_s) if x.sum() > 0]
        self.time_s = [self.time_s[i] for i in ix]
        self.status_s = [self.status_s[i] for i in ix]
        self.exog_s = [self.exog_s[i] for i in ix]
        self.entry_s = [self.entry_s[i] for i in ix]

        # The number of strata
        nstrat = len(self.time_s)
        self.nstrat = nstrat

        # Remove subjects whose entry time occurs after the last event
        # in their stratum.
        for stx in range(nstrat):
            last_failure = max(self.time_s[stx][self.status_s[stx]==1])
            ii = [i for i,t in enumerate(self.entry_s[stx]) if
                  t < last_failure]
            self.time_s[stx] = self.time_s[stx][ii]
            self.status_s[stx] = self.status_s[stx][ii]
            self.exog_s[stx] = self.exog_s[stx][ii,:]
            self.entry_s[stx] = self.entry_s[stx][ii]

        # Order by time within each stratum
        for stx in range(nstrat):
            ii = np.argsort(self.time_s[stx])
            self.time_s[stx] = self.time_s[stx][ii]
            self.status_s[stx] = self.status_s[stx][ii]
            self.exog_s[stx] = self.exog_s[stx][ii,:]
            self.entry_s[stx] = self.entry_s[stx][ii]

        # Precalculate some indices needed to fit Cox models.
        # Distinct failure times within a stratum are always taken to
        # be sorted in ascending order.
        #
        # ufailt_ix[stx][k] is a list of indices for subjects who fail
        # at the k^th sorted unique failure time in stratum stx
        #
        # risk_enter[stx][k] is a list of indices for subjects who
        # enter the risk set at the k^th sorted unique failure time in
        # stratum stx
        #
        # risk_exit[stx][k] is a list of indices for subjects who exit
        # the risk set at the k^th sorted unique failure time in
        # stratum stx
        self.ufailt_ix, self.risk_enter, self.risk_exit =\
            [], [], []

        for stx in range(self.nstrat):

            # All failure times
            ift = np.flatnonzero(self.status_s[stx] == 1)
            ft = self.time_s[stx][ift]

            # Unique failure times
            uft = np.unique(ft)
            nuft = len(uft)

            # Indices of cases that fail at each unique failure time
            uft_map = {x:i for i,x in enumerate(uft)}
            uft_ix = [[] for k in range(nuft)]
            for ix,ti in zip(ift,ft):
                uft_ix[uft_map[ti]].append(ix)

            # Indices of cases (failed or censored) that enter the
            # risk set at each unique failure time.
            risk_enter1 = [[] for k in range(nuft)]
            for i,t in enumerate(self.time_s[stx]):
                ix = np.searchsorted(uft, t, "right") - 1
                if ix >= 0:
                    risk_enter1[ix].append(i)

            # Indices of cases (failed or censored) that exit the
            # risk set at each unique failure time.
            risk_exit1 = [[] for k in range(nuft)]
            for i,t in enumerate(self.entry_s[stx]):
                ix = np.searchsorted(uft, t)
                risk_exit1[ix].append(i)

            self.ufailt_ix.append(uft_ix)
            self.risk_enter.append(risk_enter1)
            self.risk_exit.append(risk_exit1)



class PHreg(model.LikelihoodModel):
    """Cox proportional hazards regression model."""

    def __init__(self, endog, exog, status=None, entry=None,
                 strata=None, ties='breslow'):
        """
        Fit the Cox proportional hazards regression model for right
        censored data.

        Arguments
        ---------
        endog : array-like
            The observed times
        exog : 2D array-like
            The covariates or exogeneous variables
        status : array-like
            The censoring status values; status=1 indicates that an
            event occured (e.g. failure or death), status=0 indicates
            that the observation was right censored. If None, defaults
            to no censoring.
        entry : array-like
            The entry times, if left truncation occurs
        strata : array-like
            Stratum labels.  If None, all observations are taken to be
            in a single stratum.
        ties : string
            The method used to handle tied times.
        """

        if status is None:
            status = np.ones(len(endog))

        super(PHreg, self).__init__(endog, exog, status=status,
                                    entry=entry, strata=strata)

        self.surv = PH_SurvivalTime(self.endog, self.status,
                                    self.exog, self.strata,
                                    self.entry)

        ties = ties.lower()
        if ties not in ("efron", "breslow"):
            raise ValueError("`ties` must be either `efron` or " +
                             "`breslow`")

        self.ties = ties


    def fit(self, **args):

        rslts = model.LikelihoodModel.fit(self, **args)

        results = PHregResults(self, rslts.params, rslts.cov_params())

        return results


    def loglike(self, b):

        if self.ties == "breslow":
            return self.breslow_loglike(b)
        elif self.ties == "efron":
            return self.efron_loglike(b)


    def score(self, b):

        if self.ties == "breslow":
            return self.breslow_gradient(b)
        elif self.ties == "efron":
            return self.efron_gradient(b)


    def hessian(self, b):

        if self.ties == "breslow":
            return self.breslow_hessian(b)
        else:
            return self.efron_hessian(b)


    def breslow_loglike(self, b):

        surv = self.surv

        like = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            uft_ix = surv.ufailt_ix[stx]
            exog1 = surv.exog_s[stx]
            nuft = len(uft_ix)

            linpred = np.dot(exog1, b)
            linpred -= linpred.max()
            elinpred = np.exp(linpred)

            xp0 = 0.

            # Iterate backward through the unique failure times.
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                xp0 += elinpred[ix].sum()

                # Loop over all cases that fail at this point.
                ix = uft_ix[i]
                like += (linpred[ix] - np.log(xp0)).sum()

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                xp0 -= elinpred[ix].sum()

        return like


    def efron_loglike(self, b):

        surv = self.surv

        like = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            exog1 = surv.exog_s[stx]

            linpred = np.dot(exog1, b)
            linpred -= linpred.max()
            elinpred = np.exp(linpred)

            xp0 = 0.

            # Iterate backward through the unique failure times.
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                xp0 += elinpred[ix].sum()
                xp0f = elinpred[uft_ix[i]].sum()

                # Consider all cases that fail at this point.
                ix = uft_ix[i]
                like += linpred[ix].sum()
                m = len(ix)
                for j in range(m):
                    like -= np.log(xp0 - j*xp0f/float(m))

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                xp0 -= elinpred[ix].sum()

        return like


    def breslow_gradient(self, b):

        surv = self.surv

        grad = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)

            exog1 = surv.exog_s[stx]

            linpred = np.dot(exog1, b)
            linpred -= linpred.max()
            elinpred = np.exp(linpred)

            xp0, xp1 = 0., 0.

            # Iterate backward through the unique failure times.
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                v = exog1[ix,:]
                xp0 += elinpred[ix].sum()
                xp1 += (elinpred[ix][:,None] * v).sum(0)

                # Account for all cases that fail at this point.
                ix = uft_ix[i]
                grad += (exog1[ix,:] - xp1 / xp0).sum(0)

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                v = exog1[ix,:]
                xp0 -= elinpred[ix].sum()
                xp1 -= (elinpred[ix][:,None] * v).sum(0)

        return grad


    def efron_gradient(self, b):

        surv = self.surv

        grad = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            exog1 = surv.exog_s[stx]

            linpred = np.dot(exog1, b)
            linpred -= linpred.max()
            elinpred = np.exp(linpred)

            xp0, xp1 = 0., 0.

            # Iterate backward through the unique failure times.
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                v = exog1[ix,:]
                xp0 += elinpred[ix].sum()
                xp1 += (elinpred[ix][:,None] * v).sum(0)
                ixf = uft_ix[i]
                v = exog1[ixf,:]
                xp0f = elinpred[ixf].sum()
                xp1f = (elinpred[ixf][:,None] * v).sum(0)

                # Consider all cases that fail at this point.
                grad += v.sum(0)
                m = len(ixf)
                for j in range(m):
                    numer = xp1 - j*xp1f/float(m)
                    denom = xp0 - j*xp0f/float(m)
                    grad -= numer / denom

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                v = exog1[ix,:]
                xp0 -= elinpred[ix].sum()
                xp1 -= (elinpred[ix][:,None] * v).sum(0)

        return grad



    def breslow_hessian(self, b):

        surv = self.surv

        hess = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)

            exog1 = surv.exog_s[stx]

            linpred = np.dot(exog1, b)
            linpred -= linpred.max()
            elinpred = np.exp(linpred)

            xp0, xp1, xp2 = 0., 0., 0.

            # Iterate backward through the unique failure times.
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                xp0 += elinpred[ix].sum()
                v = exog1[ix,:]
                xp1 += (elinpred[ix][:,None] * v).sum(0)
                mat = v[None,:,:]
                elx = elinpred[ix]
                xp2 += (mat.T * mat * elx[None,:,None]).sum(1)

                # Account for all cases that fail at this point.
                m = len(uft_ix[i])
                hess += m*(xp2 / xp0  - np.outer(xp1, xp1) / xp0**2)

                # Update for new cases entering the risk set.
                ix = surv.risk_exit[stx][i]
                xp0 -= elinpred[ix].sum()
                v = exog1[ix,:]
                xp1 -= (elinpred[ix][:,None] * v).sum(0)
                mat = v[None,:,:]
                elx = elinpred[ix]
                xp2 -= (mat.T * mat * elx[None,:,None]).sum(1)

        return -hess


    def efron_hessian(self, b):

        surv = self.surv

        hess = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            exog1 = surv.exog_s[stx]

            linpred = np.dot(exog1, b)
            linpred -= linpred.max()
            elinpred = np.exp(linpred)

            xp0, xp1, xp2 = 0., 0., 0.

            # Iterate backward through the unique failure times.
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                xp0 += elinpred[ix].sum()
                v = exog1[ix,:]
                xp1 += (elinpred[ix][:,None] * v).sum(0)
                mat = v[None,:,:]
                elx = elinpred[ix]
                xp2 += (mat.T * mat * elx[None,:,None]).sum(1)
                ixf = uft_ix[i]
                v = exog1[ixf,:]
                xp0f = elinpred[ixf].sum()
                xp1f = (elinpred[ixf][:,None] * v).sum(0)
                mat = v[None,:,:]
                elx = elinpred[ixf]
                xp2f = (mat.T * mat * elx[None,:,None]).sum(1)

                # Account for all cases that fail at this point.
                m = len(uft_ix[i])
                for j in range(m):
                    c0 = xp0 - j*xp0f/float(m)
                    hess += (xp2 - j*xp2f/float(m)) / c0
                    c1 = xp1 - j*xp1f/float(m)
                    hess -= np.outer(c1, c1) / c0**2

                # Update for new cases entering the risk set.
                ix = surv.risk_exit[stx][i]
                xp0 -= elinpred[ix].sum()
                v = exog1[ix,:]
                xp1 -= (elinpred[ix][:,None] * v).sum(0)
                mat = v[None,:,:]
                elx = elinpred[ix]
                xp2 -= (mat.T * mat * elx[None,:,None]).sum(1)

        return -hess


class PHregResults(base.LikelihoodModelResults):
    '''
    Class to contain results of fitting a Cox proportional hazards
    survival model.

    PHregResults inherits from statsmodels.LikelihoodModelResults

    Parameters
    ----------
    See statsmodels.LikelihoodModelResults

    Returns
    -------
    **Attributes**

    model : class instance
        Pointer to PHreg model instance that called fit.
    normalized_cov_params : array
        The sampling covariance matrix of the estimates
    params : array
        The coefficients of the fitted model.  Each coefficient is the
        log hazard ratio corresponding to a 1 unit difference in a
        single covariate while holding the other covariates fixed.
    bse : array
        The standard errors of the fitted parameters.

    See Also
    --------
    statsmodels.LikelihoodModelResults
    '''


    def __init__(self, model, params, cov_params):

        super(PHregResults, self).__init__(model, params,
           normalized_cov_params=cov_params)


    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """Summarize the Regression Results

        Parameters
        -----------
        yname : string, optional
            Default is `y`
        xname : list of strings, optional
            Default is `x#` for ## in p the number of regressors
        title : string, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be
            printed or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results

        """

        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        float_format = "%.3f"
        smry.add_base(results=self, alpha=alpha,
                      float_format=float_format,
                      xname=xname, yname=yname, title=title)

        return smry
